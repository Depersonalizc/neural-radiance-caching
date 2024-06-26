/*
 * Copyright (c) 2024, Jamie Chen <jamiechenang@gmail>, modified upon 
 * Copyright (c) 2013-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "config.h"

#include <optix.h>

#include "system_data.h"
#include "neural_radiance_caching.h"
#include "per_ray_data.h"
#include "shader_common.h"
#include "half_common.h"
#include "random_number_generators.h"


extern "C" __constant__ SystemData sysData;

__forceinline__ __device__ float3 safe_div(const float3 &a, const float3 &b)
{
    const float x = (b.x != 0.0f) ? a.x / b.x : 0.0f;
    const float y = (b.y != 0.0f) ? a.y / b.y : 0.0f;
    const float z = (b.z != 0.0f) ? a.z / b.z : 0.0f;

    return make_float3(x, y, z);
}

__forceinline__ __device__ float sampleDensity(const float3 &albedo,
                                               const float3 &throughput,
                                               const float3 &sigma_t,
                                               const float u,
                                               float3 &pdf)
{
    const float3 weights = throughput * albedo;

    const float sum = weights.x + weights.y + weights.z;

    pdf = (0.0f < sum) ? weights / sum : make_float3(1.0f / 3.0f);

    if (u < pdf.x) {
        return sigma_t.x;
    }
    if (u < pdf.x + pdf.y) {
        return sigma_t.y;
    }
    return sigma_t.z;
}

// Determine Henyey-Greenstein phase function cos(theta) of scattering direction
__forceinline__ __device__ float sampleHenyeyGreensteinCos(const float xi, const float g)
{
    // PBRT v3: Chapter 15.2.3
    if (fabsf(g) < 1e-3f) // Isotropic.
    {
        return 1.0f - 2.0f * xi;
    }

    const float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi);
    return (1.0f + g * g - s * s) / (2.0f * g);
}

// Determine scatter reflection direction with Henyey-Greenstein phase function.
__forceinline__ __device__ void sampleVolumeScattering(const float2 xi, const float g, float3 &dir)
{
    const float cost = sampleHenyeyGreensteinCos(xi.x, g);

    float sint = 1.0f - cost * cost;
    sint = (0.0f < sint) ? sqrtf(sint) : 0.0f;

    const float phi = 2.0f * M_PIf * xi.y;

    // This vector is oriented in its own local coordinate system:
    const float3 d = make_float3(cosf(phi) * sint, sinf(phi) * sint, cost);

    // Align the vector with the incoming direction.
    const TBN tbn(dir); // Just some ortho-normal basis along dir as z-axis.

    dir = tbn.transformToWorld(d);
}

namespace {
    __forceinline__ __device__ int tileIndex(const uint2 &launchIndex)
    {
        const auto tileIndexX = launchIndex.x / sysData.pf.tileSize.x;
        const auto tileIndexY = launchIndex.y / sysData.pf.tileSize.y;

        return tileIndexY * sysData.pf.numTiles.x + tileIndexX;
    }

    __forceinline__ __device__ bool isBoundaryRay(const uint2 &launchIndex)
    {
        const auto xEnd = sysData.pf.numTiles.x * sysData.pf.tileSize.x;
        const auto yEnd = sysData.pf.numTiles.y * sysData.pf.tileSize.y;

        return (launchIndex.x >= xEnd || launchIndex.y >= yEnd);
    }

    __forceinline__ __device__ bool isTrainingRay(const uint2 &launchIndex)
    {
        // Discard boundary tile
        if (isBoundaryRay(launchIndex)) {
            return false;
        }

        // Compute the local index within tile
        const auto xLocal = launchIndex.x % sysData.pf.tileSize.x;
        const auto yLocal = launchIndex.y % sysData.pf.tileSize.y;
        const auto idxLocal = yLocal * sysData.pf.tileSize.x + xLocal;

        return idxLocal == sysData.pf.tileTrainingIndex;
    }

#define VOLUME_RENDER 1
    __forceinline__ __device__ float3 nrcIntegrator(PerRayData &prd)
    {
        // The integrator starts with black radiance and full path throughput.
        prd.radiance = make_float3(0.0f);
        prd.pdf = 0.0f;
        prd.throughput = make_float3(1.0f);
        prd.flags &= FLAG_MASK_PERSISTENT; // Clear trasient flags
        prd.sigma_t = make_float3(0.0f); // Extinction coefficient: sigma_a + sigma_s.
        prd.walk = 0; // Number of random walk steps taken through volume scattering.
        prd.eventType = mi::neuraylib::BSDF_EVENT_ABSORB; // Initialize for exit. (Otherwise miss programs do not work.)

        prd.areaThreshold = INFINITY;
        prd.areaSpread = 0.0f;
        prd.lastTrainRecordIndex = nrc::TRAIN_RECORD_INDEX_NONE;
        prd.lastRenderThroughput = make_float3(0.0f);
#if 0
        prd.lastRenderThroughput = float3{ 0.0f, 1000000.0f, 0.0f }; // super green
#endif

        // Nested material handling.
        // Small stack of MATERIAL_STACK_SIZE = 4 entries of which the first is vacuum.
        prd.idxStack = 0;
        prd.stack[0].ior = make_float3(1.0f); // No effective IOR.
        prd.stack[0].sigma_a = make_float3(0.0f); // No volume absorption.
        prd.stack[0].sigma_s = make_float3(0.0f); // No volume scattering.
        prd.stack[0].bias = 0.0f; // Isotropic volume scattering.

        // Put payload pointer into two unsigned integers. Actually const, but that's not what optixTrace() expects.
        uint2 payload = splitPointer(&prd);

        // Russian Roulette path termination after a specified number of bounces needs the current depth.
        //int depth = 0; // Path segment index. Primary ray is depth == 0.
        for (int depth = 0;; depth++) {
            // Self-intersection avoidance:
            // Offset the ray t_min value by sysData.sceneEpsilon when a geometric primitive was hit by the previous ray.
            // Primary rays and volume scattering miss events will not offset the ray t_min.
            const float epsilon = (prd.flags & FLAG_HIT) ? sysData.sceneEpsilon : 0.0f;

            prd.wo = -prd.wi; // Direction to observer.
            prd.distance = RT_DEFAULT_MAX; // Shoot the next ray with maximum length.
            prd.flags &= FLAG_MASK_PERSISTENT; // reset transient flags
            prd.depth = depth;

            // Special cases for volume scattering!
            // Set ray FLAG_VOLUME_SCATTERING and optionally tmax
#if VOLUME_RENDER
            if (prd.idxStack > 0) // Inside a volume?
            {
                // Note that this only supports homogeneous volumes so far!
                // No change in sigma_s along the random walk here.
                const float3 &sigma_s = prd.stack[prd.idxStack].sigma_s;

                if (isNotNull(sigma_s)) // We're inside a volume and it has volume scattering?
                {
                    // Indicate that we're inside a random walk. This changes the behavior of the miss programs.
                    prd.flags |= FLAG_VOLUME_SCATTERING;

                    // Random walk through scattering volume, sampling the distance.
                    // Note that the entry and exit of the volume is done according to the BSDF sampling.
                    // Means glass with volume scattering will still do the proper refractions.
                    // When the number of random walk steps has been exceeded, the next ray is shot with distance RT_DEFAULT_MAX
                    // to hit something. If that results in a transmission the scattering volume is left.
                    // If not, this continues until the maximum path length has been exceeded.
                    if (prd.walk < sysData.walkLength) {
                        const float3 albedo = safe_div(sigma_s, prd.sigma_t);
                        const float2 xi = rng2(prd.seed);

                        const float s = sampleDensity(albedo, prd.throughput, prd.sigma_t, xi.x, prd.pdfVolume);

                        // Prevent logf(0.0f) by sampling the inverse range (0.0f, 1.0f].
                        prd.distance = -logf(1.0f - xi.y) / s;
                    }
                }
            }
#endif

            // Note that the primary rays and volume scattering miss cases do not offset the ray t_min by sysSceneEpsilon.
            // Updated: prd.radiance, .throughput, .flags, .eventType (for BSDF importance sampling)
            //             .pos, .distance
            //             .wi (next ray)
            //             .stack, .idxStack
            optixTrace(sysData.topObject,
                       prd.pos, prd.wi, // origin, direction
                       epsilon, prd.distance, 0.0f, // tmin, tmax, time
                       OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_NONE,
                       TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
                       payload.x, payload.y);

#if (USE_SHADER_EXECUTION_REORDERING == 1 && OPTIX_VERSION >= 80000) // OptiX Shader Execution Reordering (SER) implementation.

            unsigned int hint = 0; // miss uses some default value. The record type itself will distinguish this case.
            if (optixHitObjectIsHit()) {
                const int idMaterial = sysData.geometryInstanceData[optixHitObjectGetInstanceId()].ids.x;
                hint = sysData.materialDefinitionsMDL[idMaterial].indexShader; // Shader configuration only.
            }
            optixReorder(hint, sysData.numBitsShaders);

            optixInvoke(payload.x, payload.y);
#endif

            // Path termination by miss shader or sample() routines.
            //if (prd.eventType == mi::neuraylib::BSDF_EVENT_ABSORB || isNull(prd.throughput))
            if (prd.eventType == mi::neuraylib::BSDF_EVENT_ABSORB) {
                break;
            }

            // Unbiased Russian Roulette path termination.
            static constexpr auto FLAG_TRAIN_UNBIASED_SUFFIX = FLAG_TRAIN_UNBIASED | FLAG_TRAIN_SUFFIX;
            const bool isUnbiasedSuffix = (prd.flags & FLAG_TRAIN_UNBIASED_SUFFIX) == FLAG_TRAIN_UNBIASED_SUFFIX;
            const bool doRR = isUnbiasedSuffix && (depth >= sysData.pathLengths.x);
            if (doRR) {
                const float probability = max(fmaxf(prd.throughput), 0.005f);

                if (probability < rng(prd.seed)) // Paths with lower probability to continue are terminated earlier.
                {
                    // End the train suffix by a zero-radiance unbiased
                    // terminal vertex that links to thePrd->lastTrainRecordIndex.
                    nrc::endTrainSuffixUnbiased(sysData, prd);
                    break;
                }

                prd.throughput /= probability;
                // Path isn't terminated. Adjust the throughput so that the average is right again.
            }

#if VOLUME_RENDER
            // We're inside a volume and the scatter ray missed.
            if (prd.flags & FLAG_VOLUME_SCATTERING_MISS) // This implies FLAG_VOLUME_SCATTERING.
            {
                // Random walk through scattering volume, sampling the direction according to the phase function.
                sampleVolumeScattering(rng2(prd.seed), prd.stack[prd.idxStack].bias, prd.wi);
            }
#endif

            // Max #bounces exceeded
            if (depth >= sysData.pathLengths.y) {
                prd.lastRenderThroughput = make_float3(0.f);

                // Terminate training suffix
                if (prd.flags & FLAG_TRAIN) {
                    // End the train suffix by a zero-radiance unbiased
                    // terminal vertex that links to thePrd->lastTrainRecordIndex.
                    nrc::endTrainSuffixUnbiased(sysData, prd);
                }

                break;
            }
        }

        return prd.radiance;
    }
}

extern "C" __global__ void __raygen__nrc_path_tracer()
{
#if USE_TIME_VIEW
        clock_t clockBegin = clock();
#endif

    const uint2 theLaunchDim = make_uint2(optixGetLaunchDimensions());
    const uint2 theLaunchIndex = make_uint2(optixGetLaunchIndex());

    PerRayData prd;

    // Initialize the random number generator seed from the linear pixel index and the iteration index.
    const unsigned int index = theLaunchDim.x * theLaunchIndex.y + theLaunchIndex.x;
    prd.seed = tea<4>(index, sysData.pf.totalSubframeIndex);
    // PERF This template really generates a lot of instructions.

    // Decoupling the pixel coordinates from the screen size will allow for partial rendering algorithms.
    // Resolution is the actual full rendering resolution and for the single GPU strategy, theLaunchDim == resolution.
    const float2 screen = make_float2(sysData.resolution); // == theLaunchDim for rendering strategy RS_SINGLE_GPU.
    const float2 pixel = make_float2(theLaunchIndex);
    const float2 sample = rng2(prd.seed);

    // Set ray flags
    prd.flags = 0;

#if 1
    const bool isDebug = (theLaunchIndex.x == theLaunchDim.x / 2)
                         && (theLaunchIndex.y == theLaunchDim.y / 2)
                         && (sysData.pf.iterationIndex == 0);
    if (isDebug) {
        prd.flags |= FLAG_DEBUG;
    }
#endif

    const bool isTrain = ::isTrainingRay(theLaunchIndex);
    if (isTrain) {
        prd.flags |= FLAG_TRAIN;

        // Set about 1/16 of training ray to be unbiased (terminated with RR)
        if (rng(prd.seed) < sysData.pf.nrcTrainUnbiasedRatio/*nrc::TRAIN_UNBIASED_RATIO*/) {
            prd.flags |= FLAG_TRAIN_UNBIASED;
        }

        // Record tile index.
        prd.tileIndex = ::tileIndex(theLaunchIndex);
    }

    // DEBUG INFO
#if 0
        if (isDebug)
        {
                printf("Tile size: (%d, %d), train index: %d\n",
                        sysData.pf.tileSize.x, sysData.pf.tileSize.y, sysData.pf.tileTrainingIndex);

                printf("#Training records: %d, Max #records allowed: %d\n",
                        sysData.nrcCB->numTrainingRecords, nrc::NUM_TRAINING_RECORDS_PER_FRAME/*sysData.nrcCB->maxNumTrainingRecords*/);
        }
#endif

    // Lens shaders
    const LensRay ray = optixDirectCall<LensRay>(sysData.typeLens, screen, pixel, sample);
    prd.pos = ray.org;
    prd.wi = ray.dir;

    prd.pixelIndex = index;

    // Zero out the cache vis query
    sysData.nrcCB->bufDynamic.radianceQueriesCacheVis[index] = {};

    // Perform the ray tracing
    float3 radiance = ::nrcIntegrator(prd);

    // Record the last throughput.
    float3 *lastThroughputBuffer = sysData.nrcCB->bufDynamic.lastRenderThroughput;
    lastThroughputBuffer[index] = prd.lastRenderThroughput;

#if USE_DEBUG_EXCEPTIONS
        // DEBUG Highlight numerical errors.
        if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
        {
                radiance = make_float3(1000000.0f, 0.0f, 0.0f); // super red
        }
        else if (isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z))
        {
                radiance = make_float3(0.0f, 1000000.0f, 0.0f); // super green
        }
        else if (radiance.x < 0.0f || radiance.y < 0.0f || radiance.z < 0.0f)
        {
                radiance = make_float3(0.0f, 0.0f, 1000000.0f); // super blue
        }
#else
    // NaN values will never go away. Filter them out before they can arrive in the output buffer.
    // This only has an effect if the debug coloring above is off!
    if (!(isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z)))
#endif
    {
#if USE_FP32_OUTPUT

        float4 *buffer = reinterpret_cast<float4 *>(sysData.outputBuffer);

#if USE_TIME_VIEW
                clock_t clockEnd = clock();
                const float alpha = (clockEnd - clockBegin) * sysData.clockScale;

                float4 result = make_float4(radiance, alpha);

                if (sysData.pf.iterationIndex > 0)
                {
                        const float4 dst = buffer[index]; // RGBA32F

                        result = lerp(dst, result, 1.0f / float(sysData.pf.iterationIndex + 1)); // Accumulate the alpha as well.
                }
                buffer[index] = result;
#else // if !USE_TIME_VIEW
        if (sysData.pf.iterationIndex > 0) {
            const float3 dst = make_float3(buffer[index]); // RGB24F
            radiance = lerp(dst, radiance, 1.0f / float(sysData.pf.iterationIndex + 1));
            // Only accumulate the radiance, alpha stays 1.0f.
        }
        buffer[index] = make_float4(radiance, 1.0f);

#endif // USE_TIME_VIEW

#else // if !USE_FP32_OUPUT

                Half4* buffer = reinterpret_cast<Half4*>(sysData.outputBuffer);

#if USE_TIME_VIEW
                clock_t clockEnd = clock();
                float alpha = (clockEnd - clockBegin) * sysData.clockScale;

                if (sysData.pf.iterationIndex > 0)
                {
                        const float t = 1.0f / float(sysData.pf.iterationIndex + 1);

                        const Half4 dst = buffer[index]; // RGBA16F

                        radiance.x = lerp(__half2float(dst.x), radiance.x, t);
                        radiance.y = lerp(__half2float(dst.y), radiance.y, t);
                        radiance.z = lerp(__half2float(dst.z), radiance.z, t);
                        alpha = lerp(__half2float(dst.z), alpha, t);
                }
                buffer[index] = make_Half4(radiance, alpha);
#else // if !USE_TIME_VIEW
                if (sysData.pf.iterationIndex > 0)
                {
                        const float t = 1.0f / float(sysData.pf.iterationIndex + 1);

                        const Half4 dst = buffer[index]; // RGBA16F

                        radiance.x = lerp(__half2float(dst.x), radiance.x, t);
                        radiance.y = lerp(__half2float(dst.y), radiance.y, t);
                        radiance.z = lerp(__half2float(dst.z), radiance.z, t);
                }
                buffer[index] = make_Half4(radiance, 1.0f);
#endif // USE_TIME_VIEW

#endif // USE_FP32_OUTPUT
    }

    // DEBUG INFO
#if 0
        if (isDebug)
        {
                printf("Ray tracing iteration done. Last render throughput: (%f, %f, %f)\n",
                        prd.lastRenderThroughput.x, prd.lastRenderThroughput.y, prd.lastRenderThroughput.z);
        }
#endif

    // DEBUG VIS (prd.lastRenderThroughput)
#if 0
        float4* buffer = reinterpret_cast<float4*>(sysData.outputBuffer);
        buffer[index] = make_float4(prd.lastRenderThroughput, 1.0f);
#endif

    // DEBUG VIS
#if 0
        if (isTrain)
        {
                auto buffer = reinterpret_cast<float4*>(sysData.outputBuffer);
                buffer[index] = (prd.flags & FLAG_TRAIN_UNBIASED)
                                          ? float4{ 0.0f, 1000000.0f, 0.0f, 1.0f }  // super green for unbiased training
                                          : float4{ 1000000.0f, 0.0f, 0.0f, 1.0f }; // super red for self-training ray
                return;
        }
#endif
}
