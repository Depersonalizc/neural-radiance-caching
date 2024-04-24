/*
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

#include "neural_radiance_caching.h"
#include "per_ray_data.h"
#include "light_definition.h"
#include "shader_common.h"
#include "system_data.h"
#include "transform.h"

extern "C" __constant__ SystemData sysData;


// Take the step along the volume scattering random walk.
__forceinline__ __device__ void stepVolume(PerRayData* thePrd)
{
	// Calculate the new position at the end of the random walk ray segment.
	thePrd->pos += thePrd->wi * thePrd->distance;

	// Change the throughput along the random walk according to the current extinction and the sampled density.
	const float3 transmittance = expf(thePrd->sigma_t * -thePrd->distance);
	const float pdf = dot(thePrd->pdfVolume, thePrd->sigma_t * transmittance);

	thePrd->throughput *= thePrd->stack[thePrd->idxStack].sigma_s * transmittance / pdf;

	// Indicate that the random walk missed.
	thePrd->flags |= FLAG_VOLUME_SCATTERING_MISS;

	// Increment the number of steps done for the random walk
	++thePrd->walk;
}


// Not actually a light. Never appears inside the sysLightDefinitions.
extern "C" __global__ void __miss__env_null()
{
	// Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
	PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

	if (thePrd->flags & FLAG_VOLUME_SCATTERING)
	{
		stepVolume(thePrd);

		return; // Continue the random walk.
	}

	// The null environment adds nothing to the radiance.
	thePrd->eventType = mi::neuraylib::BSDF_EVENT_ABSORB;

	// The ray is terminated early, mask off the queried radiance for rendering.
	const bool isTrainSuffix = thePrd->flags & FLAG_TRAIN_SUFFIX;
	if (!isTrainSuffix) // Either pure rendering, or rendering path of training ray
	{
		thePrd->lastRenderThroughput = make_float3(0.f);
	}

	// Terminate (unbiased) if training.
	const bool isTrain = thePrd->flags & FLAG_TRAIN;
	if (isTrain)
	{
		nrc::endTrainSuffixUnbiased(*thePrd);
	}
}


extern "C" __global__ void __miss__env_constant()
{
	// Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
	PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

	if (thePrd->flags & FLAG_VOLUME_SCATTERING)
	{
		stepVolume(thePrd);

		return; // Continue the random walk.
	}

	// The environment light is always in the first element.
	float3 emission = sysData.lightDefinitions[0].emission; // Constant emission.

	// If the last surface intersection was diffuse or glossy which was directly lit with multiple importance sampling,
	// then calculate implicit light emission with multiple importance sampling as well.
    static constexpr auto BSDF_EVENT_NON_DIRAC = mi::neuraylib::BSDF_EVENT_DIFFUSE 
                                               | mi::neuraylib::BSDF_EVENT_GLOSSY;
    const auto eventWasNonDirac = static_cast<bool>(thePrd->eventType & BSDF_EVENT_NON_DIRAC);
	if (sysData.directLighting && eventWasNonDirac)
	{
		// Note that we don't need to multiply by numLights here because this light 
		// doesn't have to match the previous one sampled for direct lighting estimation
		const float weightMIS = balanceHeuristic(thePrd->pdf, 0.25f * M_1_PIf);
		emission *= weightMIS;
	}

	emission *= thePrd->throughput;

	thePrd->radiance += emission;
	thePrd->eventType = mi::neuraylib::BSDF_EVENT_ABSORB;

	// !! Add the BSDF-sampling part of the MIS to last vertex's target radiance.
    if (thePrd->lastTrainRecordIndex >= 0) [[unlikely]]
    {
        sysData.nrcCB->trainingRadianceTargets[thePrd->lastTrainRecordIndex] += emission;
    }

	// Terminate rendering path if it hasn't
	const bool isTrainSuffix = thePrd->flags & FLAG_TRAIN_SUFFIX;
	if (!isTrainSuffix)
	{
		// Set rendering throughput to zero because the emission has already
		// been accounted for by Direct Lighting. Avoid double counting!
		thePrd->lastRenderThroughput = make_float3(0.f);
	}

	// Terminate (unbiased) if training.
	const bool isTrain = thePrd->flags & FLAG_TRAIN;
	if (isTrain)
	{
		nrc::endTrainSuffixUnbiased(*thePrd);
	}
}


extern "C" __global__ void __miss__env_sphere()
{
	// Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
	PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

	if (thePrd->flags & FLAG_VOLUME_SCATTERING)
	{
		stepVolume(thePrd);

		return; // Continue the random walk.
	}

	// The environment light is always in the first element.
	const LightDefinition& light = sysData.lightDefinitions[0];

	// Transform the ray.direction from world space to light object space.
	const float3 R = transformVector(light.oriInv, thePrd->wi);

	// All lights shine down the positive z-axis in this renderer.
	const float u = (atan2f(-R.x, R.z) + M_PIf) * 0.5f * M_1_PIf;
	// Texture is with origin at lower left, v == 0.0f is south pole.
	const float v = acosf(-R.y) * M_1_PIf;

	float3 emission = make_float3(tex2D<float4>(light.textureEmission, u, v));

	// If the last surface intersection was a diffuse event which was directly lit with multiple importance sampling,
	// then calculate implicit light emission with multiple importance sampling as well.
    static constexpr auto BSDF_EVENT_NON_DIRAC = mi::neuraylib::BSDF_EVENT_DIFFUSE 
                                               | mi::neuraylib::BSDF_EVENT_GLOSSY;
    const auto eventWasNonDirac = static_cast<bool>(thePrd->eventType & BSDF_EVENT_NON_DIRAC);
	if (sysData.directLighting && eventWasNonDirac)
	{
		// For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
		// and not the Gaussian smoothed one used to actually generate the CDFs.
		const float pdfLight = intensity(emission) * light.invIntegral;

		// Note that we don't need to multiply by numLights here because this light 
		// doesn't have to match the previous one sampled for direct lighting estimation
		emission *= balanceHeuristic(thePrd->pdf, pdfLight);
	}

	emission *= (thePrd->throughput * light.emission);

	thePrd->radiance += emission;
	thePrd->eventType = mi::neuraylib::BSDF_EVENT_ABSORB;

	// !! Add the BSDF-sampling part of the MIS to last vertex's target radiance.
    if (thePrd->lastTrainRecordIndex >= 0) [[unlikely]]
    {
        sysData.nrcCB->trainingRadianceTargets[thePrd->lastTrainRecordIndex] += emission;
    }

	// Terminate rendering path if it hasn't
	const bool isTrainSuffix = thePrd->flags & FLAG_TRAIN_SUFFIX;
	if (!isTrainSuffix)
	{
		// Set rendering throughput to zero because the emission has already
		// been accounted for by Direct Lighting. Avoid double counting!
		thePrd->lastRenderThroughput = make_float3(0.f);
	}

	// Terminate (unbiased) if training.
	const bool isTrain = thePrd->flags & FLAG_TRAIN;
	if (isTrain)
	{
		nrc::endTrainSuffixUnbiased(*thePrd);
	}
}
