#include "config.h"

#include <optix.h>

#include "system_data.h"
#include "neural_radiance_caching.h"
#include "vector_math.h"

// NOTE: This is a separate copy of sys data from the one managed by Optix.
extern "C" __constant__ SystemData sysData;

namespace {

__forceinline__ __device__ unsigned int getLaunchIndex1D()
{
	return blockDim.x * blockIdx.x + threadIdx.x;
}

__forceinline__ __device__ uint2 getLaunchIndex2D()
{
	return { blockDim.x * blockIdx.x + threadIdx.x, 
			 blockDim.y * blockIdx.y + threadIdx.y };
}

[[maybe_unused]] __forceinline__ __device__ float3 safeDiv(const float3 &a, const float3 &b)
{
	return {
		b.x != 0.f ? a.x / b.x : 0.f,
		b.y != 0.f ? a.y / b.y : 0.f,
		b.z != 0.f ? a.z / b.z : 0.f
	};
}

[[maybe_unused]] __forceinline__ __device__ void debug_fill_tile(const float3 &rgb, const uint2 &launchIndex)
{
	float4* buffer = reinterpret_cast<float4*>(sysData.outputBuffer);
	const int xBase = launchIndex.x * sysData.pf.tileSize.x;
	const int xEnd = xBase + sysData.pf.tileSize.x;
	const int yBase = launchIndex.y * sysData.pf.tileSize.y;
	const int yEnd = yBase + sysData.pf.tileSize.y;
	for (int y = yBase; y < yEnd; y++) {
		for (int x = xBase; x < xEnd; x++) {
			const auto index = y * sysData.resolution.x + x;
			buffer[index] = make_float4(rgb, 1.f);
		}
	}
}

}

// For radiance visualization at first (non-spcular) vertex
extern "C" __global__ void copy_radiance_to_output_buffer(float3 *radiance
#if USE_REFLECTANCE_FACTORING
														, nrc::RadianceQuery *queries
#endif
)
{
	const auto launchIndex = getLaunchIndex2D();
	if (launchIndex.x >= sysData.resolution.x || launchIndex.y >= sysData.resolution.y) return;

	const unsigned int index = launchIndex.y * sysData.resolution.x + launchIndex.x; // Pixel index
	const auto buffer = reinterpret_cast<float4*>(sysData.outputBuffer);

	float3 L = radiance[index];

#if USE_REFLECTANCE_FACTORING
	L *= queries[index].reflectance();
#endif
	
	buffer[index] = make_float4(L, 1.0f);
}

// Radiance accumulator kernel to add the radiance * throughput 
// at the end of the rendering paths to the output buffer.
extern "C" __global__ void accumulate_render_radiance(float3 *endRenderRadiance, 
#if USE_REFLECTANCE_FACTORING
													  nrc::RadianceQuery *endRenderQuery,
#endif
													  float3 *endRenderThroughput, 
													  nrc::RenderMode mode)
{
	const auto launchIndex = getLaunchIndex2D();
	if (launchIndex.x >= sysData.resolution.x || launchIndex.y >= sysData.resolution.y) return;
	
	const unsigned int index = launchIndex.y * sysData.resolution.x + launchIndex.x; // Pixel index
	const auto buffer = reinterpret_cast<float4*>(sysData.outputBuffer);
	
	using namespace nrc;

	switch (mode) {
	case RenderMode::Full: {
		float3 radiance = endRenderThroughput[index] * endRenderRadiance[index];
#if USE_REFLECTANCE_FACTORING
		radiance *= endRenderQuery[index].reflectance();
#endif
		const float accWeight = 1.0f / float(sysData.pf.iterationIndex + 1);
		float3 dst = make_float3(buffer[index]);
		dst += (radiance * accWeight);
		buffer[index] = make_float4(dst, 1.0f);
		break;
	}
	case RenderMode::NoCache: // Ideally this kernel shouldn't be called at all.
	default: {
		return;
	}
	case RenderMode::CacheOnly: {
		float3 radiance = endRenderRadiance[index] * endRenderThroughput[index];
#if USE_REFLECTANCE_FACTORING
		radiance *= endRenderQuery[index].reflectance();
#endif
		buffer[index] = make_float4(radiance, 1.0f);
		break;
	}
	case RenderMode::DebugCacheNoThroughputModulation: {
		float3 radiance = endRenderRadiance[index];
#if USE_REFLECTANCE_FACTORING
		radiance *= endRenderQuery[index].reflectance();
#endif
		buffer[index] = make_float4(radiance, 1.0f);
		break;
	}
	case RenderMode::DebugThroughputOnly: {
		buffer[index] = make_float4(endRenderThroughput[index], 1.0f);
		break;
	}}
}

extern "C" __global__ void propagate_train_radiance(nrc::TrainingSuffixEndVertex *trainSuffixEndVertices, // [:#tiles]
													float3 *endTrainRadiance, // [:#tiles]
#if USE_REFLECTANCE_FACTORING
													nrc::RadianceQuery *endTrainQueries, // [#tiles]
#endif
													nrc::TrainingRecord *trainRecords, // [65536]
													float3 *trainRadianceTargets // [65536]
#if USE_REFLECTANCE_FACTORING
												  , nrc::RadianceQuery* trainRadianceQueries
#endif
)
{
	const auto launchIndex = getLaunchIndex2D();
	if (launchIndex.x >= sysData.pf.numTiles.x || launchIndex.y >= sysData.pf.numTiles.y)
	{
		return;
	}

	// DEBUG coverage test: SUPER RED
	//debug_fill_tile({ 1000000.f, 0.f, 0.f }, launchIndex); return;

	const unsigned int tileIndex = launchIndex.y * sysData.pf.numTiles.x + launchIndex.x; // Tile index
	
	const auto &endVert = trainSuffixEndVertices[tileIndex];
	float3 lastRadiance = endTrainRadiance[tileIndex] * endVert.radianceMask;

#if USE_REFLECTANCE_FACTORING
	// `query` could be stale if unbiased, but in that case 
	// lastRadiance will be masked to 0, so we're good.
	lastRadiance *= endTrainQueries[tileIndex].reflectance();
#endif

	int iTo = endVert.startTrainRecord;

// Sanity check
#if 0
	if (iTo >= sysData.nrcCB->numTrainingRecords || !(endVert.radianceMask == 0.f || endVert.radianceMask == 1.f))
	{
		printf("[TILE %d/%d] Invalid end vertex: startTrainRecord(int) = %d (/%d). radianceMask(float) = %f\n", 
			   tileIndex+1, sysData.pf.numTiles.x * sysData.pf.numTiles.y, iTo, sysData.nrcCB->numTrainingRecords, endVert.radianceMask);
		return;
	}
#endif

	// DEBUG: SUPER BLUE if no prop happens
	//if (iTo < 0) { debug_fill_tile({ 0.f, 0.f, 100000.f }, launchIndex); return; }
	
	//[[maybe_unused]] float3 secondToLastRadiance = lastRadiance;

	while (iTo >= 0)
	{
		//secondToLastRadiance = lastRadiance;

		//if (launchIndex.x == sysData.pf.numTiles.x / 2 && launchIndex.y == sysData.pf.numTiles.y / 2)
		//	printf("%d->", iTo);

		const auto &recordTo = trainRecords[iTo];

		// If USE_REFLECTANCE_FACTORING == 1, equals radiance / reflectance
		// If USE_REFLECTANCE_FACTORING == 0, equals radiance
		auto &targetTo = trainRadianceTargets[iTo];

#if USE_REFLECTANCE_FACTORING
		auto refl = trainRadianceQueries[iTo].reflectance();
		//refl += make_float3(DENOMINATOR_EPSILON);
		float3 radianceTo = targetTo * refl;
#else
		float3 radianceTo = targetTo;
#endif
		
		radianceTo += recordTo.localThroughput * lastRadiance;

#if USE_REFLECTANCE_FACTORING
		//targetTo = radianceTo / refl;
		targetTo = ::safeDiv(radianceTo, refl);
#else
		targetTo = radianceTo;
#endif

		// DEBUG
		//recordTo.pixelIndex = endVert.pixelIndex;
		//recordTo.tileIndex = endVert.tileIndex;
		
		// Go to next record
		lastRadiance = radianceTo;
		iTo = recordTo.propTo;

		// DEBUG: Visualize the radiance propped all the way to the first hit vertex
		//if (iTo < 0) debug_fill_tile(lastRadiance, launchIndex);
	}
	//if (launchIndex.x == sysData.pf.numTiles.x / 2 && launchIndex.y == sysData.pf.numTiles.y / 2)
	//	printf("[END]\n");

	// DEBUG: second to last
	//debug_fill_tile(secondToLastRadiance, launchIndex);
}

extern "C" __global__ void permute_train_data(nrc::DoubleBuffer<nrc::RadianceQuery> trainRadianceQueries,
											  nrc::DoubleBuffer<float3>             trainRadianceTargets,
											  //nrc::RadianceQuery *queriesSrc, nrc::RadianceQuery *queriesDst,
											  //float3 *targetsSrc, float3 *targetsDst,
											  int                                   *permutation)
{
	const auto launchIndex = getLaunchIndex1D();
	if (launchIndex >= nrc::NUM_TRAINING_RECORDS_PER_FRAME) return;

	// Modulo this to duplicate training data in case RT undersampled.
	const auto numRecords = min(sysData.nrcCB->numTrainingRecords, nrc::NUM_TRAINING_RECORDS_PER_FRAME);
	if (numRecords <= 0) [[unlikely]] return;

	const auto queriesSrc = trainRadianceQueries.getBuffer(0);
	const auto queriesDst = trainRadianceQueries.getBuffer(1);
	const auto targetsSrc = trainRadianceTargets.getBuffer(0);
	const auto targetsDst = trainRadianceTargets.getBuffer(1);
	
	const auto d = launchIndex;
	const auto s = permutation[d] % numRecords;

	queriesDst[d] = queriesSrc[s];
	targetsDst[d] = targetsSrc[s];
}