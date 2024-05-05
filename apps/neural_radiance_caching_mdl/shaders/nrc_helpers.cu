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

__forceinline__ __device__ bool allZero(float3 v)
{
	return v.x == 0.f && v.y == 0.f && v.z == 0.f;
}

__forceinline__ __device__ void debug_fill_tile(float3 rgb, uint2 launchIndex)
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

// Radiance accumulator kernel to add the radiance * throughput 
// at the end of the rendering paths to the output buffer.
extern "C" __global__ void accumulate_render_radiance(float3 *endRenderRadiance, float3 *endRenderThroughput)
{
	const auto launchIndex = getLaunchIndex2D();
	if (launchIndex.x >= sysData.resolution.x || launchIndex.y >= sysData.resolution.y) return;
	
	const unsigned int index = launchIndex.y * sysData.resolution.x + launchIndex.x; // Pixel index
	const auto buffer = reinterpret_cast<float4*>(sysData.outputBuffer);
	
#if 0
	const float3 radiance = endRenderThroughput[index] * endRenderRadiance[index];
	const float accWeight = 1.0f / float(sysData.pf.iterationIndex + 1);
	float3 dst = make_float3(buffer[index]);
	dst += (radiance * accWeight);
	buffer[index] = make_float4(dst, 1.0f);
#elif 1 // DEBUG: directly visualize radiance at terminal vertex of rendering path
	if (allZero(endRenderThroughput[index]))
		buffer[index] = { 0.f, 0.f, 0.f, 1.f };
	else
		buffer[index] = make_float4(endRenderRadiance[index], 1.0f);
#elif 0 // DEBUG: visualize radiance * throughput
	float3 dst = endRenderRadiance[index] * endRenderThroughput[index];
	buffer[index] = make_float4(dst, 1.0f);
#elif 0
	buffer[index] = { 0.f, 0.f, 0.f, 1.f };
#else
	return;
#endif
}

extern "C" __global__ void propagate_train_radiance(nrc::TrainingSuffixEndVertex *trainSuffixEndVertices, // [:#tiles]
													float3 *endTrainRadiance, // [:#tiles]
													nrc::TrainingRecord *trainRecords, // [65536]
													float3 *trainRadianceTargets) // [65536]
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

	int iTo = endVert.startTrainRecord;

// Sanity check
#if 1
	if (iTo >= sysData.nrcCB->numTrainingRecords || !(endVert.radianceMask == 0.f || endVert.radianceMask == 1.f))
	{
		printf("[TILE %d/%d] Invalid end vertex: startTrainRecord(int) = %d (/%d). radianceMask(float) = %f\n", 
			   tileIndex+1, sysData.pf.numTiles.x * sysData.pf.numTiles.y, iTo, sysData.nrcCB->numTrainingRecords, endVert.radianceMask);
		return;
	}
	if (endVert.tileIndex != tileIndex)
	{
		printf("ERROR: Tile index mismatch!\n");
		return;
	}
#endif

	// DEBUG: SUPER BLUE if no prop happens
	//if (iTo < 0) { debug_fill_tile({ 0.f, 0.f, 100000.f }, launchIndex); return; }
	
	float3 secondToLastRadiance = lastRadiance;

	while (iTo >= 0)
	{
		secondToLastRadiance = lastRadiance;

		//if (launchIndex.x == sysData.pf.numTiles.x / 2 && launchIndex.y == sysData.pf.numTiles.y / 2)
		//	printf("%d->", iTo);

		auto &recordTo   = trainRecords[iTo];
		auto &radianceTo = trainRadianceTargets[iTo];
		
		radianceTo += recordTo.localThroughput * lastRadiance;

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

extern "C" __global__ void permute_train_data(/*nrc::DoubleBuffer<nrc::RadianceQuery> trainRadianceQueries,
											  nrc::DoubleBuffer<float3>             trainRadianceTargets,*/
											  nrc::RadianceQuery *queriesSrc, nrc::RadianceQuery *queriesDst,
											  float3 *targetsSrc, float3 *targetsDst,
											  int                                   *permutation)
{
	const auto launchIndex = getLaunchIndex1D();
	if (launchIndex >= nrc::NUM_TRAINING_RECORDS_PER_FRAME) return;

	// Modulo this to duplicate training data in case RT undersampled.
	const auto numRecords = min(sysData.nrcCB->numTrainingRecords, nrc::NUM_TRAINING_RECORDS_PER_FRAME);
	if (numRecords <= 0) return;

	//const auto queriesSrc = trainRadianceQueries.getBuffer(0);
	//const auto queriesDst = trainRadianceQueries.getBuffer(1);
	//const auto targetsSrc = trainRadianceTargets.getBuffer(0);
	//const auto targetsDst = trainRadianceTargets.getBuffer(1);
	
	const auto d = launchIndex;
	const auto s = permutation[d] % numRecords;

	queriesDst[d] = queriesSrc[s];
	targetsDst[d] = targetsSrc[s];
}