#include "config.h"

#include <optix.h>

#include "system_data.h"
#include "neural_radiance_caching.h"
#include "vector_math.h"

// NOTE: This is a separate copy of sys data from the one managed by Optix.
extern "C" __constant__ SystemData sysData;

extern "C" __global__ void placeholder() { return; }

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
#else // DEBUG: directly visualize radiance at terminal veretex
	buffer[index] = make_float4(endRenderRadiance[index], 1.0f);
#endif
}

extern "C" __global__ void propagate_train_radiance(nrc::TrainingSuffixEndVertex *trainSuffixEndVertices, // [:#tiles]
													float3 *endTrainRadiance, // [:#tiles]
													nrc::TrainingRecord *trainRecords, // [65536]
													float3 *trainRadianceTargets) // [65536]
{
	const auto launchIndex = getLaunchIndex2D();
	if (launchIndex.x >= sysData.pf.numTiles.x || launchIndex.y >= sysData.pf.numTiles.y) return;

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
#endif

	while (iTo >= 0)
	{
		//if (launchIndex.x == sysData.pf.numTiles.x / 2 && launchIndex.y == sysData.pf.numTiles.y / 2)
		//	printf("%d->", iTo);

		auto &recordTo   = trainRecords[iTo];
		auto &radianceTo = trainRadianceTargets[iTo];
		
		radianceTo += recordTo.localThroughput * lastRadiance;
		
		// Go to next record
		lastRadiance = radianceTo;
		iTo = recordTo.propTo;

#if 0
		// Zero radiance targets might cause issues in training...?
		if (radianceTo.x + radianceTo.y + radianceTo.z == 0.0f) {
			radianceTo = make_float3(0.001f);
		}
#endif
	}
	//if (launchIndex.x == sysData.pf.numTiles.x / 2 && launchIndex.y == sysData.pf.numTiles.y / 2)
	//	printf("[END]\n");
}

extern "C" __global__ void permute_train_data(nrc::DoubleBuffer<nrc::RadianceQuery> trainRadianceQueries,
											  nrc::DoubleBuffer<float3>             trainRadianceTargets,
											  int                                   *permutation)
{
	const auto launchIndex = getLaunchIndex1D();
	if (launchIndex >= nrc::NUM_TRAINING_RECORDS_PER_FRAME) return;

	// Modulo this to duplicate training data in case RT undersampled.
	const auto numRecords = min(sysData.nrcCB->numTrainingRecords, nrc::NUM_TRAINING_RECORDS_PER_FRAME);
	if (numRecords <= 0) return;

	const auto queriesSrc = trainRadianceQueries.getBuffer(0);
	const auto queriesDst = trainRadianceQueries.getBuffer(1);
	const auto targetsSrc = trainRadianceTargets.getBuffer(0);
	const auto targetsDst = trainRadianceTargets.getBuffer(1);
	
	const auto d = launchIndex;
	const auto s = permutation[d] % numRecords;

	queriesDst[d] = queriesSrc[s];
	targetsDst[d] = targetsSrc[s];
}