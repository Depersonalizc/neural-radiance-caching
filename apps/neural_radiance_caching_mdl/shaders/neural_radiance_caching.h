#pragma once

#ifndef NEURAL_RADIANCE_CACHING_H
#define NEURAL_RADIANCE_CACHING_H

namespace nrc
{
	constexpr int NUM_BATCHES = 4;
	constexpr int NUM_TRAINING_RECORDS_PER_FRAME = 65536;

	constexpr int TRAIN_RECORD_INDEX_NONE = -1; // Indicate primary ray
	constexpr int TRAIN_RECORD_INDEX_BUFFER_FULL = -2; // All secondary rays if buffer is full
	constexpr float TRAIN_UNBIASED_RATIO = 1.f / 16.f;

	// Keep track of the ray path for radiance prop
	struct TrainingRecord
	{
		// 16 byte alignment

		// 8 byte alignment

		// 4 byte alignment
		int propTo/* = TRAIN_RECORD_INDEX_NONE*/; // Link to next training record in the direction of radiance prop.
		
		// Used to modulate radiance prop'd from *previous* record.
		float3 localThroughput; 

		// A radiance prop looks like this: (if propTo >= 0)
		// propFrom = index of this TrainingRecord
		// const auto &nextRec = trainingRecords[propTo];
		// trainingRadianceTargets[propTo] += nextRec.localThroughput * trainingRadianceTargets[propFrom];
	};

	// Terminal vertex of a training suffix. Used to initiate a radiance prop.
	struct TrainingSuffixEndVertex
	{
		// 16 byte alignment

		// 8 byte alignment

		// 4 byte alignment

		// Start of the radiance propagation chain. 
		// Index into ControlBlock::trainingRecords to get the train record to prop radiance to
		int startTrainRecord/* = TRAIN_RECORD_INDEX_NONE*/;

		// Used to mask off queried radiance for unbiased rays.
		// = 1.0f if self-training, 0.0f if unbiased.
		float radianceMask;
	};

	struct RadianceQuery
	{
		float3 position;

		float2 direction;
		float2 normal;
		float2 roughness;

		float3 diffuse;
		float3 specular;
	};

	template<typename T>
	struct DoubleBuffer
	{
		T* data[2]{ nullptr, nullptr };

		__forceinline__ __device__ __host__ T* & buffer(int idx) { return data[idx]; }
		__forceinline__ __device__ __host__ T* getBuffer(int idx) { return data[idx]; }
		
		// operator[] gets an entry in the first buffer
		__forceinline__ __device__ T& operator[](int idx) { return data[0][idx]; }
		__forceinline__ __device__ const T& operator[](int idx) const { return data[0][idx]; }

		// operator() gets an entry in the second buffer
		__forceinline__ __device__ T& operator()(int idx) { return data[1][idx]; }
		__forceinline__ __device__ const T& operator()(int idx) const { return data[1][idx]; }

		// Copy data from the first buffer to the second buffer
		//__host__ void copy0To1Async(int numElems, CUstream hStream)
		//{
		//	MY_ASSERT(data[0] && data[1]);
		//	CU_CHECK(cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(data[1]), reinterpret_cast<CUdeviceptr>(data[0]), numElems * sizeof(T), hStream));
		//}
	};

	struct ControlBlock
	{
		// 16 byte alignment

		// 8 byte alignment
		
		struct StaticBuffers {

		// Training records (vertices) + target radiance
		TrainingRecord *trainingRecords = nullptr; // numTrainingRecords -> 65536, static
		DoubleBuffer<float3> trainingRadianceTargets{}; // numTrainingRecords -> 65536, static

		// The results of those queries will be used to train the NRC.
		DoubleBuffer<RadianceQuery> radianceQueriesTraining{}; // numTrainingRecords -> 65536, static
		float3 *radianceResultsTraining = nullptr; // numTrainingRecords -> 65536, static

		// Auxiliary buffers for shuffling
		// Use randomValues as keys to sort trainingRecordIndices.buffer(0) - which is just the indices [0..65536)
		// into trainingRecordIndices.buffer(1) - which becomes a permutation for shuffling the training data.
		DoubleBuffer<unsigned int> randomValues{}; // 65536, static. We don't really need the second buffer (the sorted random values), but cub::DeviceRadixSort demands it.
		DoubleBuffer<int> trainingRecordIndices{}; // buffer0: [0..65536), buffer1: shuffled indices used to permute radianceQueriesTraining/trainingRadianceTargets.
		
		void *tempStorage = nullptr; // Temporary storage for cub::DeviceRadixSort
		size_t tempStorageBytes = 0;
		} bufStatic;

		struct DynamicBuffers {

		// Points to a dynamic array of (#pixels + #tiles) randiance queries. Note the #tiles is dynamic each frame.
		// Capacity is (#pixels + #2x2-tiles ~= 1.25*#pixels). Re-allocate when the render resolution changes.
		//
		// The first #pixels queries are at the end of short rendering paths, in the flattened order of pixels.
		// -- Results (potentially) used for rendering (Remember to modulate with `lastRenderingThroughput`)
		// -- For rays that are terminated early by missing into envmap, the RadianceQuery contains garbage inputs. For convenience
		//    we still query but the results won't get accumulated into the pixel buffer, since `lastRenderingThroughput` should be 0.
		// 
		// The following #tiles queries are at the end of training suffixes, in the flattened order of tiles.
		// -- Results (potentially) used for initiating radiance propagation in self-training.
		// -- For unbiased training rays, the RadianceQuery contains garbage inputs. For convenience we still query
		//    but the results won't be used to initiate radiance propagation, as indicated by the corresponding TrainTerminalVertex.
		RadianceQuery *radianceQueriesInference = nullptr;
		float3        *radianceResultsInference = nullptr;

		float3 *lastRenderThroughput = nullptr; // #pixels

		// End of all training suffixes
		TrainingSuffixEndVertex *trainSuffixEndVertices = nullptr; // #tiles. Capacity is ~.25*#pixels
		
		} bufDynamic;

		// 4 byte alignment
		int numTrainingRecords = 0;   // Number of training records generated. Upated per-frame
		
		//int maxNumTrainingRecords = NUM_TRAINING_RECORDS_PER_FRAME;
	};

	//__forceinline__ __device__ void endTrainSuffixUnbiased(const PerRayData& thePrd)
	//{
	//	// Just leave the stale query there - we will mask off the inferenced result with endVertex.radianceMask = 0
	//	//auto& query = sysData.nrcCB->radianceQueriesInference[NUM_TRAINING_RECORDS_PER_FRAME + thePrd.tileIndex];
	//	//addQuery(mdlState, thePrd, auxData, query);
	//	// Add the TrainingSuffixEndVertex
	//	auto& endVertex = sysData.nrcCB->bufDynamic.trainSuffixEndVertices[thePrd.tileIndex];
	//	endVertex.radianceMask = 0.0f; // 0 for unbiased: Don't use the inferenced radiance to initiate propagation.
	//	endVertex.startTrainRecord = thePrd.lastTrainRecordIndex;
	//}
}


#endif