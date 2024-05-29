// Copyright (c) 2024, Jamie Chen <jamiechenang@gmail>

#pragma once

#ifndef NEURAL_RADIANCE_CACHING_H
#define NEURAL_RADIANCE_CACHING_H

#include <optix.h>
#include "vector_math.h"
#include "system_data.h"
#include "per_ray_data.h"

namespace nrc {
    enum class RenderMode : int {
        Full = 0,
        NoCache,
        CacheOnly,
        CacheFirstVertex, // TODO, PERF: dedicated shaders
        // Debug modes
        DebugCacheNoThroughputModulation,
        DebugThroughputOnly,
    };

    enum class InputEncoding : int {
        Frequency = 0,
        Hash,
    };

    constexpr int NUM_BATCHES = 4;
    constexpr int NUM_TRAINING_RECORDS_PER_FRAME = 65536;
    constexpr int BATCH_SIZE = NUM_TRAINING_RECORDS_PER_FRAME / NUM_BATCHES;

#if USE_COMPACT_RADIANCE_QUERY
    // pos(3), dir(2), normal(2), roughness(2), diffuse(3), specular(3)
    constexpr int NN_INPUT_DIMS = 3 + 2 + 2 + 2 + 3 + 3;
    //constexpr int NN_INPUT_DIMS = 3 + 2+2+1 + 3+3;
#else
	// Additional padding float after pos
	constexpr int NN_INPUT_DIMS = 3+1 + 2+2+2 + 3+3;
#endif
    constexpr int NN_OUTPUT_DIMS = 3; // RGB radiance

    constexpr int TRAIN_RECORD_INDEX_NONE = -1; // Indicate primary ray
    constexpr int TRAIN_RECORD_INDEX_BUFFER_FULL = -2; // All secondary rays if buffer is full
    constexpr float TRAIN_UNBIASED_RATIO = 1.f / 16.f;

    constexpr float TRAIN_LR(InputEncoding encoding)
    {
        switch (encoding) {
            case InputEncoding::Frequency: return 1e-3f;
            case InputEncoding::Hash: return 1e-2f;
            default: return 1e-4f;
        }
    }

    // Keep track of the ray path for radiance prop
    struct TrainingRecord {
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

        // DEBUG
        int pixelIndex;
        int tileIndex;
        int propLength;
    };

    // Terminal vertex of a training suffix. Used to initiate a radiance prop.
    struct TrainingSuffixEndVertex {
        // 16 byte alignment

        // 8 byte alignment

        // 4 byte alignment

        // Start of the radiance propagation chain.
        // Index into ControlBlock::trainingRecords to get the train record to prop radiance to
        int startTrainRecord/* = TRAIN_RECORD_INDEX_NONE*/;

        // Used to mask off queried radiance for unbiased rays.
        // = 1.0f if self-training, 0.0f if unbiased.
        float radianceMask;

        // DEBUG
        int pixelIndex;
        int tileIndex;
    };

    struct RadianceQuery {
        float3 position;
#if USE_COMPACT_RADIANCE_QUERY
        // Split float2 to save some space
        float direction1, direction2;
        float normal1, normal2;
        float roughness1, roughness2;
        //float roughness;
#else
		float  pad_;
		float2 direction;
		float2 normal;
		float2 roughness;
#endif
        float3 diffuse;
        float3 specular;

        __device__ __host__ float3 reflectance() { return diffuse + specular; }
    };

    template<typename T>
    struct DoubleBuffer {
        T *data[2]{nullptr, nullptr};

        __forceinline__ __device__ __host__ T * &buffer(int idx) { return data[idx]; }
        __forceinline__ __device__ __host__ T *getBuffer(int idx) { return data[idx]; }

        // operator[] gets an entry in the first buffer
        __forceinline__ __device__ T &operator[](int idx) { return data[0][idx]; }
        __forceinline__ __device__ const T &operator[](int idx) const { return data[0][idx]; }

        // operator() gets an entry in the second buffer
        __forceinline__ __device__ T &operator()(int idx) { return data[1][idx]; }
        __forceinline__ __device__ const T &operator()(int idx) const { return data[1][idx]; }
    };

    struct ControlBlock {
        // 16 byte alignment

        // 8 byte alignment

        struct StaticBuffers {
            // Training records (vertices) + target radiance
            TrainingRecord *trainingRecords = nullptr; // numTrainingRecords -> 65536, static
            DoubleBuffer<float3> trainingRadianceTargets{}; // numTrainingRecords -> 65536, static

            // The results of those queries will be used to train the NRC.
            DoubleBuffer<RadianceQuery> radianceQueriesTraining{}; // numTrainingRecords -> 65536, static

            // Auxiliary buffers for shuffling
            // Use randomValues as keys to sort trainingRecordIndices.buffer(0) - which is just the indices [0..65536)
            // into trainingRecordIndices.buffer(1) - which becomes a permutation for shuffling the training data.
            DoubleBuffer<unsigned int> randomValues{};
            // 65536, static. We don't really need the second buffer (the sorted random values), but cub::DeviceRadixSort demands it.
            DoubleBuffer<int> trainingRecordIndices{};
            // buffer0: [0..65536), buffer1: shuffled indices used to permute radianceQueriesTraining/trainingRadianceTargets.

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
            float3 *radianceResultsInference = nullptr;

            float3 *lastRenderThroughput = nullptr; // #pixels

            // Queries at the first non-specular vertices. Used to visualize the radiance cache as is.
            // #pixels
            RadianceQuery *radianceQueriesCacheVis = nullptr;
            float3 *radianceResultsCacheVis = nullptr;

            // End of all training suffixes
            TrainingSuffixEndVertex *trainSuffixEndVertices = nullptr; // #tiles. Capacity is ~.25*#pixels
        } bufDynamic;

        // 4 byte alignment
        int numTrainingRecords = 0; // Number of training records generated. Updated per-frame
    };

    __forceinline__ __device__ void endTrainSuffixUnbiased(const SystemData &sysData, const PerRayData &thePrd)
    {
        // Just leave the stale query there - we will mask off the inferenced result with endVertex.radianceMask = 0
        //const auto offset = sysData.resolution.x * sysData.resolution.y;
        //auto& query = sysData.nrcCB->radianceQueriesInference[offset + thePrd.tileIndex];
        //addQuery(mdlState, thePrd, auxData, query);

        // Add the TrainingSuffixEndVertex
        auto &endVertex = sysData.nrcCB->bufDynamic.trainSuffixEndVertices[thePrd.tileIndex];
        endVertex.radianceMask = 0.0f; // 0 for unbiased: Don't use the inferenced radiance to initiate propagation.
        endVertex.startTrainRecord = thePrd.lastTrainRecordIndex; // Index into static trainRecords[65536]

        // DEBUG
        //endVertex.pixelIndex = thePrd.pixelIndex;
        //endVertex.tileIndex = thePrd.tileIndex;
    }
}


#endif
