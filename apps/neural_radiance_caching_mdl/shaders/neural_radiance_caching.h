#pragma once

#ifndef NEURAL_RADIANCE_CACHING_H
#define NEURAL_RADIANCE_CACHING_H

#include <optix.h>

#include "system_data.h"
#include "material_definition_mdl.h"
#include "per_ray_data.h"
#include "shader_common.h"
#include <mi/neuraylib/target_code_types.h>

using Mdl_state = mi::neuraylib::Shading_state_material;

extern "C" __constant__ SystemData sysData;

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
		// Index into ControlBlock::trainingRecords, ::trainingRadianceTargets
		int startTrainRecord/* = TRAIN_RECORD_INDEX_NONE*/;

		// Used to mask off queried radiance for unbiased rays.
		// = 1.0f if self-training, 0.0f if unbiased.
		float radianceMask;
	};

	struct RadianceQuery
	{
		// 16 byte alignment

		// 8 byte alignment
		float2 direction;
		float2 normal;

		// 4 byte alignment
		float3 position;
		float3 diffuse;
		float3 specular;
		float3 roughness;
	};

	struct ControlBlock
	{
		// 16 byte alignment

		// 8 byte alignment
		
		struct StaticBuffers {

		// Training records (vertices) + target radiance
		TrainingRecord *trainingRecords = nullptr; // numTrainingRecords -> 65536, static
		float3 *trainingRadianceTargets = nullptr; // numTrainingRecords -> 65536, static

		// The results of those queries will be used to train the NRC.
		RadianceQuery *radianceQueriesTraining = nullptr; // numTrainingRecords -> 65536, static
		float3 *radianceResultsTraining = nullptr; // numTrainingRecords -> 65536, static
		
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

	// These inlines unfortunately must be put into the hit.cu module, because MDL types cannot be converted to regular floatX here.
#if 0
	__forceinline__ __device__ void addQuery(const Mdl_state& mdlState, 
											 const PerRayData& thePrd, 
											 const mi::neuraylib::Bsdf_auxiliary_data<mi::neuraylib::DF_HSM_NONE> &auxData,
											 /*out:*/ RadianceQuery& radianceQuery)
	{
		radianceQuery.normal    = cartesianToSphericalUnitVector(mdlState.normal);
		radianceQuery.direction = cartesianToSphericalUnitVector(thePrd.wo);
		radianceQuery.position  = mdlState.position;
		radianceQuery.diffuse   = auxData.albedo_diffuse;
		radianceQuery.specular  = auxData.albedo_glossy;
		radianceQuery.roughness = auxData.roughness;
	}

	__forceinline__ __device__ void addRenderQuery(const Mdl_state& mdlState, 
												   const PerRayData& thePrd, 
												   const mi::neuraylib::Bsdf_auxiliary_data<mi::neuraylib::DF_HSM_NONE> &auxData)
	{
		auto& renderQuery = sysData.nrcCB->radianceQueriesInference[thePrd.pixelIndex];
		addQuery(mdlState, thePrd, auxData, renderQuery);
	}

	__forceinline__ __device__ void addTrainQuery(const Mdl_state& mdlState, 
												  const PerRayData& thePrd, 
												  const mi::neuraylib::Bsdf_auxiliary_data<mi::neuraylib::DF_HSM_NONE> &auxData,
												  int trainRecordIndex)

	{
		auto& trainQuery = sysData.nrcCB->radianceQueriesTraining[trainRecordIndex];
		addQuery(mdlState, thePrd, auxData, trainQuery);
	}

	__forceinline__ __device__ void endTrainSuffixSelfTrain(const Mdl_state& mdlState, 
															const PerRayData& thePrd, 
															const mi::neuraylib::Bsdf_auxiliary_data<mi::neuraylib::DF_HSM_NONE> &auxData)
	{
		// Add an inference query at the end vertex of the train suffix.
		auto& query = sysData.nrcCB->radianceQueriesInference[NUM_TRAINING_RECORDS_PER_FRAME + thePrd.tileIndex];
		addQuery(mdlState, thePrd, auxData, query);

		// Add the TrainingSuffixEndVertex
		auto& endVertex = sysData.nrcCB->trainSuffixEndVertices[thePrd.tileIndex];
		endVertex.radianceMask = 1.f; // 1 for self-training: Use the inferenced radiance to initiate propagation.
		endVertex.startTrainRecord = thePrd.lastTrainRecordIndex;
	}
#endif

	__forceinline__ __device__ void endTrainSuffixUnbiased(const PerRayData& thePrd)
	{
		// Just leave the stale query there - we will mask off the inferenced result with endVertex.radianceMask = 0
		//auto& query = sysData.nrcCB->radianceQueriesInference[NUM_TRAINING_RECORDS_PER_FRAME + thePrd.tileIndex];
		//addQuery(mdlState, thePrd, auxData, query);

		// Add the TrainingSuffixEndVertex
		auto& endVertex = sysData.nrcCB->bufDynamic.trainSuffixEndVertices[thePrd.tileIndex];
		endVertex.radianceMask = 0.f; // 0 for unbiased: Don't use the inferenced radiance to initiate propagation.
		endVertex.startTrainRecord = thePrd.lastTrainRecordIndex;
	}
}


#endif