// Copyright (c) 2024, Jamie Chen <jamiechenang@gmail>

#include "inc/NRCUtil.h"

#include <cub/cub.cuh>

__host__ size_t nrc::getRadixSortTempStorageBytes(nrc::ControlBlock& cb, CUstream hStream)
{
	size_t tempStorageBytes;
	auto &bufs = cb.bufStatic;
	cub::DeviceRadixSort::SortPairs(nullptr, tempStorageBytes,
									bufs.randomValues.getBuffer(0), bufs.randomValues.getBuffer(1), 
								    bufs.trainingRecordIndices.getBuffer(0), bufs.trainingRecordIndices.getBuffer(1),
								    nrc::NUM_TRAINING_RECORDS_PER_FRAME, 
									0, sizeof(bufs.randomValues[0]) * 8, static_cast<cudaStream_t>(hStream));
	return tempStorageBytes;
}

__host__ void nrc::generateRandomPermutationForTrain(nrc::ControlBlock& cb, CUstream hStream, curandGenerator_t generator)
{
	auto& bufs = cb.bufStatic;

	// Generate random values. We assume stream has already been set for the generator
	curandGenerate(generator, bufs.randomValues.getBuffer(0), nrc::NUM_TRAINING_RECORDS_PER_FRAME);

	// Given bufs.randomValues, sort as keys to get the permutation as values
	cub::DeviceRadixSort::SortPairs(bufs.tempStorage, bufs.tempStorageBytes,
									/*keys_in*/bufs.randomValues.getBuffer(0), /*keys_out*/bufs.randomValues.getBuffer(1),
									/*vals_in*/bufs.trainingRecordIndices.getBuffer(0), /*vals_out*/bufs.trainingRecordIndices.getBuffer(1),
									nrc::NUM_TRAINING_RECORDS_PER_FRAME, 
									0, sizeof(bufs.randomValues[0]) * 8, static_cast<cudaStream_t>(hStream));		
}