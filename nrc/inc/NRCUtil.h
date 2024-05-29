// Copyright (c) 2024, Jamie Chen <jamiechenang@gmail>

#include "shaders/config.h"
#include "shaders/neural_radiance_caching.h"
#include <cuda.h>
#include <curand.h>

namespace nrc {
    __host__ size_t getRadixSortTempStorageBytes(nrc::ControlBlock &cb, CUstream hStream);

    __host__ void generateRandomPermutationForTrain(nrc::ControlBlock &cb, CUstream hStream,
                                                    curandGenerator_t generator);
};
