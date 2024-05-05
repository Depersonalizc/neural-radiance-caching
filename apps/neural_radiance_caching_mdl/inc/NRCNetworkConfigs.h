#include "shaders/config.h"
#include "shaders/neural_radiance_caching.h"

#include <tiny-cuda-nn/config.h>
#include <memory>

namespace nrc {
namespace cfg {

	inline const nlohmann::json MODEL_CONFIG_PAPER{
		{"loss", {
			{"otype", "RelativeL2Luminance"},
			//{"otype", "L2"},
			//{"otype", "L1"},
			//{"otype", "RelativeL1"},
		}},
		// https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#exponential-moving-average-ema
		{"optimizer", {
			{"otype", "EMA"},
			{"decay", 0.99f},
			//{"decay", 0.90f},
			//{"decay", 0.00f},
			{"nested", {
				// https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#adam
				{"otype", "Adam"},
				{"learning_rate", nrc::TRAIN_LEARNING_RATE},
				//{"learning_rate", 0.0f},
				//{"beta1", 0.9f},
				//{"beta2", 0.99f},
				//{"epsilon", 1e-8f},
			}}
		}},
#if 1
		// https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#composite
		// Also See Table 1 of the paper
		{"encoding", {
			{"otype", "Composite"},
			{"nested", {
				// Position (3) -> 36
				{
					{"otype", "TriangleWave"},
					{"n_dims_to_encode", 3},
					{"n_frequencies", 12},
				},
#if !USE_COMPACT_RADIANCE_QUERY
				// Padding (1)
				{
					{"otype", "Identity"},
					{"n_dims_to_encode", 1},
				}
#endif
				// Direction, normal, roughness (2+2+2) -> 24
				{
					{"otype", "OneBlob"},
					{"n_dims_to_encode", 2+2+2},
					//{"n_dims_to_encode", 2+2+1},
					{"n_bins", 4},
				},
				// Diffuse, specular albedos (3+3) -> 6
				{
					{"otype", "Identity"},
					{"n_dims_to_encode", 6},
				},
			}}
		}},
#endif
		{"network", {
			{"otype", "FullyFusedMLP"},
			//{"otype", "CutlassMLP"},
			{"activation", "ReLU"},
			//{"activation", "None"},
			{"output_activation", "ReLU"}, // Output is radiance
			//{"output_activation", "None"}, // Output is radiance
			{"n_neurons", 64},
			{"n_hidden_layers", 5},
		}},
	};

}
};