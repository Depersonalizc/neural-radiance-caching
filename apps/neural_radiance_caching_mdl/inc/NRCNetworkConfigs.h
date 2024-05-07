#include "shaders/config.h"
#include "shaders/neural_radiance_caching.h"

#include <tiny-cuda-nn/config.h>
#include <memory>

namespace nrc {
namespace cfg {

	inline const nlohmann::json MODEL_CONFIG_PAPER{
		{"loss", {
			{"otype", "RelativeL2Luminance"},
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
				//{"beta1", 0.9f},
				//{"beta2", 0.99f},
				{"l2_reg",  1e-6f},
#if USE_HASH_ENCODING
				{"epsilon", 1e-15f},
#endif
			}}
		}},
		// https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#composite
		// Also See Table 1 of the paper
		{"encoding", {
			{"otype", "Composite"},
			{"nested", {
#if USE_HASH_ENCODING
				// Position (3)
				{
					{"otype", "HashGrid"},
					{"n_dims_to_encode", 3},
					{"per_level_scale", 2.0f},
					{"log2_hashmap_size", 15},
					{"base_resolution", 16},
					{"n_levels", 16},
					{"n_features_per_level", 2},
				},
#else
				// Position (3) -> 36
				{
					{"otype", "TriangleWave"},
					{"n_dims_to_encode", 3},
					{"n_frequencies", 12},
				},
#endif
#if !USE_COMPACT_RADIANCE_QUERY
				// Padding (1)
				{
					{"otype", "Identity"},
					{"n_dims_to_encode", 1},
				},
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
		{"network", {
			{"otype", "FullyFusedMLP"},
			{"activation", "ReLU"},
			{"output_activation", "ReLU"}, // Output is radiance
			//{"output_activation", "None"},
			{"n_neurons", 64},
			{"n_hidden_layers", 5},
		}},
	};

}
};