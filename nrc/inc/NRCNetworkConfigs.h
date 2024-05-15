#include "shaders/config.h"
#include "shaders/neural_radiance_caching.h"

#include <tiny-cuda-nn/config.h>
#include <memory>

namespace nrc {
namespace cfg {

	inline const nlohmann::json MODEL_CONFIG_BASE{
		{"loss", {
			{"otype", "RelativeL2Luminance"},
		}},
		// https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#exponential-moving-average-ema
		{"optimizer", {
			{"otype", "EMA"},
			{"decay", 0.99f},
			// {"nested", ...},
		}},
		{"network", {
			{"otype", "FullyFusedMLP"},
			{"activation",        "ReLU"},
			{"output_activation", "ReLU"}, // Output is radiance
			//{"output_activation", "None"},
			{"n_neurons",             64},
			{"n_hidden_layers",        5},
		}},
	};

	nlohmann::json modelConfig(InputEncoding encoding)
	{
		nlohmann::json config = MODEL_CONFIG_BASE;
		
		switch (encoding) {
		case InputEncoding::Frequency:
		{
			config["optimizer"]["nested"] = {
				{"otype", "Adam"},
				{"learning_rate", TRAIN_LR(InputEncoding::Frequency)},
				{"l2_reg", 1e-6f},
			};

			// https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#composite
			// Also See Table 1 of the paper
			config["encoding"] = {
				{"otype", "Composite"},
				{"nested", {
					// Position (3) -> 36
					{
						{"otype", "TriangleWave"},
						{"n_dims_to_encode",   3},
						{"n_frequencies",     12},
					},
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
						{"n_bins",               4},
					},
					// Diffuse, specular albedos (3+3) -> 6
					{
						{"otype", "Identity"},
						{"n_dims_to_encode", 3+3},
					},
				}}
			};
			break;
		}
		case InputEncoding::Hash:
		{
			config["optimizer"]["nested"] = {
				{"otype", "Adam"},
				{"learning_rate", TRAIN_LR(InputEncoding::Hash)},
				{"l2_reg", 1e-6f},
				{"epsilon", 1e-15f},
			};

			config["encoding"] = {
				{"otype", "Composite"},
				{"nested", {
					// Position (3) ->
					{
						{"otype", "HashGrid"},
						{"n_dims_to_encode",        3},
						{"per_level_scale",      2.0f},
						{"log2_hashmap_size",      15},
						{"base_resolution",        16},
						{"n_levels",               16},
						{"n_features_per_level",    2},
					},
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
						{"n_bins",               4},
					},
					// Diffuse, specular albedos (3+3) -> 6
					{
						{"otype", "Identity"},
						{"n_dims_to_encode", 3+3},
					},
				}}
			};
			break;
		}
		default: {
			throw std::invalid_argument{ "Unsupported input encoding" };
		}}

		return config;
	}

}
};