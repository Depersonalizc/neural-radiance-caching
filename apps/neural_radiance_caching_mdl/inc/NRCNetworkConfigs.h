#include "shaders/config.h"
#include <tiny-cuda-nn/config.h>

#include <memory>

namespace nrc {
namespace cfg {

	inline const nlohmann::json MODEL_CONFIG_PAPER{
		{"loss", {
			{"otype", "RelativeL2Luminance"},
			//{"otype", "L2"},
			//{"otype", "L1"},
		}},
			//{"otype", "RelativeL1"},
		// https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#exponential-moving-average-ema
		{"optimizer", {
			{"otype", "EMA"},
			//{"decay", 0.99f},
			{"decay", 0.90f},
			{"nested", {
				// https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#adam
				{"otype", "Adam"},
				{"learning_rate", 1e-2f},
			}}
		}},
#if 1
		// https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#composite
		// Also See Table 1 of the paper
		{"encoding", {
			{"otype", "Composite"},
			{"nested", {
				// Position (3)
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
				// Direction, normal, roughness (2+2+2)
				{
					{"otype", "OneBlob"},
					{"n_dims_to_encode", 2+2+2},
					{"n_bins", 4},
				},
				// Diffuse, specular albedos (3+3)
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