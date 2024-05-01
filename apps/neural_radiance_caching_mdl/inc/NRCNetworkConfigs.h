#include <tiny-cuda-nn/config.h>

#include <memory>

namespace nrc {
namespace cfg {

	// pos(3), dir(2), normal(2), roughness(2), diffuse(3), specular(3)
	inline constexpr auto INPUT_DIMS  = 3 + 2+2+2 + 3+3;
	inline constexpr auto OUTPUT_DIMS = 3; // RGB radiance

	inline const nlohmann::json MODEL_CONFIG_PAPER{
		{"loss", {
			{"otype", "RelativeL2Luminance"}
		}},
		// https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#exponential-moving-average-ema
		{"optimizer", {
			{"otype", "EMA"},
			{"decay", 0.99f},
			{"nested", {
				// https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#adam
				{"otype", "Adam"},
				{"learning_rate", 1e-3f},
			}}
		}},
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
				// Direction, normal, roughness (2+2+2)
				{
					{"otype", "OneBlob"},
					{"n_dims_to_encode", 2+2+2},
					{"n_bins", 4},
				},
				// Diffuse, specular albedos (3+3, automatically derived)
				{
					{"otype", "Identity"},
				},
			}}
		}},
		{"network", {
			{"otype", "FullyFusedMLP"},
			{"activation", "ReLU"},
			{"output_activation", "ReLU"}, // Output is radiance
			{"n_neurons", 64},
			{"n_hidden_layers", 5},
		}},
	};

}
};