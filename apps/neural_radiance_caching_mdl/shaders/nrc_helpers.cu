#include "config.h"

#include <optix.h>

#include "system_data.h"
#include "compositor_data.h"
#include "vector_math.h"

// NOTE: This is a separate copy of sys data from the one managed by Optix.
extern "C" __constant__ SystemData sysData;

extern "C" __global__ void placeholder() { return; }

// Radiance accumulator kernel to add the radiance * throughput 
// at the end of the rendering paths to the output buffer.
extern "C" __global__ void accumulate_render_radiance(float3 *lastRadiance, float3 *lastThroughput)
{
	const unsigned int xLaunch = blockDim.x * blockIdx.x + threadIdx.x; // [0..resolution.x)
	const unsigned int yLaunch = blockDim.y * blockIdx.y + threadIdx.y; // [0..resolution.y)
	if (xLaunch >= sysData.resolution.x || yLaunch >= sysData.resolution.y)
	{
		return;
	}

	const unsigned int index = yLaunch * sysData.resolution.x + xLaunch;
	const auto buffer = reinterpret_cast<float4*>(sysData.outputBuffer);
	
	const float3 radiance = lastThroughput[index] * lastRadiance[index];
	//const float3 radiance = lastThroughput[index];
	//const float3 radiance = lastRadiance[index];
	const float accWeight = 1.0f / float(sysData.pf.iterationIndex + 1);
	
	float3 dst = make_float3(buffer[index]);
	dst += (radiance * accWeight);	
	buffer[index] = make_float4(dst, 1.0f);
}
