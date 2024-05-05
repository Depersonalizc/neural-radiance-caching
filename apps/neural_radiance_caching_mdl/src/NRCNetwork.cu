#include "inc/NRCNetwork.h"
#include "inc/NRCNetworkConfigs.h"

#include "inc/CheckMacros.h"
#include "shaders/neural_radiance_caching.h"

#include <iostream>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>

namespace nrc {

struct Network::Impl {

	tcnn::TrainableModel model{};

	// The network (hyper)parameters 
	// Default as specified in the paper
	nlohmann::json config = cfg::MODEL_CONFIG_PAPER;

};

using GPUMatrix_t = tcnn::GPUMatrix<float, tcnn::MatrixLayout::ColumnMajor>;

Network::Network() 
	: pImpl{ std::make_unique<Impl>() } 
{}

Network::~Network()
{
	if (!m_destroyed)
		std::cerr << "WARNING: NRC Network must be explicitly destroy() in the context it was created in!\n";
}

void Network::destroy()
{
	pImpl.reset();
	m_destroyed = true;
}

void Network::train(float* batchInputs_d, float* batchTargets_d, float* loss_h)
{
	if (m_destroyed) [[unlikely]] return;

	using namespace tcnn;
	static auto& trainer = pImpl->model.trainer;

	const GPUMatrix_t inputs { batchInputs_d,  NN_INPUT_DIMS,  BATCH_SIZE };
	const GPUMatrix_t targets{ batchTargets_d, NN_OUTPUT_DIMS, BATCH_SIZE };
	auto ctx = trainer->training_step(m_stream, inputs, targets);
	//{
	//	const auto outputs = ctx->output.to_cpu_vector();
	//	outputs;
	//	const auto loss = ctx->L.to_cpu_vector();
	//	loss;
	//}
	if (loss_h)
		*loss_h = trainer->loss(m_stream, *ctx);
}

void nrc::Network::train(float* batchInputs_d, float* batchTargets_d, CUstream stream, float* loss_h)
{
	setStream(stream);
	train(batchInputs_d, batchTargets_d, loss_h);
}

void Network::infer(float* inputs_d, float* outputs_d, uint32_t numInputs)
{
	if (m_destroyed) [[unlikely]] return;

	using namespace tcnn;
	static auto& network = pImpl->model.network;
	
	// Round up to nearest multiple of BATCH_SIZE_GRANULARITY.
	numInputs = ((numInputs + BATCH_SIZE_GRANULARITY - 1) / BATCH_SIZE_GRANULARITY) * BATCH_SIZE_GRANULARITY;
	const GPUMatrix_t inputs { inputs_d,  NN_INPUT_DIMS,  numInputs };
	GPUMatrix_t       outputs{ outputs_d, NN_OUTPUT_DIMS, numInputs };

	network->inference(m_stream, inputs, outputs);
}

void Network::infer(float* inputs_d, float* outputs_d, uint32_t numInputs, CUstream stream)
{
	setStream(stream);
	infer(inputs_d, outputs_d, numInputs);
}

void Network::setStream(CUstream stream)
{
	m_stream = stream;
}

void Network::init_(CUstream stream)
{
	setStream(stream);
	pImpl->model = tcnn::create_from_config(NN_INPUT_DIMS, NN_OUTPUT_DIMS, pImpl->config);
}

void Network::init_(CUstream stream, float lr, float emaDecay/* = 0.99f*/)
{
	auto& opt = pImpl->config["optimizer"];
	opt["nested"]["learning_rate"] = lr;
	opt["decay"] = emaDecay;
	
	init_(stream);
}

void Network::printConfig_() const
{
	std::cout << "\n----------------------- NETWORK CONFIG -----------------------\n"
		      << pImpl->config
			  << "\n--------------------------------------------------------------\n\n";
}



}
