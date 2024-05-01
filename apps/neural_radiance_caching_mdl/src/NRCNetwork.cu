#include "inc/NRCNetwork.h"
#include "inc/NRCNetworkConfigs.h"

#include "inc/CheckMacros.h"

#include <iostream>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>

namespace nrc {

struct Network::Impl {

	tcnn::TrainableModel model;

	// The network (hyper)parameters 
	// Default as specified in the paper
	nlohmann::json config = cfg::MODEL_CONFIG_PAPER;

};

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

void Network::setStream(CUstream stream)
{
	m_stream = stream;
}

void Network::init_(CUstream stream)
{
	setStream(stream);
	pImpl->model = tcnn::create_from_config(cfg::INPUT_DIMS, cfg::OUTPUT_DIMS, pImpl->config);
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
