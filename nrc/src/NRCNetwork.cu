// Copyright (c) 2024, Jamie Chen <jamiechenang@gmail>

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
        nlohmann::json config{};
    };

    using GPUMatrix_t = tcnn::GPUMatrix<float, tcnn::MatrixLayout::ColumnMajor>;

    Network::Network()
        : pImpl{std::make_unique<Impl>()}
    {
    }

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

    void Network::train(float *batchInputs_d, float *batchTargets_d, float *loss_h)
    {
        if (m_destroyed) [[unlikely]] return;

        using namespace tcnn;
        auto &trainer = pImpl->model.trainer;

        static_assert(BATCH_SIZE % BATCH_SIZE_GRANULARITY == 0 &&
                      "Training batch size is not a multiple of tcnn::BATCH_SIZE_GRANULARITY");

        const GPUMatrix_t inputs{batchInputs_d, NN_INPUT_DIMS, BATCH_SIZE};
        const GPUMatrix_t targets{batchTargets_d, NN_OUTPUT_DIMS, BATCH_SIZE};
        auto ctx = trainer->training_step(m_stream, inputs, targets);
        if (loss_h)
            *loss_h = trainer->loss(m_stream, *ctx);
    }

    void Network::train(float *batchInputs_d, float *batchTargets_d, CUstream stream, float *loss_h)
    {
        setStream(stream);
        train(batchInputs_d, batchTargets_d, loss_h);
    }

    void Network::infer(float *inputs_d, float *outputs_d, uint32_t numInputs)
    {
        if (m_destroyed) [[unlikely]] return;

        using namespace tcnn;
        auto &network = pImpl->model.network;

        // Round up to nearest multiple of BATCH_SIZE_GRANULARITY.
        numInputs = ((numInputs + BATCH_SIZE_GRANULARITY - 1) / BATCH_SIZE_GRANULARITY) * BATCH_SIZE_GRANULARITY;
        const GPUMatrix_t inputs{inputs_d, NN_INPUT_DIMS, numInputs};
        GPUMatrix_t outputs{outputs_d, NN_OUTPUT_DIMS, numInputs};

        network->inference(m_stream, inputs, outputs);
    }

    void Network::infer(float *inputs_d, float *outputs_d, uint32_t numInputs, CUstream stream)
    {
        setStream(stream);
        infer(inputs_d, outputs_d, numInputs);
    }

    void Network::setStream(CUstream stream)
    {
        m_stream = stream;
    }

    void Network::setHyperParams(const HyperParams &hp)
    {
        pImpl->config["optimizer"]["nested"]["learning_rate"] = hp.learningRate;
        pImpl->model.optimizer->set_learning_rate(hp.learningRate);
    }

    void Network::setConfig(InputEncoding encoding)
    {
        pImpl->config = cfg::modelConfig(encoding);
    }

    float Network::getLearningRate() const
    {
        return pImpl->model.optimizer->learning_rate();
    }

    void Network::init_(CUstream stream, InputEncoding encoding)
    {
        setStream(stream);
        setConfig(encoding);
        // Use the default config for `encoding` to create the model.
        pImpl->model = tcnn::create_from_config(NN_INPUT_DIMS, NN_OUTPUT_DIMS, pImpl->config);
    }

    //void Network::init_(CUstream stream, InputEncoding, float lr, float emaDecay/* = 0.99f*/)
    //{
    //	auto& opt = pImpl->config["optimizer"];
    //	opt["nested"]["learning_rate"] = lr;
    //	opt["decay"] = emaDecay;
    //	init_(stream);
    //}

    void Network::printConfig_() const
    {
        std::cout << "\n----------------------- NETWORK CONFIG -----------------------\n"
                << pImpl->config
                << "\n--------------------------------------------------------------\n\n";
    }
}
