#pragma once

#include "Header.h"

class Neuron 
{
public:
    Neuron() = default;
    Neuron(Neuron&&) = default;
    ~Neuron() = default;

    void create(const std::vector<float*> &weightsIn, const std::vector<float*> &weightsOut);

    std::vector<float*> getWeightsInReferences();
    std::vector<float*> getWeightsOutReferences();
    float getWeightOut(int outputNeuronIndex);
    float getWeightIn(int inputNeuronIndex);
    void updateWeightsIn(const std::vector<float> &weightsIn);
    void updateWeightsOut(const std::vector<float> &weightsOut);

    float getNetValue();
    void setNetValue(float netValue);
    float getBias();
    void setBias(float bias);
    void setDelta(float delta);
    float getDelta();
private:
    std::vector<float*> p_weightsIn;
    std::vector<float*> p_weightsOut;
    float m_netValue;
    float m_bias;
    float m_delta;
};
