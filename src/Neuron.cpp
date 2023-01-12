#include "Neuron.h"

void Neuron::create(const std::vector<float*> &weightsIn, const std::vector<float*> &weightsOut)
{
    static std::mt19937 gen(time(0));
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    p_weightsIn = weightsIn;
    p_weightsOut = weightsOut;
    m_netValue = 0.f;
    m_bias = dist(gen);
    m_delta = 0.f;
}

std::vector<float*> Neuron::getWeightsOutReferences()
{
    return p_weightsOut;
}

std::vector<float*> Neuron::getWeightsInReferences()
{
    return p_weightsIn;
}

float Neuron::getWeightOut(int outputNeuronIndex)
{
    return *(p_weightsOut.at(outputNeuronIndex));
}

float Neuron::getWeightIn(int inputNeuronIndex)
{
    return *(p_weightsIn.at(inputNeuronIndex));
}

void Neuron::updateWeightsIn(const std::vector<float> &weightsIn)
{
    if (weightsIn.size() != p_weightsIn.size()) {
        throw std::invalid_argument("Weights size from argument does not match size of neurons weights in.");
    }
    for (int i = 0; i < p_weightsIn.size(); i++) {
        *(p_weightsIn.at(i)) = weightsIn.at(i);
    }
}

void Neuron::updateWeightsOut(const std::vector<float> &weightsOut)
{
    if (weightsOut.size() != p_weightsOut.size()) {
        throw std::invalid_argument("Weights size from argument does not match size of neurons weights out.");
    }
    for (int i = 0; i < p_weightsOut.size(); i++) {
        *(p_weightsOut.at(i)) = weightsOut.at(i);
    }
}

float Neuron::getNetValue()
{
    return m_netValue;
}

void  Neuron::setNetValue(float netValue)
{
    m_netValue = netValue;
}

float Neuron::getBias()
{
    return m_bias;
}

void Neuron::setBias(float bias)
{
    m_bias = bias;
}

void Neuron::setDelta(float delta)
{
    m_delta = delta;
}

float Neuron::getDelta()
{
    return m_delta;
}
