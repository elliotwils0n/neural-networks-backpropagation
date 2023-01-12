#include "Layer.h"

Layer::Layer(const std::vector<std::vector<float*>> &previousWeightsOut, int layerSize, int nextLayerSize, LayerType layerType)
{
    static std::mt19937 gen(time(0));
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    m_weightsOut.reserve(layerSize * nextLayerSize);
    m_neurons = std::vector<Neuron>(layerSize);

    for (int neuronIdx = 0; neuronIdx < m_neurons.size(); neuronIdx++) {
        std::vector<float*> weightsIn;
        std::vector<float*> weightsOut;

        if (layerType != LayerType::INPUT_LAYER) {
            for (int prevWeightIdx = 0; prevWeightIdx < previousWeightsOut.size(); prevWeightIdx++) {
                weightsIn.push_back(previousWeightsOut.at(prevWeightIdx).at(neuronIdx));
            }
        }

        if (layerType != LayerType::OUTPUT_LAYER) {
            for (int i = 0; i < nextLayerSize; i++) {
                m_weightsOut.push_back(dist(gen));
                weightsOut.push_back(&(m_weightsOut.back())); 
            }
        }

        m_neurons.at(neuronIdx).create(weightsIn, weightsOut);
    }
}

void Layer::setValues(const std::vector<float> &input)
{
    if (input.size() != m_neurons.size()) {
        throw std::invalid_argument("Input size does not match layer size!");
    }
    for (int i = 0; i < m_neurons.size(); i++) {
        m_neurons.at(i).setNetValue(input.at(i));
    }
}

std::vector<Neuron>& Layer::getNeurons()
{
    return m_neurons;
}

std::vector<float> Layer::getNetValues()
{
    std::vector<float> values;
    for (Neuron &neuron : m_neurons) {
        values.push_back(neuron.getNetValue());
    }
    return values;
}

std::vector<std::vector<float*>> Layer::getWeightsOutReferences()
{
    std::vector<std::vector<float*>> result;

    for (auto &neuron : m_neurons) {
        result.push_back(neuron.getWeightsOutReferences());
    }
    
    return result;
}

float Layer::getWeightedSumForOutputNeuron(int outputNeuronIdx)
{
    if (outputNeuronIdx >= m_weightsOut.size() / m_neurons.size()) {
        throw std::invalid_argument("Invalid argument. Output neuron index out of range!");
    }

    float sum = 0.f;
    for (Neuron &neuron : m_neurons) {
        sum += neuron.getNetValue() * neuron.getWeightOut(outputNeuronIdx);
    }
    return sum;
}

float Layer::getWeightedDeltaSum(int weightIdx)
{
    float sum = 0.f;
    for (Neuron &neuron : m_neurons) {
        sum += neuron.getDelta() * neuron.getWeightIn(weightIdx);
    }
    return sum;
}
