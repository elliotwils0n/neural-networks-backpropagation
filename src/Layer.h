#pragma once

#include "Header.h"
#include "Neuron.h"

class Layer
{
public:
    enum class LayerType {
        INPUT_LAYER,
        HIDDEN_LAYER,
        OUTPUT_LAYER
    };

    Layer(const std::vector<std::vector<float*>> &previousWeightsOut, int layerSize, int nextLayerSize, LayerType layerType);
    Layer(Layer&&) = default;
    ~Layer() = default;

    void setValues(const std::vector<float> &input);
    std::vector<Neuron>& getNeurons();
    std::vector<float> getNetValues();
    std::vector<std::vector<float*>> getWeightsOutReferences();
    float getWeightedSumForOutputNeuron(int outputNeuronIdx);
    float getWeightedDeltaSum(int weightIdx);
private:
    std::vector<Neuron> m_neurons;
    std::vector<float> m_weightsOut;
};
