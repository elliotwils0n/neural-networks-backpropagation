#include "Network.h"

void Network::create(const std::vector<int> &layersSizes, float learningRate)
{
    m_layers.clear();
    m_layers.push_back(Layer({}, layersSizes.at(0), layersSizes.at(1), Layer::LayerType::INPUT_LAYER));

    for (int layerSizeIdx = 1; layerSizeIdx < layersSizes.size() - 1; layerSizeIdx++) {
        m_layers.push_back(Layer(m_layers.back().getWeightsOutReferences(), layersSizes.at(layerSizeIdx), layersSizes.at(layerSizeIdx + 1), Layer::LayerType::HIDDEN_LAYER));
    }

    if (layersSizes.size() > 1) {
        m_layers.push_back(Layer(m_layers.back().getWeightsOutReferences(), layersSizes.back(), 0, Layer::LayerType::OUTPUT_LAYER));
    }

    m_learningRate = learningRate;
}

void Network::fit(const std::vector<float> &input, const std::vector<float> &expectedOutput)
{
    if (input.size() != m_layers.at(0).getNeurons().size()) {
        throw std::invalid_argument("Given input does not match first layer size!");
    }
    if (expectedOutput.size() != m_layers.back().getNeurons().size()) {
        throw std::invalid_argument("Expected output does not match last layer size!");
    }

    feedForward(input);
    backpropagation(expectedOutput);
}

std::vector<float> Network::predict(const std::vector<float> &input)
{
    feedForward(input);
    std::vector<float> output;
    for (float &outputLayerNetValue : m_layers.back().getNetValues()) {
        output.push_back(activation(outputLayerNetValue));
    }
    return output;
}

void Network::feedForward(const std::vector<float> &input)
{
    m_layers.at(0).setValues(input);

    if(m_layers.size() > 2) {
         for (int neuronIdx = 0; neuronIdx < m_layers.at(1).getNeurons().size(); neuronIdx++) {
            Neuron &neuron = m_layers.at(1).getNeurons().at(neuronIdx);
            float weightedSum = 0.f;
            std::vector<float> input = m_layers.at(0).getNetValues();
            for (int weightInIdx = 0; weightInIdx < neuron.getWeightsInReferences().size(); weightInIdx++) {
                weightedSum += input.at(weightInIdx) * neuron.getWeightIn(weightInIdx);
            }
           neuron.setNetValue(weightedSum + neuron.getBias());
        }
    }

    for (int layerIdx = 2; layerIdx < m_layers.size(); layerIdx++) {
        for (int neuronIdx = 0; neuronIdx < m_layers.at(layerIdx).getNeurons().size(); neuronIdx++) {
            Neuron &neuron = m_layers.at(layerIdx).getNeurons().at(neuronIdx);
            float weightedSum = 0.f;
            std::vector<float> previousLayerNetValues = m_layers.at(layerIdx - 1).getNetValues();
            for (int weightInIdx = 0; weightInIdx < neuron.getWeightsInReferences().size(); weightInIdx++) {
                weightedSum += activation(previousLayerNetValues.at(weightInIdx)) * neuron.getWeightIn(weightInIdx);
            }
            neuron.setNetValue(weightedSum + neuron.getBias());
        }
    }
}

void Network::backpropagation(const std::vector<float> &expectedOutput)
{
    // calculate output layer deltas
    for (int outputNeuronIdx = 0; outputNeuronIdx < m_layers.back().getNeurons().size(); outputNeuronIdx++) {
        Neuron &currentOutputNeuron = m_layers.back().getNeurons().at(outputNeuronIdx);
        float costDerivativeValue = costDerivative(currentOutputNeuron.getNetValue(), expectedOutput.at(outputNeuronIdx));
        float valueDerivative = activationDerivative(currentOutputNeuron.getNetValue());
        currentOutputNeuron.setDelta(valueDerivative * costDerivativeValue);
        currentOutputNeuron.setBias(currentOutputNeuron.getBias() - m_learningRate * currentOutputNeuron.getDelta());
    }

    // calculate hidden layers deltas
    for (int layerIdx = m_layers.size() - 2; layerIdx >= 0; layerIdx--) {
        for (int neuronIdx = 0; neuronIdx < m_layers.at(layerIdx).getNeurons().size(); neuronIdx++) {
            Neuron &currentNeuron = m_layers.at(layerIdx).getNeurons().at(neuronIdx);
            float valueDerivative = activationDerivative(currentNeuron.getNetValue());
            currentNeuron.setDelta(valueDerivative * m_layers.at(layerIdx + 1).getWeightedDeltaSum(neuronIdx));
            currentNeuron.setBias(currentNeuron.getBias() - m_learningRate * currentNeuron.getDelta());
        }
    }
    
    // calculate output layer weights
    for (int neuronIdx = 0; neuronIdx < m_layers.back().getNeurons().size(); neuronIdx++) {
        Neuron &currentNeuron = m_layers.back().getNeurons().at(neuronIdx);
        for (int weightIdx = 0; weightIdx < currentNeuron.getWeightsInReferences().size(); weightIdx++) {
            float prevLayerOutput = activation(m_layers.at(m_layers.size() - 2).getNeurons().at(weightIdx).getNetValue());
            float err =  currentNeuron.getDelta() * prevLayerOutput;
            *(currentNeuron.getWeightsInReferences().at(weightIdx)) -= m_learningRate * err;
        }
    }

    // calculate hidden layer weights
    for (int layerIdx = m_layers.size() - 2; layerIdx > 0; layerIdx--) {
        for (int neuronIdx = 0; neuronIdx < m_layers.at(layerIdx).getNeurons().size(); neuronIdx++) {
            Neuron &currentNeuron = m_layers.at(layerIdx).getNeurons().at(neuronIdx);
            for (int weightIdx = 0; weightIdx < currentNeuron.getWeightsInReferences().size(); weightIdx++) {
                float prevLayerValue = m_layers.at(layerIdx - 1).getNeurons().at(weightIdx).getNetValue();
                float err =  currentNeuron.getDelta() * prevLayerValue;
                *(currentNeuron.getWeightsInReferences().at(weightIdx)) -= m_learningRate * err;
            }
        }
    }
}

float Network::totalCost(const std::vector<float> &expectedOutput)
{
    if (expectedOutput.size() != m_layers.back().getNeurons().size()) {
        throw std::invalid_argument("Expected output size does not match output layer size!");
    }
    float totalC = 0.f;
    for (int outputNeuronIdx = 0; outputNeuronIdx < m_layers.back().getNeurons().size(); outputNeuronIdx++) {
        totalC += cost(m_layers.back().getNeurons().at(outputNeuronIdx).getNetValue(), expectedOutput.at(outputNeuronIdx));
    }
    return totalC;
}

inline float Network::cost(float netValue, float expectedValue)
{
    return 0.5 * std::pow(expectedValue - activation(netValue), 2);
}

inline float Network::costDerivative(float netValue, float expectedOutput)
{
    return activation(netValue) - expectedOutput;
}

inline float Network::activation(float input)
{
    return 1 / (1 + std::exp(-input));
}

inline float Network::activationDerivative(float input)
{
    return activation(input) * (1 - activation(input));
}
