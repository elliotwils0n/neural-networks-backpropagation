#pragma once

#include "Header.h"
#include "Layer.h"

class Network
{
public:
    Network() = default;
    ~Network() = default;

    void create(const std::vector<int> &layersSizes, float learningRate = 0.1f);
    void fit(const std::vector<float> &input, const std::vector<float> &expectedOutput);
    std::vector<float> predict(const std::vector<float> &input);
private:
    void feedForward(const std::vector<float> &input);
    void backpropagation(const std::vector<float> &expectedOutput);
    float totalCost(const std::vector<float> &expectedOutput);

    inline float cost(float netValue, float expectedOutput);
    inline float costDerivative(float netValue, float expectedOutput);
    inline float activation(float input);
    inline float activationDerivative(float input);
private:
    std::vector<Layer> m_layers;
    float m_learningRate;
};
