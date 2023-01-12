#include "Network.h"

int INPUT_LAYER_SIZE = 30;
int OUTPUT_LAYER_SIZE = 3;
int ITERATIONS = 100000;
int MAX = 255;

std::vector<float> prepareInput(float x, float y)
{
    std::vector<float> vec;
    vec.push_back(x);
    vec.push_back(y);
    for (int inputIdx = 1; inputIdx <= (INPUT_LAYER_SIZE - 2) / 4; inputIdx++) {
        vec.push_back(std::sin(x * inputIdx * 2 * M_PI));
        vec.push_back(std::cos(x * inputIdx * 2 * M_PI));
        vec.push_back(std::sin(y * inputIdx * 2 * M_PI));
        vec.push_back(std::cos(y * inputIdx * 2 * M_PI));
    }
    return vec;
}

int main(int argc, char **argv)
{
    static std::mt19937 gen(time(0));
    std::uniform_int_distribution<int> dist(0, MAX);
    std::uniform_int_distribution<int> distX(0, 2);
    std::uniform_int_distribution<int> distY(0, 2);
    std::vector<std::vector<std::vector<float>>> expectedOutput = 
        std::vector<std::vector<std::vector<float>>>(OUTPUT_LAYER_SIZE, std::vector<std::vector<float>>(OUTPUT_LAYER_SIZE, std::vector<float>(OUTPUT_LAYER_SIZE)));
    Network network;

    network.create({INPUT_LAYER_SIZE, (INPUT_LAYER_SIZE + OUTPUT_LAYER_SIZE) / 2, OUTPUT_LAYER_SIZE});

    for (int y = 0; y < OUTPUT_LAYER_SIZE; y++) {
        for (int x = 0; x < OUTPUT_LAYER_SIZE; x++) {
            expectedOutput.at(y).at(x).at(0) = dist(gen);
            expectedOutput.at(y).at(x).at(1) = dist(gen);
            expectedOutput.at(y).at(x).at(2) = dist(gen);
        }
    }

    for (int i = 1; i <= ITERATIONS; i++) {
        int x = distX(gen);
        int y = distY(gen);
        std::vector<float> input = prepareInput(x / (float)OUTPUT_LAYER_SIZE, y / (float)OUTPUT_LAYER_SIZE);
        std::vector<float> expected = { 
            expectedOutput.at(y).at(x).at(0) / float(MAX),
            expectedOutput.at(y).at(x).at(1) / float(MAX),
            expectedOutput.at(y).at(x).at(2) / float(MAX)
        };
        network.fit(input, expected);
    }

    std::cout << std::fixed << std::setprecision(0);

    for (int y = 0; y < 3; y ++) {
        for (int x = 0; x < 3; x++) {
            std::vector<float> input = prepareInput(x / (float)OUTPUT_LAYER_SIZE, y / (float)OUTPUT_LAYER_SIZE);
            std::vector<float> calculated = network.predict(input);

            std::cout << x << ", " << y << ":" << std::endl;

            std::cout << "  Expected: (";
            std::cout << std::setw(3) << expectedOutput.at(y).at(x).at(0) << ", ";
            std::cout << std::setw(3) << expectedOutput.at(y).at(x).at(1) << ", ";
            std::cout << std::setw(3) << expectedOutput.at(y).at(x).at(2) << ");" << std::endl;

            std::cout << "Calculated: (";
            std::cout << std::setw(3) << std::round(calculated.at(0) * MAX) << ", ";
            std::cout << std::setw(3) << std::round(calculated.at(1) * MAX) << ", ";
            std::cout << std::setw(3) << std::round(calculated.at(2) * MAX) << ");" << std::endl;;
        }
    }

    return 0;
}
