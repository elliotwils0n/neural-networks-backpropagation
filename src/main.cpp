#include "Header.h"
#include "Constant.h"
#include "Network.h"
#include <SFML/Graphics.hpp>

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
    sf::Vector2f imgSize(X_SIZE, Y_SIZE);
    sf::Vector2f imgScale(X_SCALE, Y_SCALE);
    std::string inputImageFilepath = DEFAULT_INPUT_IMAGE;

    if (argc == 2) {
        inputImageFilepath = argv[1];
    } else if (argc > 2) {
        std::cout << "Too much arguments passed to the program." << std::endl;
        std::cout << "Only none or one argument supported (image filepath, default: input.png)." << std::endl;
        return -1;
    }

    Network network;
    static std::mt19937 gen(time(0));
    std::uniform_int_distribution<int> distX(0, X_SIZE - 1);
    std::uniform_int_distribution<int> distY(0, Y_SIZE - 1);
    int counter = 0;

    sf::RenderWindow m_window;
    sf::IntRect m_rect;
    sf::Image m_inputImage, m_outputImage;
    sf::Texture m_texture;
    sf::Sprite m_sprite;

    network.create({INPUT_LAYER_SIZE, (INPUT_LAYER_SIZE + OUTPUT_LAYER_SIZE) / 2, OUTPUT_LAYER_SIZE});
    // std::cout << std::fixed << std::setprecision(2);
    // network.printNetworkInfo();

    m_window.create(sf::VideoMode(imgSize.x * imgScale.x, imgSize.y * imgScale.y), "RGB", sf::Style::Titlebar | sf::Style::Close);

    m_rect = sf::IntRect(0, 0, imgSize.x, imgSize.y);

    m_inputImage.create(imgSize.x, imgSize.y);
    if (!m_inputImage.loadFromFile(inputImageFilepath)) {
        std::invalid_argument("Image not loaded ( "+ inputImageFilepath + ").");
    }

    m_outputImage.create(imgSize.x, imgSize.y);
    for (int y = 0; y < Y_SIZE; y++) {
        for (int x = 0; x < X_SIZE; x++) {
            m_outputImage.setPixel(x, y, sf::Color::Black);
        }
    }

    m_texture.create(imgSize.x, imgSize.y);
    m_texture.loadFromImage(m_outputImage, m_rect);

    m_sprite.setPosition({0.f, 0.f});
    m_sprite.setScale(imgScale);
    m_sprite.setTextureRect(m_rect);
    m_sprite.setTexture(m_texture);
    
    while (m_window.isOpen())
    {
        sf::Event event;
        while (m_window.pollEvent(event))
            if (event.type == sf::Event::Closed)
                m_window.close();

        for (int i = 0; i < DATA_SIZE; i++) {
            int x = distX(gen);
            int y = distY(gen);
            sf::Color expected = m_inputImage.getPixel(x, y);
            std::vector<float> input = prepareInput(x / imgSize.x, y / imgSize.y);
            std::vector<float> output = {
                expected.r / (float)MAX_COLOR_VALUE,
                expected.g / (float)MAX_COLOR_VALUE,
                expected.b / (float)MAX_COLOR_VALUE
            };
            network.fit(input, output);  
        }

        counter = (counter + 1) % REFRESH_FREQUENCY;
        if (counter == 0) {
            for (int y = 0; y < Y_SIZE; y++) {
                for (int x = 0; x < X_SIZE; x++) {
                    std::vector<float> input = prepareInput(x / imgSize.x, y / imgSize.y);
                    std::vector<float> calculatedOutput = network.predict(input);
                    if (calculatedOutput.size() != OUTPUT_LAYER_SIZE) {
                        throw std::invalid_argument("Invalid calculated output size.");
                    }
                    sf::Color newColor = sf::Color(
                        (unsigned char)(calculatedOutput.at(0) * MAX_COLOR_VALUE),
                        (unsigned char)(calculatedOutput.at(1) * MAX_COLOR_VALUE),
                        (unsigned char)(calculatedOutput.at(2) * MAX_COLOR_VALUE)
                    );
                    m_outputImage.setPixel(x, y,newColor );
                }
            }

            m_texture.loadFromImage(m_outputImage, m_rect);
            // m_outputImage.saveToFile(DEFAULT_OUTPUT_IMAGE);
            m_window.clear(sf::Color(MAX_COLOR_VALUE, MAX_COLOR_VALUE, MAX_COLOR_VALUE));
            m_window.draw(m_sprite);
        }

        m_window.display();
    }

    return 0;
}
