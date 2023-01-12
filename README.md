# backpropagation

## Table of contents
* [Description](#description)
* [Usage](#usage)

## Description
Backpropagation algorithm implemented in C++.
By default input.png image from project's root directory is taken as an argument but there is an option to provide path to an image as program's argument, the image size has to be 50x50.

At the beginning the output image is black and the backpropagation algorithm is fixing weights and biases of neural network to make it look more like the image provided as the program's argument.
Image coords are taken as the neural network's input (current pixel) and expected output is the color (R, G, B) of the provided image at same coords.

## Usage
### Prepare SFML library
* Download SFML 2.5.1[^1]
* Copy SFML libraries
    - (WINDOWS) Copy `*.dll` files from SFML's bin directory to project's root directory.
    - (LINUX) Copy `*.so.*` files from SFML's lib directory to project's root directory.
* Copy SFML directory from SFML's include directory to project's include directory.
### Available commands
    make [help]

[^1]:[SFML 2.5.1 Download](https://www.sfml-dev.org/download/sfml/2.5.1/)
