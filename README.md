# Neuronov Net

A simple C++11 library for basic neural network operations on CPU.

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [License](#license)

## Features

- **Feed-forward Neural Network**: Supports basic feed-forward neural networks.
- **Learning Algorithm**: Implements a simple backpropagation algorithm for training.
- **Customizable Activation Functions**: Allows custom activation functions and their derivatives.
- **Serialization**: Supports saving and loading neural network architectures and weights.
- **Template-based Design**: Highly customizable with C++ templates.

## Installation

To use the Neuronov Net library, simply include the `neuronov_net.hpp` header file in your project.

```cpp
#include "neuronov_net.hpp"
```

## Usage

### Creating a Neural Network

To create a neural network, you need to specify the architecture (number of neurons in each layer), activation functions, and a random number generator for initializing weights.

```cpp
#include "neuronov_net.hpp"
#include <cmath>
#include <random>

// Define activation function and its derivative
float sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}

float sigmoid_derivative(float x) {
    const float s = sigmoid(x);
    return s * (1.0 - s);
}

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-1.0, 1.0);
auto random_number_generator = [&]() { return dis(gen); };

int main() {
    // Define the architecture of the neural network
    std::vector<std::size_t> architecture = {2, 3, 1}; // Input layer: 2 neurons, Hidden layer: 3 neurons, Output layer: 1 neuron

    // Create the neural network
    neuronov_net::perseptron net(architecture, sigmoid, sigmoid_derivative, random_number_generator);

    // ...
}
```

### Feed-forward Pass

To perform a feed-forward pass, set the input values and call the `feed_forward` method.

```cpp
// Set input values
auto input = net.get_input();
input[0].value = 0.5;
input[1].value = 0.2;

// Perform feed-forward pass
net.feed_forward();

// Get output values
const auto output = net.get_output();
for (const auto& neuron : output)
    std::cout << "Output: " << neuron.value << '\n';
```

### Training the Neural Network

To train the neural network, provide the correct results and a learning rate, then call the `learn` method.

```cpp
// Correct results for the output layer
std::vector<float> correct_results = {0.8};

// Learning rate
float learn_rate = 0.1;

// Perform learning step
net.learn(correct_results, learn_rate);
```

### Saving and Loading the Neural Network

You can save the neural network architecture and weights to a stream and load them back.

```cpp
// Save the neural network to a stream
{
  std::ofstream output_stream("network.dat");
  net.safe(output_stream);
}

// Load the neural network from a stream
{
  std::ifstream input_stream("network.dat");
  net.load(input_stream);
}
```

## Examples

Here is a simple example demonstrating the basic usage of the Neuronov Net library:

```cpp
#include "neuronov_net.hpp"
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>

// Define activation function and its derivative
float sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}

float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0 - s);
}

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-1.0, 1.0);
auto random_number_generator = [&]() { return dis(gen); };

int main() {
    // Define the architecture of the neural network
    std::vector<std::size_t> architecture = {2, 3, 1}; // Input layer: 2 neurons, Hidden layer: 3 neurons, Output layer: 1 neuron

    // Create the neural network
    neuronov_net::perseptron net(architecture, sigmoid, sigmoid_derivative, random_number_generator);

    // Set input values
    auto input = net.get_input();
    input[0].value = 0.5;
    input[1].value = 0.2;

    // Perform feed-forward pass
    net.feed_forward();

    // Get output values
    const auto& output = net.get_output();
    for (const auto& neuron : output)
        std::cout << "Output: " << neuron.value << '\n';

    // Correct results for the output layer
    std::vector<float> correct_results = {0.8};

    // Learning rate
    float learn_rate = 0.1;

    // Perform learning step
    net.learn(correct_results, learn_rate);

    // Save the neural network to a stream
    {
      std::ofstream output_stream("network.dat");
      net.safe(output_stream);
    }

    // Load the neural network from a stream
    {
      std::ifstream input_stream("network.dat");
      net.load(input_stream);
    }

    return 0;
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
