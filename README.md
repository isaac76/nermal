# Nermal Neural Network Library

Hello, this is my attempt to learn how neural networks work. I named mine Nermal after the Garfield character because it kind of sounds like "neural" and I like Garfield. This project started with the book "Make Your Own Neural Network" by Tariq Rashid. You can find his GitHub project [here](https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork).

I went through his examples and developed a neural network in Python, then converted the project to C++ using Eigen to create a reusable library. While the examples and tests focus on MNIST digit recognition, **this is a general-purpose neural network library** that can be adapted for various machine learning tasks.

I used Copilot / Claude Sonnet to help convert from Python and to build the development tooling. It was a fun little project that taught me a lot about neural networks.

## Features

- Simple 3-layer neural network implementation (input, hidden, output)
- Sigmoid activation function with backpropagation training
- Serialization support for saving/loading trained models
- Cross-platform support (Linux, Windows, macOS)
- Both static (.a) and shared (.so/.dll/.dylib) library builds
- Modern CMake build system with easy integration
- Comprehensive unit and functional tests
- Well-documented API for easy integration

## Dependencies

- **CMake** 3.16 or higher
- **Eigen3** - Linear algebra library
- **C++17** compatible compiler
- **Google Test** (for unit tests)

## Building

### Quick Start

```bash
cd cpp
./build.sh
```

### Manual Build

```bash
cd cpp
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

### Build Options

```bash
# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Custom install prefix
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local

# Disable tests
cmake .. -DBUILD_TESTING=OFF

# Static libraries only
cmake .. -DBUILD_SHARED_LIBS=OFF
```

## Installation

### System-wide Installation

```bash
cd build
sudo cmake --install .
```

This installs:
- Libraries to `/usr/local/lib/`
- Headers to `/usr/local/include/nermal/`
- CMake config files to `/usr/local/lib/cmake/nermal/`

### Custom Installation

```bash
cmake --install . --prefix /your/custom/path
```

## Usage

### CMake Integration

After installation, use in your CMake project:

```cmake
find_package(nermal REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app nermal::nermal)  # Shared library
# or
target_link_libraries(my_app nermal::static)  # Static library
```

### pkg-config Integration

```bash
pkg-config --cflags --libs nermal
```

### Basic Usage Example

```cpp
#include <nermal/neuralnetwork.h>
#include <vector>

int main() {
    // Create a network: 784 inputs, 100 hidden, 10 outputs
    NeuralNetwork nn(784, 100, 10, 0.1);
    
    // Training data (example)
    std::vector<double> inputs(784, 0.5);  // Normalized pixel values
    std::vector<double> targets(10, 0.01); // One-hot encoded labels
    targets[3] = 0.99; // This is a "3"
    
    // Train the network
    nn.train(inputs, targets);
    
    // Query the network
    std::vector<double> outputs = nn.query(inputs);
    
    // Find predicted digit
    auto max_it = std::max_element(outputs.begin(), outputs.end());
    int predicted_digit = std::distance(outputs.begin(), max_it);
    
    return 0;
}
```

### Serialization Example

```cpp
// Save trained model
std::vector<uint8_t> model_data = nn.serializeToBytes();
// Save model_data to file...

// Load trained model
NeuralNetwork loaded_nn(784, 100, 10, 0.1);
loaded_nn.deserializeFromBytes(model_data);
```

## API Reference

### Constructor

```cpp
NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate);
```

### Training

```cpp
void train(const std::vector<double>& inputs, const std::vector<double>& targets);
```

### Inference

```cpp
std::vector<double> query(const std::vector<double>& inputs);
```

### Serialization

```cpp
std::vector<uint8_t> serializeToBytes() const;
bool deserializeFromBytes(const std::vector<uint8_t>& data);
```

### Getters

```cpp
int getInputNodes() const;
int getHiddenNodes() const;
int getOutputNodes() const;
double getLearningRate() const;
```

## Testing

The project includes comprehensive tests organized into different categories:

### Run All Tests

```bash
cd build
ctest --output-on-failure
```

### Run Specific Test Categories

#### Unit Tests
Tests individual components in isolation:
```bash
# Run via CTest
cd build
ctest --output-on-failure -R test_neuralnetwork

# Run directly
cd build
./test/unit/test_neuralnetwork
```

#### Functional Tests
End-to-end tests that validate the complete system:

**Full MNIST Test** (1000 training samples, 100 test samples, 5 epochs):
```bash
# Run via CTest
cd build
ctest --output-on-failure -R mnist_functional_test

# Run directly (from project root)
./cpp/build/test/functional/mnist_test
```

**Quick MNIST Test** (100 training samples, 20 test samples, 2 epochs):
```bash
# Run via CTest
cd build
ctest --output-on-failure -R mnist_quick_test

# Run directly (from project root)
./cpp/build/test/functional/mnist_quick_test
```

### Test Details

- **Unit tests** (`cpp/test/unit/`): Individual component testing using Google Test
- **Functional tests** (`cpp/test/functional/`): Full system validation with MNIST data
- **Quick test**: Fast validation (~65% accuracy, runs in ~10ms)
- **Full test**: Comprehensive validation (~90% accuracy, runs in ~180ms)

## Example Use Case: MNIST Digit Recognition

The included functional tests demonstrate the library's capabilities using MNIST handwritten digit recognition. This serves as both a validation of the library and an example of how to use it for classification tasks.

Training data comes from [phoebetronic/mnist](https://github.com/phoebetronic/mnist), which provides 60,000 28×28 pixel images of handwritten digits (0-9) for training and 10,000 images for testing.

### How the Neural Network Works (MNIST Example)

This neural network is designed to recognize handwritten digits (0-9) from 28x28 pixel images. Here's how it operates:

#### Network Architecture
- **784 input nodes** (28×28 pixels flattened into a single vector)
- **100 hidden nodes** (the "feature detectors")
- **10 output nodes** (one for each digit 0-9)

#### Weight Matrices: The Brain's Connections
The network uses two weight matrices to connect the layers:
- **Input→Hidden weights** (100×784): Each element represents the connection strength from a specific pixel to a specific hidden node
- **Hidden→Output weights** (10×100): Each element represents how much each hidden node influences each digit prediction

Think of weights as the "strength of belief" - positive weights excite neurons, negative weights inhibit them. During training, these weights adjust to recognize patterns.

#### The Sigmoid Function: Making Decisions
The network uses the sigmoid activation function: σ(x) = 1/(1 + e^(-x))
- Maps any input to a value between 0 and 1
- Acts like a "soft switch" - gradually turns neurons on/off rather than hard binary decisions
- Has a useful mathematical property: its derivative is σ(x)(1-σ(x)), which simplifies learning calculations

#### Processing the Very First Training Image
When the network sees its first handwritten "3":

1. **Random Start**: All 784×100 + 100×10 = 79,400 weights start as small random numbers
2. **Forward Pass**: 
   - Each pixel value (0.01 to 0.99) gets multiplied by random weights
   - Hidden nodes receive weighted sums of ALL pixels
   - Since weights are random, hidden nodes produce essentially random outputs
   - Output layer makes a random guess (maybe predicting "7" when it should predict "3")
3. **Target Vector Creation**: The correct answer (e.g., digit "3") gets converted to a target vector:
   - All 10 positions start at 0.01 (representing "not this digit")
   - Position 3 gets set to 0.99 (representing "this is a 3")
   - This one-hot encoding gives the network clear learning targets
4. **Error Calculation**: Network compares its random guess to this target vector
5. **Backpropagation**: The learning algorithm adjusts weights based on the error:
   - If a hidden node contributed to a wrong answer, its connections get weakened
   - If a hidden node could have helped the right answer, its connections get strengthened

#### After Thousands of Training Images
The magic happens through repetition:

**Hidden Node Specialization**: The 100 hidden nodes develop specialized roles:
- Some detect curves in specific image regions
- Others recognize vertical lines, horizontal segments, or diagonal strokes
- Many develop overlapping specializations for robustness

**Pattern Recognition**: For a "3", the trained network might activate:
- Hidden nodes that detect right-facing curves (strongly active)
- Hidden nodes that detect horizontal middle segments (moderately active)  
- Hidden nodes that detect closed loops (inactive - distinguishes from "8")

**Output Layer Learning**: Each output node learns which combination of hidden node activations represents its digit:
- Output node "3" learns to fire when the "right curves + middle horizontal" pattern appears
- Output node "8" learns to fire when the "closed loops + curves" pattern appears

#### Learning Rate: Controlling the Pace
The learning rate (0.1 in this implementation) controls how quickly weights change:
- Too high (e.g., 5.0): Network makes huge adjustments and becomes unstable
- Too low (e.g., 0.001): Network learns too slowly, may never reach good performance
- Just right (0.1): Steady, stable progress toward accurate digit recognition

#### The End Result
After training on thousands of diverse handwritten digits, the network develops an internal representation of what makes each digit unique. When shown a new handwritten digit, it can recognize it by comparing the activation patterns of its hidden nodes to the learned patterns for each digit class.

**Note**: While this example focuses on digit recognition, the same principles apply to other classification tasks. You could adapt this library for text classification, simple image recognition, or other pattern recognition problems by adjusting the input size, output size, and training data format.

## Cross-Platform Notes

### Linux
- Generates `libnermal.so` (shared) and `libnermal.a` (static)
- Uses standard Unix installation paths

### Windows
- Generates `nermal.dll` (shared) and `nermal.lib` (static)
- Exports all symbols automatically

### macOS
- Generates `libnermal.dylib` (shared) and `libnermal.a` (static)
- Compatible with macOS frameworks

## Python Version

The original Python implementation is included in the `python/` directory for reference and comparison. The C++ library provides the same functionality with better performance and easier integration into other C++ projects.

## License

This project is licensed under the GNU Lesser General Public License v2.1 (LGPL-2.1). See the [LICENSE](LICENSE) file for details.

The LGPL allows you to use this library in both open source and proprietary projects, but any modifications to the library itself must be shared under the same license. This makes it ideal for libraries that you want to be widely adopted while still ensuring improvements benefit the community.

## Contributing

Contributions are welcome! Whether you're fixing bugs, adding features, improving documentation, or enhancing the test suite, your help is appreciated.

**Educational Focus**: This project is primarily for educational purposes. While there are much more sophisticated and performant neural network libraries available (TensorFlow, PyTorch, etc.), the goal here is to understand how neural networks work by implementing and experimenting with the code ourselves. This makes it a perfect playground for learning and trying new ideas!

### How to Contribute

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `main`: `git checkout -b feature/your-feature-name`
3. **Make your changes** and ensure they follow the existing code style
4. **Add tests** for new functionality
5. **Run the test suite** to make sure everything works: `ctest --output-on-failure`
6. **Commit your changes** with clear, descriptive messages
7. **Push to your fork** and **submit a pull request** to the `main` branch

### What We're Looking For

- Bug fixes and performance improvements
- Additional activation functions (ReLU, tanh, etc.)
- Support for different network architectures
- Documentation improvements
- More comprehensive test coverage
- Examples for different use cases beyond MNIST

### Pull Request Guidelines

- Include a clear description of what your changes do
- Reference any related issues
- Make sure all tests pass
- Keep changes focused and atomic (one feature/fix per PR)
- Update documentation as needed

Feel free to open an issue first if you want to discuss a major change or new feature!

