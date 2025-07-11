# Nermal Neural Network

A C++ neural network implementation using Eigen3 and Qt6, designed for processing numerical data including MNIST digit recognition.

## Features

- **Neural Network**: Fully connected feedforward network with configurable architecture
- **MNIST Support**: Built-in support for MNIST digit recognition with 90%+ accuracy
- **Qt6 Integration**: Modern Qt6 Widgets GUI application
- **Comprehensive Testing**: Unit tests with Qt Test framework and functional tests
- **Professional Build System**: CMake with CTest integration

## Requirements

- **Compiler**: GCC 11+ (required for Qt6 C++17 support)
- **Qt6**: Widgets and Test components
- **Eigen3**: Version 3.4.0+ for matrix operations
- **CMake**: Version 3.5+

## Building

### Option 1: VS Code (Recommended)
The project is configured to work out-of-the-box with VS Code CMake Tools extension.

### Option 2: Command Line
```bash
# Clone and navigate to project
cd nermal

# Configure with GCC 11 toolchain (important!)
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../gcc11-toolchain.cmake -DCMAKE_BUILD_TYPE=Debug ..

# Build
make

# Run tests
make test
```

### Option 3: Qt Creator
Due to compiler requirements, Qt Creator needs special configuration:

1. **Project Settings**: Projects → Build & Run → CMake
2. **Initial Configuration**: Add or modify:
   ```
   -DCMAKE_TOOLCHAIN_FILE:FILEPATH=%{BuildConfig:Path:PrefixToSourceDir}/gcc11-toolchain.cmake
   -DCMAKE_BUILD_TYPE:STRING=Debug
   ```
3. **Reconfigure** the project

Alternative: Use the provided script:
```bash
./launch-qtcreator.sh
```

## Testing

- **Unit Tests**: `./build/tests/unit/test_neuralnetwork`
- **MNIST Functional Test**: `./build/tests/functional/mnist_test`
- **Quick MNIST Test**: `./build/tests/functional/mnist_quick_test`
- **All Tests**: `cd build && make test`

## Architecture

- **Input Layer**: 784 nodes (28×28 MNIST pixels)
- **Hidden Layer**: 100 nodes
- **Output Layer**: 10 nodes (digits 0-9)
- **Activation**: Sigmoid function
- **Learning**: Backpropagation with configurable learning rate

## Files

- `src/neuralnetwork.{h,cpp}`: Core neural network implementation
- `src/main.cpp`, `src/mainwindow.*`: Qt GUI application
- `tests/unit/`: Qt Test framework unit tests
- `tests/functional/`: MNIST integration tests
- `gcc11-toolchain.cmake`: Compiler configuration for Qt6 compatibility
