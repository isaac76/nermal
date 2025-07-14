# VS Code Setup for Nermal Neural Network Library

This document explains how to use VS Code effectively with the Nermal neural network library project.

## Prerequisites

Make sure you have the following installed:
- VS Code
- CMake (3.16 or higher)
- C++ compiler (g++)
- Google Test library
- Eigen3 library

## Recommended Extensions

When you open this project in VS Code, you'll be prompted to install recommended extensions. The key ones are:

- **C/C++** - IntelliSense, debugging, and code browsing
- **CMake Tools** - CMake integration and build system
- **C/C++ Extension Pack** - Complete C++ development environment

## Available Tasks

Press `Ctrl+Shift+P` and type "Tasks: Run Task" to access these build tasks:

### Build Tasks
- **Build Library** (Default: `Ctrl+Shift+B`) - Build the entire project
- **Configure CMake** - Configure the CMake build system
- **Clean Build** - Clean the build directory and start fresh
- **Install Library** - Install the library to a specified location

### Test Tasks
- **Run All Tests** - Run both unit and functional tests
- **Run Unit Tests** - Run only the Google Test unit tests
- **Run MNIST Quick Test** - Run the quick MNIST functional test
- **Run MNIST Full Test** - Run the full MNIST functional test

## Debugging

Use `F5` or the Debug panel to start debugging:

- **Debug Unit Tests** - Debug the Google Test unit tests
- **Debug MNIST Quick Test** - Debug the quick MNIST test
- **Debug MNIST Full Test** - Debug the full MNIST test

## Project Structure

```
.
├── cpp/
│   ├── src/
│   │   ├── neuralnetwork.cpp
│   │   └── neuralnetwork.h
│   ├── test/
│   │   ├── unit/
│   │   │   └── test_neuralnetwork.cpp
│   │   └── functional/
│   │       └── mnist_test.cpp
│   ├── build/            # Build output (auto-generated)
│   └── CMakeLists.txt
├── csv/
│   ├── mnist_train.csv
│   └── mnist_test.csv
└── .vscode/
    ├── tasks.json        # Build and test tasks
    ├── launch.json       # Debug configurations
    ├── settings.json     # Workspace settings
    └── c_cpp_properties.json  # C++ IntelliSense config
```

## Quick Start

1. **Open the project** in VS Code
2. **Install recommended extensions** when prompted
3. **Press `Ctrl+Shift+P`** → "CMake: Configure" (or it will auto-configure)
4. **Press `Ctrl+Shift+B`** to build the project
5. **Press `Ctrl+Shift+P`** → "Tasks: Run Task" → "Run All Tests"

## IntelliSense Configuration

The project is configured for optimal IntelliSense support:
- C++17 standard
- Eigen3 headers included
- Google Test headers included
- Compile commands exported for better code analysis

## CMake Integration

The CMake Tools extension provides:
- **Build variants** (Debug/Release)
- **Target selection** (library, tests, etc.)
- **CTest integration** for running tests
- **Automatic configuration** when opening the project

## Testing Integration

Tests are integrated with VS Code in multiple ways:
- **Tasks** for running different test suites
- **Debug configurations** for debugging tests
- **Terminal integration** for test output
- **Problem matcher** for parsing test results

## Tips

- Use `Ctrl+Shift+P` → "CMake: Build" for a quick build
- Use `Ctrl+Shift+P` → "CMake: Run Tests" for CTest integration
- Use `F5` to start debugging any test configuration
- The status bar shows CMake build status and active configuration
- Use `Ctrl+Shift+P` → "CMake: Clean" to clean build artifacts

## Troubleshooting

### IntelliSense Issues
- Ensure the CMake project is configured
- Check that `compile_commands.json` exists in the build directory
- Restart the C++ extension: `Ctrl+Shift+P` → "C++: Restart IntelliSense"

### Build Issues
- Check that all dependencies are installed
- Verify the build directory exists and is configured
- Use "Clean Build" task to start fresh

### Test Issues
- Ensure the project is built before running tests
- Check that CSV files are in the `csv/` directory
- Verify working directory settings in tasks.json
