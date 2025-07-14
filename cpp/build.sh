#!/bin/bash

# Nermal Neural Network Library Build Script

set -e  # Exit on any error

# Default values
BUILD_TYPE="Release"
INSTALL_PREFIX="/usr/local"
BUILD_TESTS="ON"
BUILD_DIR="build"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --install-prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --no-tests)
            BUILD_TESTS="OFF"
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Nermal Neural Network Library Build Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug              Build in Debug mode (default: Release)"
            echo "  --install-prefix DIR Install prefix (default: /usr/local)"
            echo "  --no-tests          Don't build tests"
            echo "  --build-dir DIR     Build directory (default: build)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Build release version"
            echo "  $0 --debug --no-tests                # Build debug without tests"
            echo "  $0 --install-prefix ~/.local         # Install to user directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== Nermal Neural Network Library Build ==="
echo "Build type: $BUILD_TYPE"
echo "Install prefix: $INSTALL_PREFIX"
echo "Build tests: $BUILD_TESTS"
echo "Build directory: $BUILD_DIR"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DBUILD_TESTING="$BUILD_TESTS"

# Build
echo "Building..."
cmake --build . --config "$BUILD_TYPE" -j$(nproc)

# Run tests if enabled
if [ "$BUILD_TESTS" = "ON" ]; then
    echo "Running tests..."
    ctest --output-on-failure
fi

echo ""
echo "=== Build completed successfully! ==="
echo ""
echo "To install the library, run:"
echo "  cd $BUILD_DIR && sudo cmake --install ."
echo ""
echo "Or to install to a custom prefix:"
echo "  cd $BUILD_DIR && cmake --install . --prefix /your/custom/path"
echo ""
echo "Generated files:"
echo "  Static library:  $BUILD_DIR/libnermal.a"
echo "  Shared library:  $BUILD_DIR/libnermal.so"
