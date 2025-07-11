#!/bin/bash
# Script to fix Qt Creator configuration for Nermal project
# This ensures GCC 11 and proper Debug build type are used

echo "Fixing Qt Creator configuration for Nermal project..."

# Clean Qt Creator build directories
echo "Cleaning Qt Creator build directories..."
rm -rf build-Nermal-Desktop-Debug
rm -rf build-Nermal-Desktop-Release

# Create fresh Debug build directory
echo "Creating fresh Debug build directory..."
mkdir build-Nermal-Desktop-Debug
cd build-Nermal-Desktop-Debug

# Configure with proper settings
echo "Configuring with GCC 11 and Debug build type..."
cmake -DCMAKE_TOOLCHAIN_FILE=../gcc11-toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ..

cd ..

echo "Configuration complete!"
echo ""
echo "Now in Qt Creator:"
echo "1. Open the project (File → Open File or Project → CMakeLists.txt)"
echo "2. When prompted, choose 'Configure Project'"
echo "3. In Build & Run settings, ensure:"
echo "   - Build directory points to: build-Nermal-Desktop-Debug"
echo "   - CMAKE_BUILD_TYPE is set to Debug"
echo "   - Toolchain file is set to: gcc11-toolchain.cmake"
echo ""
echo "If you still see errors, go to Projects → Manage Kits and ensure"
echo "a kit with GCC 11 is selected."
