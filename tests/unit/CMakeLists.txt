# Unit tests for individual components

# Find Qt Test component  
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Test)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Test)

# Create a simple unit test for the NeuralNetwork class
add_executable(test_neuralnetwork
    test_neuralnetwork.cpp
    ${CMAKE_SOURCE_DIR}/src/neuralnetwork.cpp
)

# Include directories
target_include_directories(test_neuralnetwork PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIRS}
    /usr/include/qt6
)

# Link Qt Test framework
target_link_libraries(test_neuralnetwork PRIVATE
    Qt${QT_VERSION_MAJOR}::Test
)

# Add test to CTest
add_test(NAME test_neuralnetwork COMMAND test_neuralnetwork)

# Set test properties
set_tests_properties(test_neuralnetwork PROPERTIES
    TIMEOUT 30
)
