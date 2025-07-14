# Unit tests for individual components

# Create a simple unit test for the NeuralNetwork class
add_executable(test_neuralnetwork
    test_neuralnetwork.cpp
)

# Set C++ standard for the test
set_target_properties(test_neuralnetwork PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Link against the nermal library and Google Test
target_link_libraries(test_neuralnetwork PRIVATE
    nermal::nermal
    /usr/lib64/libgtest.so
    /usr/lib64/libgtest_main.so
    pthread
)

# Add Google Test include directory
target_include_directories(test_neuralnetwork PRIVATE /usr/include)

# Add test to CTest
add_test(NAME test_neuralnetwork COMMAND test_neuralnetwork)

# Set test properties
set_tests_properties(test_neuralnetwork PROPERTIES
    TIMEOUT 30
)
