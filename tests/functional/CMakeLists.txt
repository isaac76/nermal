# Functional tests for full system behavior

# MNIST Neural Network Test
add_executable(mnist_test
    mnist_test.cpp
    ${CMAKE_SOURCE_DIR}/src/neuralnetwork.cpp
    ${CMAKE_SOURCE_DIR}/src/neuralnetwork.h
)

# Include directories
target_include_directories(mnist_test PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIRS}
)

# Add test to CTest
add_test(NAME mnist_functional_test COMMAND mnist_test)

# Set test properties
set_tests_properties(mnist_functional_test PROPERTIES
    TIMEOUT 120
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PASS_REGULAR_EXPRESSION "Accuracy: [0-9]+%"
)

# Optional: Add a quick functional test with fewer samples
add_executable(mnist_quick_test
    mnist_test.cpp
    ${CMAKE_SOURCE_DIR}/src/neuralnetwork.cpp
    ${CMAKE_SOURCE_DIR}/src/neuralnetwork.h
)

target_include_directories(mnist_quick_test PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIRS}
)

# Define a preprocessor macro to limit samples for quick test
target_compile_definitions(mnist_quick_test PRIVATE QUICK_TEST=1)

add_test(NAME mnist_quick_test COMMAND mnist_quick_test)
set_tests_properties(mnist_quick_test PROPERTIES
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PASS_REGULAR_EXPRESSION "Accuracy: [0-9]+%"
)
