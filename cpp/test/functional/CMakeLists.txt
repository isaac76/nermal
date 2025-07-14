# Functional tests for full system behavior

# MNIST Neural Network Test
add_executable(mnist_test
    mnist_test.cpp
)

# Set C++ standard for the test
set_target_properties(mnist_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Link against the nermal library
target_link_libraries(mnist_test PRIVATE
    nermal::nermal
)

# Add test to CTest
add_test(NAME mnist_functional_test COMMAND mnist_test)

# Set test properties
set_tests_properties(mnist_functional_test PROPERTIES
    TIMEOUT 120
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
    PASS_REGULAR_EXPRESSION "Accuracy: [0-9]+%"
)

# Optional: Add a quick functional test with fewer samples
add_executable(mnist_quick_test
    mnist_test.cpp
)

# Set C++ standard for the test
set_target_properties(mnist_quick_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Link against the nermal library
target_link_libraries(mnist_quick_test PRIVATE
    nermal::nermal
)

# Define a preprocessor macro to limit samples for quick test
target_compile_definitions(mnist_quick_test PRIVATE QUICK_TEST=1)

add_test(NAME mnist_quick_test COMMAND mnist_quick_test)
set_tests_properties(mnist_quick_test PROPERTIES
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
    PASS_REGULAR_EXPRESSION "Accuracy: [0-9]+%"
)
