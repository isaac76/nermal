# CMake generated Testfile for 
# Source directory: /home/isaac/src/git/nermal/tests/functional
# Build directory: /home/isaac/src/git/nermal/build/tests/functional
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(mnist_functional_test "/home/isaac/src/git/nermal/build/tests/functional/mnist_test")
set_tests_properties(mnist_functional_test PROPERTIES  PASS_REGULAR_EXPRESSION "Accuracy: [0-9]+%" TIMEOUT "120" WORKING_DIRECTORY "/home/isaac/src/git/nermal" _BACKTRACE_TRIPLES "/home/isaac/src/git/nermal/tests/functional/CMakeLists.txt;17;add_test;/home/isaac/src/git/nermal/tests/functional/CMakeLists.txt;0;")
add_test(mnist_quick_test "/home/isaac/src/git/nermal/build/tests/functional/mnist_quick_test")
set_tests_properties(mnist_quick_test PROPERTIES  PASS_REGULAR_EXPRESSION "Accuracy: [0-9]+%" TIMEOUT "60" WORKING_DIRECTORY "/home/isaac/src/git/nermal" _BACKTRACE_TRIPLES "/home/isaac/src/git/nermal/tests/functional/CMakeLists.txt;41;add_test;/home/isaac/src/git/nermal/tests/functional/CMakeLists.txt;0;")
