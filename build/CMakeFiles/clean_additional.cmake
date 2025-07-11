# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "src/CMakeFiles/Nermal_autogen.dir/AutogenUsed.txt"
  "src/CMakeFiles/Nermal_autogen.dir/ParseCache.txt"
  "src/Nermal_autogen"
  "tests/functional/CMakeFiles/mnist_quick_test_autogen.dir/AutogenUsed.txt"
  "tests/functional/CMakeFiles/mnist_quick_test_autogen.dir/ParseCache.txt"
  "tests/functional/CMakeFiles/mnist_test_autogen.dir/AutogenUsed.txt"
  "tests/functional/CMakeFiles/mnist_test_autogen.dir/ParseCache.txt"
  "tests/functional/mnist_quick_test_autogen"
  "tests/functional/mnist_test_autogen"
  "tests/unit/CMakeFiles/test_neuralnetwork_autogen.dir/AutogenUsed.txt"
  "tests/unit/CMakeFiles/test_neuralnetwork_autogen.dir/ParseCache.txt"
  "tests/unit/test_neuralnetwork_autogen"
  )
endif()
