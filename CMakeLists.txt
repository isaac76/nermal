cmake_minimum_required(VERSION 3.5)

# Use newer compiler for C++17 support
set(CMAKE_C_COMPILER gcc-11)
set(CMAKE_CXX_COMPILER g++-11)

project(Nermal VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Test)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Test)

# Try to find Eigen3 via pkg-config first, then fallback to system headers
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(EIGEN3 eigen3)
endif()

# If not found via pkg-config, try to find it manually
if(NOT EIGEN3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Core
        PATHS /usr/include/eigen3 /usr/local/include/eigen3
        PATH_SUFFIXES eigen3
    )
    if(EIGEN3_INCLUDE_DIR)
        set(EIGEN3_FOUND TRUE)
        set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
        message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
    endif()
endif()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
