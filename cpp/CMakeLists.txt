cmake_minimum_required(VERSION 3.16)

# Project setup
project(nermal VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform-specific settings
if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Find dependencies
find_package(Eigen3 REQUIRED)

# For testing, we'll use Google Test
if(BUILD_TESTING)
    # Try standard CMake approach first
    find_package(GTest QUIET)
    
    if(NOT GTest_FOUND)
        # Try pkg-config approach (works well on openSUSE)
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(GTEST gtest)
            pkg_check_modules(GTEST_MAIN gtest_main)
            if(GTEST_FOUND AND GTEST_MAIN_FOUND)
                message(STATUS "Found Google Test via pkg-config")
            endif()
        endif()
    else()
        message(STATUS "Found Google Test via CMake")
    endif()
    
    # Final fallback: download via FetchContent
    if(NOT GTest_FOUND AND NOT GTEST_FOUND)
        message(STATUS "Google Test not found, downloading via FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/03597a01ee50f33f9b9fd98d1ac6e7ddb1cf86d1.zip
        )
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        set(GTEST_FOUND TRUE)
    endif()
endif()

# Library source files
set(NERMAL_SOURCES
    src/neuralnetwork.cpp
)

set(NERMAL_HEADERS
    src/neuralnetwork.h
)

# Create shared library (.so/.dll/.dylib)
add_library(nermal_shared SHARED ${NERMAL_SOURCES})
target_include_directories(nermal_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(nermal_shared PUBLIC Eigen3::Eigen)
set_target_properties(nermal_shared PROPERTIES
    OUTPUT_NAME nermal
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# Create static library (.a/.lib)
add_library(nermal_static STATIC ${NERMAL_SOURCES})
target_include_directories(nermal_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(nermal_static PUBLIC Eigen3::Eigen)
set_target_properties(nermal_static PROPERTIES
    OUTPUT_NAME nermal
    POSITION_INDEPENDENT_CODE ON
)

# Create alias targets for convenience
add_library(nermal::shared ALIAS nermal_shared)
add_library(nermal::static ALIAS nermal_static)
add_library(nermal::nermal ALIAS nermal_shared)  # Default to shared

# Enable testing
enable_testing()

# Add test subdirectory
add_subdirectory(test)

# Installation configuration
include(GNUInstallDirs)

# Install libraries
install(TARGETS nermal_shared nermal_static
    EXPORT nermalTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(FILES ${NERMAL_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nermal
)

# Install CMake config files
install(EXPORT nermalTargets
    FILE nermalTargets.cmake
    NAMESPACE nermal::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nermal
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    nermalConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/nermalConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/nermalConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nermal
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/nermalConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/nermalConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nermal
)

# Optionally, create a pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/nermal.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/nermal.pc
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nermal.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# Print summary
message(STATUS "Nermal Neural Network Library Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Libraries will be installed to: ${CMAKE_INSTALL_FULL_LIBDIR}")
message(STATUS "  Headers will be installed to: ${CMAKE_INSTALL_FULL_INCLUDEDIR}/nermal")
