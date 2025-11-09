cmake_minimum_required(VERSION 3.10)
project(LevenbergMarquardtNoEigen)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(lm_no_eigen main.cpp)

# Set build type to Release for optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lm_no_eigen PRIVATE -O3 -march=native)
endif()

if(MSVC)
    target_compile_options(lm_no_eigen PRIVATE /O2)
endif()

