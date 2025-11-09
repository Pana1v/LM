cmake_minimum_required(VERSION 3.10)
project(LevenbergMarquardtEigen)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
# Try multiple methods to find Eigen3
if(DEFINED ENV{EIGEN3_ROOT})
    set(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_ROOT}")
    message(STATUS "Using Eigen3 from EIGEN3_ROOT: ${EIGEN3_INCLUDE_DIR}")
elseif(EXISTS "C:/Eigen3/Eigen")
    set(EIGEN3_INCLUDE_DIR "C:/Eigen3")
    message(STATUS "Using Eigen3 from default location: ${EIGEN3_INCLUDE_DIR}")
else()
    find_package(Eigen3 REQUIRED)
endif()

# Add executable
add_executable(lm_eigen main.cpp)

# Include Eigen3
if(EIGEN3_INCLUDE_DIR)
    target_include_directories(lm_eigen PRIVATE ${EIGEN3_INCLUDE_DIR})
else()
    target_link_libraries(lm_eigen Eigen3::Eigen)
endif()

# Set build type to Release for optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lm_eigen PRIVATE -O3 -march=native)
endif()

if(MSVC)
    target_compile_options(lm_eigen PRIVATE /O2)
endif()

