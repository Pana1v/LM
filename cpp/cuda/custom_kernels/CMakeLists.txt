cmake_minimum_required(VERSION 3.10)
project(LevenbergMarquardtCUDACustom)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Find CUDA
find_package(CUDA REQUIRED)

# Add executable
add_executable(lm_cuda_custom 
    main.cu
    lm_kernels.cu
)

# Set CUDA properties
set_property(TARGET lm_cuda_custom PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Set build type to Release for optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lm_cuda_custom PRIVATE -O3 -march=native)
    set_property(TARGET lm_cuda_custom PROPERTY CUDA_ARCHITECTURES "75;80;86;89;90")
endif()

if(MSVC)
    target_compile_options(lm_cuda_custom PRIVATE /O2)
endif()

