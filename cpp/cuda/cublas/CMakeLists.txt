cmake_minimum_required(VERSION 3.10)
project(LevenbergMarquardtCUDACuBLAS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Find CUDA
find_package(CUDA REQUIRED)

# Find cuBLAS and cuSOLVER
find_library(CUBLAS_LIBRARY cublas PATHS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib)
find_library(CUSOLVER_LIBRARY cusolver PATHS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib)

# Add executable
add_executable(lm_cuda_cublas lm_cublas.cu)

# Link libraries
target_link_libraries(lm_cuda_cublas ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY})

# Include directories
target_include_directories(lm_cuda_cublas PRIVATE ${CUDA_INCLUDE_DIRS})

# Set CUDA properties
set_property(TARGET lm_cuda_cublas PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Set build type to Release for optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lm_cuda_cublas PRIVATE -O3 -march=native)
    set_property(TARGET lm_cuda_cublas PROPERTY CUDA_ARCHITECTURES "75;80;86;89;90")
endif()

if(MSVC)
    target_compile_options(lm_cuda_cublas PRIVATE /O2)
endif()

