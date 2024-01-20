cmake_minimum_required(VERSION 3.16)
project(mandelbrot_cuda)

# Check if CUDA is available
include(CheckLanguage)
check_language(CUDA)

# Enable CUDA
enable_language(CUDA)

# Require cuda to compile the test file
find_package(CUDA REQUIRED)

# Set the CUDA standard
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set the C++ nvcc flags
string(APPEND CUDA_NVCC_FLAGS "-O3 -g -std=c++17")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(Executable mandelbrot_cuda_debug.out)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(Executable mandelbrot_cuda_release.out)
endif()

CUDA_ADD_EXECUTABLE(${Executable} Mandelbrot.cu)
target_link_libraries(${Executable} sfml-graphics)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  execute_process(COMMAND ln -sf build/cuda/${Executable} mandelbrot_cuda_debug.out)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  execute_process(COMMAND ln -sf build/cuda/${Executable} mandelbrot_cuda_release.out)
endif()
