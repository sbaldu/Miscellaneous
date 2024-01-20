cmake_minimum_required(VERSION 3.16)
project(Mandelbrot VERSION 1.0.0)

find_package(SFML 2.5 COMPONENTS graphics REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -O3")

string(APPEND CMAKE_CXX_FLAGS_DEBUG
       " -fsanitize=address -fno-omit-frame-pointer")
string(APPEND CMAKE_EXE_LINKER_FLAGS_DEBUG
       " -fsanitize=address -fno-omit-frame-pointer")

add_subdirectory(serial)
add_subdirectory(tbb)
add_subdirectory(cuda)
