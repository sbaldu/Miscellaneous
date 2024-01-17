cmake_minimum_required(VERSION 3.16)
project(mandelbrot_serial)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(Executable mandelbrot_serial_debug.out)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(Executable mandelbrot_serial_release.out)
endif()

add_executable(${Executable} Mandelbrot.cpp)
target_link_libraries(${Executable} PRIVATE sfml-graphics)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  execute_process(COMMAND ln -sf build/serial/${Executable} mandelbrot_serial_debug.out)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  execute_process(COMMAND ln -sf build/serial/${Executable} mandelbrot_serial_release.out)
endif()
