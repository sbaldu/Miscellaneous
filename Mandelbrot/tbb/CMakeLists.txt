cmake_minimum_required(VERSION 3.16)
project(mandelbrot_tbb)

find_package(TBB REQUIRED)

if (NOT TBB_FOUND)
		message(FATAL_ERROR "TBB not found")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(Exetutable mandelbrot_tbb_debug.out)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(Exetutable mandelbrot_tbb_release.out)
endif()

add_executable(${Exetutable} Mandelbrot.cpp)
target_link_libraries(${Exetutable} PRIVATE TBB::tbb sfml-graphics)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  execute_process(COMMAND ln -sf build/tbb/${Exetutable} mandelbrot_tbb_debug.out)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  execute_process(COMMAND ln -sf build/tbb/${Exetutable} mandelbrot_tbb_release.out)
endif()
