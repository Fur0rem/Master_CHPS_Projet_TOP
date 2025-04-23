cmake_minimum_required(VERSION 3.25)
project(top.matrix_product LANGUAGES CXX)

# Define custom build types
set(CMAKE_CONFIGURATION_TYPES "Release;Profiling" CACHE STRING "Available build types" FORCE)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Compiler flags for each build type
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra -Wpedantic -std=c++20")
set(CMAKE_CXX_FLAGS_PROFILING "-O3 -pg -Wall -Wextra -Wpedantic")

# Add custom build type for Profiling
if(CMAKE_BUILD_TYPE STREQUAL "Profiling")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_PROFILING}")
endif()

find_package(OpenMP REQUIRED)

include(FetchContent)

FetchContent_Declare(Kokkos GIT_REPOSITORY https://github.com/kokkos/kokkos GIT_TAG 4.6.00)
# Choose OpenMP for parallel Kokkos backend
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP" FORCE)
set(Kokkos_ENABLE_THREADS OFF CACHE BOOL "Enable Threads" FORCE)
FetchContent_MakeAvailable(Kokkos)

FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt GIT_TAG 11.1.4)
FetchContent_MakeAvailable(fmt)

# Use nanobench for benchmarking
FetchContent_Declare(nanobench GIT_REPOSITORY https://github.com/martinus/nanobench/ GIT_TAG v4.3.11)
FetchContent_MakeAvailable(nanobench)

add_subdirectory(src)
add_subdirectory(culkan)
add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(profilings)

# Use Vulkan for GPU implementation
find_package(Vulkan REQUIRED)

# Add Vulkan include directories
target_include_directories(top.check_gpu_implem PRIVATE ${Vulkan_INCLUDE_DIRS})
target_include_directories(top.gpu_implem PRIVATE ${Vulkan_INCLUDE_DIRS})

# Link Vulkan libraries
target_link_libraries(top.check_gpu_implem PRIVATE ${Vulkan_LIBRARIES})
target_link_libraries(top.gpu_implem PRIVATE ${Vulkan_LIBRARIES})