cmake_minimum_required(VERSION 3.25)
project(top.matrix_product LANGUAGES CXX)

find_package(OpenMP REQUIRED)

include(FetchContent)

FetchContent_Declare(Kokkos GIT_REPOSITORY https://github.com/kokkos/kokkos GIT_TAG 4.6.00)
# Choose OpenMP for parallel Kokkos backend
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP" FORCE)
set(Kokkos_ENABLE_THREADS OFF CACHE BOOL "Enable Threads" FORCE)
FetchContent_MakeAvailable(Kokkos)

FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt GIT_TAG 11.1.4)
FetchContent_MakeAvailable(fmt)

# Use nanobench for benckmarking
FetchContent_Declare(nanobench GIT_REPOSITORY https://github.com/martinus/nanobench/ GIT_TAG v4.3.11)
FetchContent_MakeAvailable(nanobench)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)
