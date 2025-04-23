# TOP Project

This project implements various benchmarks and profiling tools to evaluate performance optimizations, including cache blocking, GPU implementations, and layout optimizations. It leverages libraries such as Kokkos and Vulkan for high-performance computing.

## Project Structure

- **benchmarks/**: Contains benchmark implementations (e.g., `cache_blocking.cpp`, `gpu_implem.cpp`).
- **culkan/**: Custom vulkan wrapper for compute shaders
- **profilings/**: Programs used as profilees for cache analysis.
- **results/**: Contains the results of the benchmarks.
- **scripts/**: Python scripts for analyzing and plotting results.
- **src/**: Core source files for the project. Contains the CPU implementation at matrix_product.hpp and GPU implementation at operation.comp
- **tests/**: Unit tests for validating implementations.

## Build Instructions

There are 2 configurations to build the project:
1. Release: Compile with optimisations enabled (-O3)
2. Profiling: Compile with optimisations enabled and profiling enabled (-O3 -pg)

Then, you can build using CMake:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Then you can launch the benchmarks:
```bash
./build/benchmarks/top.xxxx
```
or the profiling:
```bash
perf ... ./build/benchmarks/top.xxxx
```
or the tests:
```bash
./build/tests/top_tests.xxxx
```