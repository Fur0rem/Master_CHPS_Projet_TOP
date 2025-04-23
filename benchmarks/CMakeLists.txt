# Benchmarking all layouts
add_executable(top.benchmark_layout_all)
target_sources(top.benchmark_layout_all PRIVATE benchmark_layout_all.cpp)
target_include_directories(top.benchmark_layout_all PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(top.benchmark_layout_all PRIVATE -Wall -Wextra -Wpedantic -O3)
target_link_libraries(top.benchmark_layout_all PRIVATE Kokkos::kokkos fmt::fmt nanobench::nanobench)

# Benchmarking all layouts minus the outliers
add_executable(top.benchmark_layout_minus_outliers)
target_sources(top.benchmark_layout_minus_outliers PRIVATE benchmark_layout_minus_outliers.cpp)
target_include_directories(top.benchmark_layout_minus_outliers PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(top.benchmark_layout_minus_outliers PRIVATE -Wall -Wextra -Wpedantic -O3)
target_link_libraries(top.benchmark_layout_minus_outliers PRIVATE Kokkos::kokkos fmt::fmt nanobench::nanobench)

# Benchmarking the cache blocking version
add_executable(top.benchmark_cache_blocking)
target_sources(top.benchmark_cache_blocking PRIVATE benchmark_cache_blocking.cpp)
target_include_directories(top.benchmark_cache_blocking PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(top.benchmark_cache_blocking PRIVATE -Wall -Wextra -Wpedantic -O3)
target_link_libraries(top.benchmark_cache_blocking PRIVATE Kokkos::kokkos fmt::fmt nanobench::nanobench)
