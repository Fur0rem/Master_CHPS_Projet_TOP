# Benchmarking all layouts
add_executable(top.layout_all)
target_sources(top.layout_all PRIVATE layout_all.cpp)
target_include_directories(top.layout_all PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(top.layout_all PRIVATE Kokkos::kokkos fmt::fmt nanobench::nanobench)

# Benchmarking all layouts minus the outliers
add_executable(top.layout_minus_outliers)
target_sources(top.layout_minus_outliers PRIVATE layout_minus_outliers.cpp)
target_include_directories(top.layout_minus_outliers PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(top.layout_minus_outliers PRIVATE Kokkos::kokkos fmt::fmt nanobench::nanobench)

# Benchmarking the cache blocking version
add_executable(top.cache_blocking)
target_sources(top.cache_blocking PRIVATE cache_blocking.cpp)
target_include_directories(top.cache_blocking PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(top.cache_blocking PRIVATE Kokkos::kokkos fmt::fmt nanobench::nanobench)

# Benchmarking the GPU implementation
add_executable(top.gpu_implem)
target_sources(top.gpu_implem PRIVATE gpu_implem.cpp)
target_include_directories(top.gpu_implem PRIVATE ${CMAKE_SOURCE_DIR}/culkan)
target_include_directories(top.gpu_implem PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(top.gpu_implem PRIVATE Kokkos::kokkos fmt::fmt nanobench::nanobench)