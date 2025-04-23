# Profile the layout layout
add_executable(top.worst_layout)
target_sources(top.worst_layout PRIVATE worst_layout.cpp)
target_include_directories(top.worst_layout PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(top.worst_layout PRIVATE Kokkos::kokkos fmt::fmt nanobench::nanobench)

# Profile the first implementation
add_executable(top.final_results_first)
target_sources(top.final_results_first PRIVATE final_results_first.cpp)
target_include_directories(top.final_results_first PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(top.final_results_first PRIVATE Kokkos::kokkos fmt::fmt nanobench::nanobench)

# Profile the layout improvement
add_executable(top.final_results_layout)
target_sources(top.final_results_layout PRIVATE final_results_layout.cpp)
target_include_directories(top.final_results_layout PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(top.final_results_layout PRIVATE Kokkos::kokkos fmt::fmt nanobench::nanobench)

# Profile the cache blocking version
add_executable(top.final_results_cache_blocking)
target_sources(top.final_results_cache_blocking PRIVATE final_results_cache_blocking.cpp)
target_include_directories(top.final_results_cache_blocking PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(top.final_results_cache_blocking PRIVATE Kokkos::kokkos fmt::fmt nanobench::nanobench)
