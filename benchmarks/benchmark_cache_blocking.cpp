/**
 * @file benchmarks/benchmark_layout_minus_outliers.cpp
 * @brief Benchmark for matrix product with different layouts minus the 2 outliers. Also uses bigger matrices.
 */

#include "matrix_product.hpp"

#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <fmt/core.h>
#include <nanobench.h>

#include <iostream>

auto main(int argc, char* argv[]) -> int {
	Kokkos::initialize(argc, argv);

	// Known seed for deterministic RNG
	srand48(42);

	// Dimensions of the matrices
	int m = 2000;
	int n = 2000;
	int k = 2000;

	// Generate A, B, C
	RightMatrix A = RightMatrix("A", m, k);
	LeftMatrix B  = LeftMatrix("B", k, n);
	RightMatrix C = RightMatrix("C", m, n);
	matrix_init(A);
	matrix_init(B);
	matrix_init(C);

	// Generate alpha and beta
	double alpha = drand48();
	double beta  = drand48();

	// Compare all the different layout combinations
	std::ostringstream oss;
	auto result = ankerl::nanobench::Bench()
			  .minEpochIterations(5)
			  .performanceCounters(true)
			  .output(&oss)
			  // Binary combinations of A, B, C (r/l with right/left layout)
			  .run("Reference", [&]() { matrix_product_reference(alpha, A, B, beta, C); })
			  .run("Cache Blocked i32", [&]() { matrix_product_cache_blocked_i(alpha, A, B, beta, C, 32); })
			  .run("Cache Blocked ij32", [&]() { matrix_product_cache_blocked_i(alpha, A, B, beta, C, 32); })
			  .run("Cache Blocked ijk32", [&]() { matrix_product_cache_blocked_i(alpha, A, B, beta, C, 32); })
			  .doNotOptimizeAway(A)
			  .doNotOptimizeAway(B)
			  .doNotOptimizeAway(C)
			  .doNotOptimizeAway(alpha)
			  .doNotOptimizeAway(beta)
			  .results();

	// Print oss
	// std::cout << oss.str() << '\n';

	for (auto const& res : result) {
		auto measure = res.fromString("elapsed");
		auto name    = res.config().mBenchmarkName;
		fmt::println("{}, Min: {}s, Max: {}s, Med: {}s", name, res.minimum(measure), res.maximum(measure), res.median(measure));
	}

	Kokkos::finalize();
	exit(EXIT_SUCCESS);
}