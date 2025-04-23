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
	int m = 1000;
	int n = 1000;
	int k = 1000;

	// Generate A, B, C, with right layout
	RightMatrix A_right = RightMatrix("A_right", m, k);
	RightMatrix B_right = RightMatrix("B_right", k, n);
	RightMatrix C_right = RightMatrix("C_right", m, n);
	matrix_init(A_right);
	matrix_init(B_right);
	matrix_init(C_right);

	// Generate A, B, C, with left layout
	LeftMatrix A_left = LeftMatrix("A_left", m, k);
	LeftMatrix B_left = LeftMatrix("B_left", k, n);
	LeftMatrix C_left = LeftMatrix("C_left", m, n);
	matrix_init(A_left);
	matrix_init(B_left);
	matrix_init(C_left);

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
			  .run("Ar_Br_Cr", [&]() { matrix_product_reference(alpha, A_right, B_right, beta, C_right); })
			  .run("Ar_Br_Cl", [&]() { matrix_product_reference(alpha, A_right, B_right, beta, C_left); })
			  .run("Ar_Bl_Cr", [&]() { matrix_product_reference(alpha, A_right, B_left, beta, C_right); })
			  .run("Ar_Bl_Cl", [&]() { matrix_product_reference(alpha, A_right, B_left, beta, C_left); })
			  .run("Al_Bl_Cr", [&]() { matrix_product_reference(alpha, A_left, B_left, beta, C_right); })
			  .run("Al_Bl_Cl", [&]() { matrix_product_reference(alpha, A_left, B_left, beta, C_left); })
			  .doNotOptimizeAway(A_right)
			  .doNotOptimizeAway(B_right)
			  .doNotOptimizeAway(C_right)
			  .doNotOptimizeAway(A_left)
			  .doNotOptimizeAway(B_left)
			  .doNotOptimizeAway(C_left)
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