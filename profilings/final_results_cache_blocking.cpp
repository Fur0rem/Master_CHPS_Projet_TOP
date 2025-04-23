/**
 * @file profilings/final_results_cache_blocking.cpp
 * @brief Used to compare the cache hit ratio of the matrix product with cache blocking for the final results.
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

	// Generate A, B, C with the best layout (A right, B left, C right)
	RightMatrix A_right = RightMatrix("A_right", m, k);
	LeftMatrix B_left   = LeftMatrix("B_left", k, n);
	RightMatrix C_right = RightMatrix("C_right", m, n);
	matrix_init(A_right);
	matrix_init(B_left);
	matrix_init(C_right);

	// Generate alpha and beta
	double alpha = drand48();
	double beta  = drand48();

	// Do a few runs
	for (int i = 0; i < 10; i++) {
		Kokkos::fence();
		matrix_product_cache_blocked_ij(alpha, A_right, B_left, beta, C_right, 8);
		Kokkos::fence();
	}

	Kokkos::finalize();
	exit(EXIT_SUCCESS);
}