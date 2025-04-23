/**
 * @file profilings/final_results_first.cpp
 * @brief Used to compare the cache hit ratio of the very first implementation of the matrix product for the final results.
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

	// Generate A, B, C with the right layout
	RightMatrix A_right = RightMatrix("A_right", m, k);
	RightMatrix B_right = RightMatrix("B_right", k, n);
	RightMatrix C_right = RightMatrix("C_right", m, n);
	matrix_init(A_right);
	matrix_init(B_right);
	matrix_init(C_right);

	// Generate alpha and beta
	double alpha = drand48();
	double beta  = drand48();

	// Do a few runs
	for (int i = 0; i < 10; i++) {
		Kokkos::fence();
		matrix_product_reference(alpha, A_right, B_right, beta, C_right);
		Kokkos::fence();
	}

	Kokkos::finalize();
	exit(EXIT_SUCCESS);
}