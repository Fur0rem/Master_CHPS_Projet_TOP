/**
 * @file profilings/worst_layout.cpp
 * @brief Used for profiling cache misses and performance of the worst layout for matrix product.
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
	int m = 500;
	int n = 500;
	int k = 500;

	// Generate A, B, C with the worst layout (A left, B right, C left)
	LeftMatrix A_left   = LeftMatrix("A_left", m, k);
	RightMatrix B_right = RightMatrix("B_right", k, n);
	LeftMatrix C_left   = LeftMatrix("C_left", m, n);
	matrix_init(A_left);
	matrix_init(B_right);
	matrix_init(C_left);

	// Generate alpha and beta
	double alpha = drand48();
	double beta  = drand48();

	// Do a few runs
	for (int i = 0; i < 10; i++) {
		Kokkos::fence();
		matrix_product_reference(alpha, A_left, B_right, beta, C_left);
		Kokkos::fence();
	}

	Kokkos::finalize();
	exit(EXIT_SUCCESS);
}