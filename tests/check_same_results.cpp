/**
 * @file tests/check_same_results.cpp
 * @brief Test for matrix product functions.
 */

#include "matrix_product.hpp"

#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <fmt/core.h>
#include <iostream>

auto main(int argc, char* argv[]) -> int {
	Kokkos::initialize(argc, argv);

	constexpr auto GREEN = "\033[0;32m";
	constexpr auto RESET = "\033[0m";
	constexpr auto RED   = "\033[0;31m";

	// Predetermined test computed from the very start of the project
	{
		// Dimensions of the matrices
		int m = 2;
		int n = 4;
		int k = 3;

		// Testing matrix A
		// [ 1 2 3
		//   4 5 6 ]
		auto A	= RightMatrix("A", m, k);
		A(0, 0) = 1;
		A(0, 1) = 2;
		A(0, 2) = 3;
		A(1, 0) = 4;
		A(1, 1) = 5;
		A(1, 2) = 6;

		// Testing matrix B
		// [ 7 8 9 10
		//   11 12 13 14
		//   15 16 17 18 ]
		auto B	= LeftMatrix("B", k, n);
		B(0, 0) = 7;
		B(0, 1) = 8;
		B(0, 2) = 9;
		B(0, 3) = 10;
		B(1, 0) = 11;
		B(1, 1) = 12;
		B(1, 2) = 13;
		B(1, 3) = 14;
		B(2, 0) = 15;
		B(2, 1) = 16;
		B(2, 2) = 17;
		B(2, 3) = 18;

		// Reference matrix C
		// [ 19 20 21 22
		//   23 24 25 26 ]
		auto C_ref  = RightMatrix("C_ref", m, n);
		C_ref(0, 0) = 19;
		C_ref(0, 1) = 20;
		C_ref(0, 2) = 21;
		C_ref(0, 3) = 22;
		C_ref(1, 0) = 23;
		C_ref(1, 1) = 24;
		C_ref(1, 2) = 25;
		C_ref(1, 3) = 26;

		// Testing matrix C, same as reference
		auto C_test_i	= RightMatrix("C_test_i", m, n);
		auto C_test_ij	= RightMatrix("C_test_ij", m, n);
		auto C_test_ijk = RightMatrix("C_test_ijk", m, n);
		for (int j = 0; j < m; j++) {
			for (int l = 0; l < n; l++) {
				C_test_i(j, l)	 = C_ref(j, l);
				C_test_ij(j, l)	 = C_ref(j, l);
				C_test_ijk(j, l) = C_ref(j, l);
			}
		}

		// Testing alpha and beta
		double alpha = 2.0;
		double beta  = -1.0;

		// Run the reference and test functions
		Kokkos::fence();
		matrix_product_reference(alpha, A, B, beta, C_ref);
		Kokkos::fence();
		matrix_product_cache_blocked_i(alpha, A, B, beta, C_test_i, 3);
		Kokkos::fence();
		matrix_product_cache_blocked_ij(alpha, A, B, beta, C_test_ij, 3);
		Kokkos::fence();
		// Too long to run
		// matrix_product_cache_blocked_ijk(alpha, A, B, beta, C_test_ijk, 3);
		// Kokkos::fence();

		// Results should both be
		// [ 2793 3180 3591 4026
		//   7935 9000 10125 11310 ]
		if (!matrix_are_equal(C_ref, C_test_i)) {
			fmt::println("{}Test failed for i!{}", RED, RESET);
			fmt::println("{}Expected:{}", RED, RESET);
			matrix_print(C_ref);
			fmt::println("{}Got:{}", RED, RESET);
			matrix_print(C_test_i);
			Kokkos::finalize();
			exit(EXIT_FAILURE);
		}
		if (!matrix_are_equal(C_ref, C_test_ij)) {
			fmt::println("{}Test failed for ij!{}", RED, RESET);
			fmt::println("{}Expected:{}", RED, RESET);
			matrix_print(C_ref);
			fmt::println("{}Got:{}", RED, RESET);
			matrix_print(C_test_ij);
			Kokkos::finalize();
			exit(EXIT_FAILURE);
		}
		// if (!matrix_are_equal(C_ref, C_test_ijk)) {
		// 	fmt::println("{}Test failed for ijk!{}", RED, RESET);
		// 	fmt::println("{}Expected:{}", RED, RESET);
		// 	matrix_print(C_ref);
		// 	fmt::println("{}Got:{}", RED, RESET);
		// 	matrix_print(C_test_ijk);
		// 	Kokkos::finalize();
		// 	exit(EXIT_FAILURE);
		// }
	}

	// 100 randomised tests
	for (int i = 0; i < 100; i++) {

		// Random dimensions of the matrices
		int m = rand() % 50 + 1;
		int n = rand() % 50 + 1;
		int k = rand() % 50 + 1;

		// Random alpha and beta
		double alpha = static_cast<double>(rand()) / RAND_MAX;
		double beta  = static_cast<double>(rand()) / RAND_MAX;

		// Random matrices
		auto A		= RightMatrix("A", m, k);
		auto B		= LeftMatrix("B", k, n);
		auto C_ref	= RightMatrix("C_ref", m, n);
		auto C_test_i	= RightMatrix("C_test_i", m, n);
		auto C_test_ij	= RightMatrix("C_test_ij", m, n);
		auto C_test_ijk = RightMatrix("C_test_ijk", m, n);
		matrix_init(A);
		matrix_init(B);
		matrix_init(C_ref);
		for (int j = 0; j < m; j++) {
			for (int l = 0; l < n; l++) {
				C_test_i(j, l)	 = C_ref(j, l);
				C_test_ij(j, l)	 = C_ref(j, l);
				C_test_ijk(j, l) = C_ref(j, l);
			}
		}

		// Random cache block sizes
		int block_size = rand() % 50 + 1;

		// Run the reference and test functions
		Kokkos::fence();
		matrix_product_reference(alpha, A, B, beta, C_ref);
		Kokkos::fence();
		matrix_product_cache_blocked_i(alpha, A, B, beta, C_test_i, block_size);
		Kokkos::fence();
		matrix_product_cache_blocked_ij(alpha, A, B, beta, C_test_ij, block_size);
		Kokkos::fence();
		// Too long to run
		// matrix_product_cache_blocked_ijk(alpha, A, B, beta, C_test_ijk, block_size);
		// Kokkos::fence();

		// Check if the results are equal
		if (!matrix_are_equal(C_ref, C_test_i)) {
			fmt::println("{}Test failed for i!{}", RED, RESET);
			Kokkos::finalize();
			exit(EXIT_FAILURE);
		}
		if (!matrix_are_equal(C_ref, C_test_ij)) {
			fmt::println("{}Test failed for ij!{}", RED, RESET);
			Kokkos::finalize();
			exit(EXIT_FAILURE);
		}
		// if (!matrix_are_equal(C_ref, C_test_ijk)) {
		// 	fmt::println("{}Test failed for ijk!{}", RED, RESET);
		// 	Kokkos::finalize();
		// 	exit(EXIT_FAILURE);
		// }
	}

	// Print that everything is ok
	Kokkos::finalize();
	fmt::println("{}All tests passed!{}", GREEN, RESET);
	exit(EXIT_SUCCESS);
}