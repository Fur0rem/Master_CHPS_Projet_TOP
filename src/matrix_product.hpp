/**
 * @file src/matrix_product.hpp
 * @brief Header file for matrix product functions.
 */

#ifndef TOP_MATRIX_PRODUCT_HPP
#define TOP_MATRIX_PRODUCT_HPP

#include <cassert>
#include <cstdlib>

#include <Kokkos_Core.hpp>
#include <fmt/core.h>

using RightMatrix = Kokkos::View<double**, Kokkos::LayoutRight>;
using LeftMatrix  = Kokkos::View<double**, Kokkos::LayoutLeft>;

template <class MatrixType> auto matrix_init(MatrixType& M) -> void {
	static_assert(2 == MatrixType::rank(), "View must be of rank 2");

	Kokkos::parallel_for(
	    "init", M.extent(0), KOKKOS_LAMBDA(int i) {
		    for (int j = 0; j < int(M.extent(1)); ++j) {
			    M(i, j) = drand48();
		    }
	    });
}

template <class AMatrixType, class BMatrixType, class CMatrixType>
auto matrix_product_reference(double alpha, AMatrixType const& A, BMatrixType const& B, double beta, CMatrixType& C) -> void {
	static_assert(AMatrixType::rank() == 2 && BMatrixType::rank() == 2 && CMatrixType::rank() == 2, "Views must be of rank 2");
	assert(A.extent(0) == C.extent(0));
	assert(B.extent(1) == C.extent(1));
	assert(A.extent(1) == B.extent(0));

	Kokkos::parallel_for(
	    "dgemm_kernel", A.extent(0), KOKKOS_LAMBDA(int i) {
		    for (int j = 0; j < int(B.extent(1)); ++j) {
			    double acc = 0.0;
			    for (int k = 0; k < int(A.extent(1)); ++k) {
				    acc += alpha * A(i, k) * B(k, j);
			    }
			    C(i, j) *= beta + acc;
		    }
	    });
}

auto matrix_product_cache_blocked_i(double alpha, RightMatrix const& A, LeftMatrix const& B, double beta, RightMatrix& C, int block_size)
    -> void {
	static_assert(RightMatrix::rank() == 2 && LeftMatrix::rank() == 2, "Views must be of rank 2");
	assert(A.extent(0) == C.extent(0));
	assert(B.extent(1) == C.extent(1));
	assert(A.extent(1) == B.extent(0));

	Kokkos::parallel_for(
	    "dgemm_kernel", (A.extent(0) + block_size) / block_size, KOKKOS_LAMBDA(int _bi) {
		    int bi = _bi * block_size;
		    for (int j = 0; j < int(B.extent(1)); j += 1) {
			    for (int i = bi; i < std::min(bi + block_size, int(A.extent(0))); i++) {
				    double acc = 0.0;
				    for (int k = 0; k < int(A.extent(1)); k++) {
					    acc += A(i, k) * B(k, j);
				    }
				    C(i, j) *= beta + (alpha * acc);
			    }
		    }
	    });
}

auto matrix_product_cache_blocked_ij(double alpha, RightMatrix const& A, LeftMatrix const& B, double beta, RightMatrix& C, int block_size)
    -> void {
	static_assert(RightMatrix::rank() == 2 && LeftMatrix::rank() == 2, "Views must be of rank 2");
	assert(A.extent(0) == C.extent(0));
	assert(B.extent(1) == C.extent(1));
	assert(A.extent(1) == B.extent(0));

	Kokkos::parallel_for(
	    "dgemm_kernel", (A.extent(0) + block_size) / block_size, KOKKOS_LAMBDA(int _bi) {
		    int bi = _bi * block_size;
		    for (int bj = 0; bj < int(B.extent(1)); bj += block_size) {
			    for (int i = bi; i < std::min(bi + block_size, int(A.extent(0))); i++) {
				    for (int j = bj; j < std::min(bj + block_size, int(B.extent(1))); j++) {
					    double acc = 0.0;
					    for (int k = 0; k < int(A.extent(1)); k++) {
						    acc += A(i, k) * B(k, j);
					    }
					    C(i, j) *= beta + (alpha * acc);
				    }
			    }
		    }
	    });
}

auto matrix_product_cache_blocked_ijk(double alpha, RightMatrix const& A, LeftMatrix const& B, double beta, RightMatrix& C, int block_size)
    -> void {
	static_assert(RightMatrix::rank() == 2 && LeftMatrix::rank() == 2, "Views must be of rank 2");
	assert(A.extent(0) == C.extent(0));
	assert(B.extent(1) == C.extent(1));
	assert(A.extent(1) == B.extent(0));

	Kokkos::parallel_for(
	    "dgemm_kernel", (A.extent(0) + block_size) / block_size, KOKKOS_LAMBDA(int _bi) {
		    int bi	     = _bi * block_size;
		    RightMatrix accs = RightMatrix("accs", block_size, block_size);
		    for (int bj = 0; bj < int(B.extent(1)); bj += block_size) {
			    for (int bk = 0; bk < int(A.extent(1)); bk += block_size) {
				    for (int i = bi; i < std::min(bi + block_size, int(A.extent(0))); i++) {
					    for (int j = bj; j < std::min(bj + block_size, int(B.extent(1))); j++) {
						    for (int k = bk; k < std::min(bk + block_size, int(A.extent(1))); k++) {
							    double acc = 0.0;
							    for (int k = 0; k < int(A.extent(1)); k++) {
								    acc += A(i, k) * B(k, j);
							    }
							    accs(i - bi, j - bj) = acc;
						    }
					    }
				    }
			    }
			    for (int i = bi; i < std::min(bi + block_size, int(A.extent(0))); i++) {
				    for (int j = bj; j < std::min(bj + block_size, int(B.extent(1))); j++) {
					    C(i, j) *= beta + (alpha * accs(i - bi, j - bj));
				    }
			    }
		    }
	    });
}

template <class AMatrixType, class BMatrixType> auto matrix_are_equal(AMatrixType& A, BMatrixType& B) -> bool {
	static_assert(AMatrixType::rank() == 2 && BMatrixType::rank() == 2, "Views must be of rank 2");
	if (A.extent(0) != B.extent(0) || A.extent(1) != B.extent(1)) {
		fmt::print("Matrix dimensions do not match: ({}, {}) vs ({}, {})\n", A.extent(0), A.extent(1), B.extent(0), B.extent(1));
		return false;
	}

	constexpr double EPS = 1e-10;
	for (int i = 0; i < int(A.extent(0)); i++) {
		for (int j = 0; j < int(A.extent(1)); j++) {
			if (std::abs(A(i, j) - B(i, j)) > EPS) {
				fmt::print("Mismatch at ({}, {}): {} != {}\n", i, j, A(i, j), B(i, j));
				return false;
			}
		}
	}
	return true;
}

template <class MatrixType> auto matrix_print(MatrixType& A) -> void {
	static_assert(MatrixType::rank() == 2, "View must be of rank 2");
	for (int i = 0; i < int(A.extent(0)); i++) {
		for (int j = 0; j < int(A.extent(1)); j++) {
			fmt::print("{:.3f} ", A(i, j));
		}
		fmt::print("\n");
	}
}

#endif