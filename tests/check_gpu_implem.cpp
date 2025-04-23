#include <stdio.h>
#include <stdlib.h>

#include "matrix_product.hpp"
#include <Kokkos_Core.hpp>

#include "culkan.h"

auto main(int argc, char* argv[]) -> int {
	Kokkos::initialize(argc, argv);

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

	// Testing alpha and beta
	double alpha = 2.0;
	double beta  = -1.0;

	CulkanBinding bindings[] = {
	    // Binding for n
	    {.size = sizeof(int), .type = UNIFORM_BUFFER},
	    // Binding for m
	    {.size = sizeof(int), .type = UNIFORM_BUFFER},
	    // Binding for k
	    {.size = sizeof(int), .type = UNIFORM_BUFFER},
	    // Binding for A
	    {.size = m * k * sizeof(double), .type = STORAGE_BUFFER},
	    // Binding for B
	    {.size = k * n * sizeof(double), .type = STORAGE_BUFFER},
	    // Binding for C
	    {.size = m * n * sizeof(double), .type = STORAGE_BUFFER},
	    // Binding for alpha
	    {.size = sizeof(double), .type = UNIFORM_BUFFER},
	    // Binding for beta
	    {.size = sizeof(double), .type = UNIFORM_BUFFER},
	};
	CulkanLayout layout = {.bindingCount = 8, .bindings = bindings};

	// Compile the shader
	int ret = system("glslc ./src/operation.comp -o ./build/operation.spv");
	if (ret != 0) {
		fprintf(stderr, "Failed to compile shader\n");
		return 1;
	}

	Culkan* culkan = culkanInit(&layout, "./build/operation.spv", (CulkanInvocations){1024, 1, 1});
	culkanWriteBinding(culkan, 0, &n);
	culkanWriteBinding(culkan, 1, &m);
	culkanWriteBinding(culkan, 2, &k);
	culkanWriteBinding(culkan, 3, A.data());
	culkanWriteBinding(culkan, 4, B.data());
	culkanWriteBinding(culkan, 5, C_ref.data());
	culkanWriteBinding(culkan, 6, &alpha);
	culkanWriteBinding(culkan, 7, &beta);

	culkanSetup(culkan);

	// Do the GPU computation
	culkanRun(culkan);
	double* result = (double*)malloc(m * n * sizeof(double));
	culkanReadBinding(culkan, 5, result);

	// Do the CPU computation
	matrix_product_reference(alpha, A, B, beta, C_ref);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (result[i * n + j] != C_ref(i, j)) {
				fmt::print("Mismatch at ({}, {}): {} != {}\n", i, j, result[i * n + j], C_ref(i, j));
				free(result);
				culkanDestroy(culkan);
				return 1;
			}
		}
	}

	fmt::print("GPU result matches reference result!\n");
	free(result);
	culkanDestroy(culkan);

	Kokkos::finalize();
	exit(EXIT_SUCCESS);
}