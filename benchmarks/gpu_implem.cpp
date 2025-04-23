/**
 * @file benchmarks/gpu_implem.cpp
 * @brief Benchmarking the GPU implementation of the matrix product compared to the CPU implementation.
 */

#include "matrix_product.hpp"

#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <fmt/core.h>
#include <nanobench.h>

#include "culkan.h"

#include <iostream>

auto main(int argc, char* argv[]) -> int {
	Kokkos::initialize(argc, argv);

	// Known seed for deterministic RNG
	srand48(42);

	// Dimensions of the matrices
	constexpr int matrix_sizes[] = {
	    250,
	    500,
	    1000,
	    2000,
	};

	for (const auto& size : matrix_sizes) {
		int m = size;
		int n = size;
		int k = size;

		RightMatrix A = RightMatrix("A", m, k);
		LeftMatrix B  = LeftMatrix("B", k, n);
		RightMatrix C = RightMatrix("C", m, n);
		matrix_init(A);
		matrix_init(B);
		matrix_init(C);

		// Generate alpha and beta
		double alpha = drand48();
		double beta  = drand48();

		// Bindings of the shader
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

		// Compare all the different layout combinations
		std::ostringstream oss;
		auto result = ankerl::nanobench::Bench()
				  .minEpochIterations(3)
				  .performanceCounters(true)
				  .output(&oss)
				  .run(fmt::format("CPU {}", size), [&]() { matrix_product_cache_blocked_i(alpha, A, B, beta, C, 8); })
				  .run("GPU with memory overhead",
				       [&]() {
					       // Send the data to the GPU
					       culkanWriteBinding(culkan, 0, &n);
					       culkanWriteBinding(culkan, 1, &m);
					       culkanWriteBinding(culkan, 2, &k);
					       culkanWriteBinding(culkan, 3, A.data());
					       culkanWriteBinding(culkan, 4, B.data());
					       culkanWriteBinding(culkan, 5, C.data());
					       culkanWriteBinding(culkan, 6, &alpha);
					       culkanWriteBinding(culkan, 7, &beta);

					       // Do the GPU computation
					       culkanSetup(culkan);
					       culkanRun(culkan);

					       // Read the result from the GPU
					       double* result = (double*)malloc(m * n * sizeof(double));
					       culkanReadBinding(culkan, 5, result);
					       free(result);
				       })
				  .doNotOptimizeAway(A)
				  .doNotOptimizeAway(C)
				  .doNotOptimizeAway(B)
				  .doNotOptimizeAway(alpha)
				  .doNotOptimizeAway(beta)
				  .results();

		// Print oss
		// std::cout << oss.str() << '\n';

		for (auto const& res : result) {
			auto measure = res.fromString("elapsed");
			auto name    = res.config().mBenchmarkName;
			fmt::println(
			    "{}, Min: {}s, Max: {}s, Med: {}s", name, res.minimum(measure), res.maximum(measure), res.median(measure));
		}

		// Do it without memory overhead
		std::ostringstream oss2;
		culkanWriteBinding(culkan, 0, &n);
		culkanWriteBinding(culkan, 1, &m);
		culkanWriteBinding(culkan, 2, &k);
		culkanWriteBinding(culkan, 3, A.data());
		culkanWriteBinding(culkan, 4, B.data());
		culkanWriteBinding(culkan, 5, C.data());
		culkanWriteBinding(culkan, 6, &alpha);
		culkanWriteBinding(culkan, 7, &beta);

		culkanSetup(culkan);

		auto result2 = ankerl::nanobench::Bench()
				   .minEpochIterations(3)
				   .performanceCounters(true)
				   .output(&oss2)
				   .run("GPU without memory overhead", [&]() { culkanRun(culkan); })
				   .results();

		for (auto const& res : result2) {
			auto measure = res.fromString("elapsed");
			auto name    = res.config().mBenchmarkName;
			fmt::println(
			    "{}, Min: {}s, Max: {}s, Med: {}s", name, res.minimum(measure), res.maximum(measure), res.median(measure));
		}
	}

	Kokkos::finalize();
	exit(EXIT_SUCCESS);
}