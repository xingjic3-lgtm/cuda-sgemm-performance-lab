#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <dispatch.cuh>
#include <fstream>
#include <iostream>
#include <random>
#include <runner.cuh>
#include <string>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv) {
  const bool randomTestMode =
      argc == 4 && std::string(argv[2]) == "--random-tests";
  if (argc != 2 && argc != 5 && !randomTestMode) {
    std::cerr << "Usage:\n"
              << "  sgemm <kernel 0-13>\n"
              << "  sgemm <kernel 0-13> <M> <N> <K>\n"
              << "  sgemm <kernel 0-13> --random-tests <count>"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 13) {
    std::cerr << "Please enter a valid kernel number (0-13)" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (randomTestMode && kernel_num == 0) {
    std::cerr << "Random correctness tests require a custom kernel (1-13)."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  std::vector<std::array<int, 3>> shapes;
  if (randomTestMode) {
    const int testCount = std::stoi(argv[3]);
    if (testCount <= 0) {
      std::cerr << "Random test count must be positive." << std::endl;
      exit(EXIT_FAILURE);
    }

    // A fixed seed makes a failed case exactly reproducible.
    std::mt19937 generator(5060);
    std::uniform_int_distribution<int> dimension(1, 511);
    shapes.reserve(testCount);
    for (int test = 0; test < testCount; ++test) {
      shapes.push_back(
          {dimension(generator), dimension(generator), dimension(generator)});
    }
  } else if (argc == 5) {
    const int customM = std::stoi(argv[2]);
    const int customN = std::stoi(argv[3]);
    const int customK = std::stoi(argv[4]);
    if (customM <= 0 || customN <= 0 || customK <= 0) {
      std::cerr << "M, N and K must all be positive." << std::endl;
      exit(EXIT_FAILURE);
    }
    shapes.push_back({customM, customN, customK});
  } else {
    for (int size : {128, 256, 512, 1024, 2048, 4096}) {
      shapes.push_back({size, size, size});
    }
  }

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

  // print some device info
  // CudaDeviceInfo();

  // Declare the handle, create the handle, cublasCreate will return a value of
  // type cublasStatus_t to determine whether the handle was created
  // successfully (the value is 0)
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  int m, n, k;
  size_t maxAElements = 0;
  size_t maxBElements = 0;
  size_t maxCElements = 0;
  for (const auto &shape : shapes) {
    maxAElements = std::max(maxAElements,
                            static_cast<size_t>(shape[0]) * shape[2]);
    maxBElements = std::max(maxBElements,
                            static_cast<size_t>(shape[2]) * shape[1]);
    maxCElements = std::max(maxCElements,
                            static_cast<size_t>(shape[0]) * shape[1]);
  }

  float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

  float *A = nullptr, *B = nullptr, *C_initial = nullptr, *C = nullptr,
        *C_ref = nullptr; // host matrices
  float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices

  A = (float *)malloc(sizeof(float) * maxAElements);
  B = (float *)malloc(sizeof(float) * maxBElements);
  C_initial = (float *)malloc(sizeof(float) * maxCElements);
  C = (float *)malloc(sizeof(float) * maxCElements);
  C_ref = (float *)malloc(sizeof(float) * maxCElements);

  randomize_matrix(A, maxAElements);
  randomize_matrix(B, maxBElements);
  randomize_matrix(C_initial, maxCElements);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * maxAElements));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * maxBElements));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * maxCElements));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * maxCElements));

  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * maxAElements,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * maxBElements,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C_initial, sizeof(float) * maxCElements,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C_initial, sizeof(float) * maxCElements,
                       cudaMemcpyHostToDevice));

  int repeat_times = 50;
  int testIndex = 0;
  for (const auto &shape : shapes) {
    ++testIndex;
    m = shape[0];
    n = shape[1];
    k = shape[2];

    if (randomTestMode) {
      std::cout << "[" << testIndex << "/" << shapes.size() << "] ";
    }
    std::cout << "dimensions M=" << m << ", N=" << n << ", K=" << k
              << ", alpha: " << alpha << ", beta: " << beta << std::endl;
    if (kernel_num == 10) {
#ifdef SGEMM_K10_AUTOTUNE_MODE
      std::cout << "Kernel 10 dispatch selected: compile-time-autotune-candidate"
                << std::endl;
#else
      const K10ConfigId selectedConfig = selectK10Config(m, n, k);
      std::cout << "Kernel 10 dispatch selected: "
                << k10ConfigName(selectedConfig) << std::endl;
#endif
    }
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0) {
      if (randomTestMode) {
        // Every random case starts from the same C values for the custom
        // kernel and cuBLAS reference, independent of the preceding case.
        cudaCheck(cudaMemcpy(dC, C_initial, sizeof(float) * m * n,
                             cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dC_ref, C_initial, sizeof(float) * m * n,
                             cudaMemcpyHostToDevice));
      }
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
                 handle); // cuBLAS
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                 handle); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

      if (!verify_matrix(C_ref, C, m * n)) {
        std::cout
            << "Failed to pass the correctness verification against NVIDIA "
               "cuBLAS."
            << std::endl;
        if (m <= 128) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix(A, m, k, fs);
          fs << "B:\n";
          print_matrix(B, k, n, fs);
          fs << "C:\n";
          print_matrix(C, m, n, fs);
          fs << "Should:\n";
          print_matrix(C_ref, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }

      if (randomTestMode) {
        std::cout << "PASS" << std::endl;
      }
    } else {
      // Kernel 0 is cuBLAS itself. Warm it up once so the first timed shape
      // does not include lazy CUDA/cuBLAS initialization.
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC, handle);
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError());
    }

    if (randomTestMode) {
      continue;
    }

    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      // We don't reset dC between runs to save time
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    double flops = 2.0 * static_cast<double>(m) * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%d x %d x %d).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m, n, k);
    fflush(stdout);
    // make dC and dC_ref equal again (we modified dC while calling our kernel
    // for benchmarking)
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                         cudaMemcpyDeviceToDevice));
  }

  if (randomTestMode) {
    std::cout << "All " << shapes.size()
              << " random correctness tests passed." << std::endl;
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C_initial);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cublasDestroy(handle);

  return 0;
};
