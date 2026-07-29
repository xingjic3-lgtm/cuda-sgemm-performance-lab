#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _WIN32
#include <winsock.h>
#include <chrono>

inline int gettimeofday(timeval *tv, void *) {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto usec =
      std::chrono::duration_cast<std::chrono::microseconds>(now);
  tv->tv_sec = static_cast<long>(usec.count() / 1000000);
  tv->tv_usec = static_cast<long>(usec.count() % 1000000);
  return 0;
}
#else
#include <sys/time.h>
#include <unistd.h>
#endif

void cudaCheck(cudaError_t error, const char *file,
               int line); // CUDA error check
void CudaDeviceInfo();    // print CUDA information

void range_init_matrix(float *mat, int N);
void randomize_matrix(float *mat, int N);
void zero_init_matrix(float *mat, int N);
void copy_matrix(const float *src, float *dest, int N);
void print_matrix(const float *A, int M, int N, std::ofstream &fs);
bool verify_matrix(float *mat1, float *mat2, int N);

float get_current_sec();                        // Get the current moment
float cpu_elapsed_time(float &beg, float &end); // Calculate time difference

void run_kernel(int kernel_num, int m, int n, int k, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle);

void runSgemmAuto(int M, int N, int K, float alpha, float *A, float *B,
                  float beta, float *C);
