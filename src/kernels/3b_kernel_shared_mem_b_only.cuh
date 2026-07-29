#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>


// warp间只复用了B,于是我们只对Bsharedmemory
// 只sharedmemoryB但是计算是需要A的,同步又要等A.B确实优化了，但短板在 A。B 再快，也要等 A。
template <const int BLOCKSIZE>
template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_b_only(int M, int N, int K, float alpha,
                                        const float *A, const float *B,
                                        float beta, float *C) {
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
    __syncthreads();

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += A[threadRow * K + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;
  }

  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}
