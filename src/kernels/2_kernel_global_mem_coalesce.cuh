#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>


// 让一个warp内的thread横向计算C的一行中的32列,此时A访存广播,B访存是连续的
// kernel3优化思路
// 现在如果32个warp每个warp分别计算 warp1:C00,C01,C02...C31   warp2:C10,C11,C12...C31   warp3:C20,C21,C22...C31   warp4:C30,C31,C32...C31
// 可以发现这些warp分别都是用不同的A的行但是用的是同样的0-31列B,于是我们想能不能复用B,读一次B完成多行,kernel3使用sharedmemory把A的32行B的31列搬入
// 计算一个小矩阵C0,0-C31,31 此时读一次B计算多行A,复用了B
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // if statement is necessary to make things work under tile quantization
  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}