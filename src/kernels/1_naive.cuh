#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

// 发射是dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32)); dim3 blockDim(32, 32);
// 所以这里的x对应M方向，y对应N方向   计算资源编号按照x递增，比如看一个block内先是x增再是y增，所以这里的const uint x先增
// 一个warp内计算的任务是竖直方向的32行1列(原因是因为上面算的x).比如说C01,C11,C21,C31,C41...
// 所有thread的计算节拍是按照i来递增的,现在第0拍(i=0),取A的32个行的第0列和B的第y列的第一行,此时A访存是不连续的,B是广播的
// 优化思路:kernel2:让一个warp内的thread横向计算C的一行中的32列,此时A访存广播,B访存是连续的

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
