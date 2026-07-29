#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 已经理解kernel7的理论  就是当计算任务多时，假设计算C的64列，TM=8，那么一个thread中就连续计算8次且这8个数据是连续存储在bank上的，如果bank32，
// 那么8个thread中thread0要计算的8个数据在bank0-7 thread1要计算的8个数据在8-15.......但是在同一时刻它们只会读bank0，8，...，此时thread4-thread7也要读bank0,8,...，
// 但是其它的bamk比如1-7是可读的，这就导致了bankconflict


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmResolveBankConflicts(int M, int N, int K, float alpha,
                                          float *A, float *B, float beta,
                                          float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    // "linearize" Bs while storing it     这里的[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2]直接用最下面的
    // // 新 row = k * 8 + (4g + 0) % 8 和 // 新 col = (4g + 0) / 8 回代就得到这个式子
    tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2] = tmp.x;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 1) * 16 + innerColB / 2] = tmp.y;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 2) * 16 + innerColB / 2] = tmp.z;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 3) * 16 + innerColB / 2] = tmp.w;
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[(dotIdx * 8 + i) * 16 + threadCol];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}


// 目标：
// 第 0 拍会读的放一行：
// B0 B8 B16 B24 B32 ...

// 第 1 拍会读的放一行：
// B1 B9 B17 B25 B33 ...

// 第 2 拍会读的放一行：
// B2 B10 B18 B26 B34 ...
// 现在看 kernel7 的索引为什么刚好做到这件事。

// 这一行代码正在处理一个 float4：
// tmp = B[innerRowB][innerColB * 4 ... innerColB * 4 + 3]
// 也就是：
// innerRowB = k
// innerColB = g

// tmp.x = B[k][4g + 0]
// tmp.y = B[k][4g + 1]
// tmp.z = B[k][4g + 2]
// tmp.w = B[k][4g + 3]
// 我们现在只看 tmp.x。
// 它原来是：
// B[k][4g + 0]
// 它应该被放到哪里？
// 因为我们想按“第几拍”排，所以：
// B[k][0]  应该放到第 0 拍那行
// B[k][1]  应该放到第 1 拍那行
// ...
// B[k][7]  应该放到第 7 拍那行
// B[k][8]  又回到第 0 拍那行
// B[k][9]  又回到第 1 拍那行
// 所以对任意 B[k][n]：
// 新 row = k * 8 + n % 8
// 新 col = n / 8
// 这里 8 来自 TN=8。
// 现在把 tmp.x 的 n = 4g + 0 代进去：
// 新 row = k * 8 + (4g + 0) % 8
// 新 col = (4g + 0) / 8
//
