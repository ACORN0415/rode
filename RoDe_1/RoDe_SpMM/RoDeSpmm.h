#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Work Queue (device resident)
struct RoDeWorkQueue {
    unsigned total_tiles;   // total tiles to process
    unsigned* counter;      // device-side counter
};

#ifdef __CUDACC__
// extern __device__ RoDeWorkQueue __rode_wq;
#endif

// Host-side API
void RoDeWQInitOnce();
void RoDeWQResetBeforeLaunch(unsigned total_tiles_for_this_launch);

// Debug counters
void RoDeDebugResetCounters();
void RoDeDebugDumpCounters();

// SpMM entry points
void RoDeSpmm_n32(int m1,int m2,int k,int n,
                  const float* values,
                  const int* column_indices,
                  const int* row_offsets,
                  const int* row_indices1,
                  const int* row_indices2,
                  const int* row_seg_st_offsets,
                  const float* B, float* C,
                  cudaStream_t stream1, cudaStream_t stream2);

void RoDeSpmm_n128(int m1,int m2,int k,int n,
                   const float* values,
                   const int* column_indices,
                   const int* row_offsets,
                   const int* row_indices1,
                   const int* row_indices2,
                   const int* row_seg_st_offsets,
                   const float* B, float* C,
                   cudaStream_t stream1, cudaStream_t stream2);

void RoDeSpmm_n32(int m1,int m2,int k,int n,
                  const double* values,
                  const int* column_indices,
                  const int* row_offsets,
                  const int* row_indices1,
                  const int* row_indices2,
                  const int* row_seg_st_offsets,
                  const double* B, double* C,
                  cudaStream_t stream1, cudaStream_t stream2);

void RoDeSpmm_n128(int m1,int m2,int k,int n,
                   const double* values,
                   const int* column_indices,
                   const int* row_offsets,
                   const int* row_indices1,
                   const int* row_indices2,
                   const int* row_seg_st_offsets,
                   const double* B, double* C,
                   cudaStream_t stream1, cudaStream_t stream2);
