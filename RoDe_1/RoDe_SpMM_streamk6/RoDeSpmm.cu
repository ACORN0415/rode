// Stream_RoDe_SpMM.cu

#include "RoDeSpmm.h"
#include "basic_utils.h"
#include "cuda_runtime.h"
#include "common_utils.h"
#include <cooperative_groups.h>
using namespace SPC;

struct __align__(8) RoDeWorkQueue { unsigned total_tiles; unsigned* counter; };

__device__ RoDeWorkQueue __rode_wq;
__device__ int   __rode_wave_threshold     = 3;
__device__ int   __rode_force_streamk      = 0;
__device__ int   __rode_avail_sms          = 0;
__device__ float __rode_dp_eff_threshold   = 0.92f;
__device__ int   __rode_min_iters_per_sk   = 2;
__device__ int   __rode_est_iters_per_tile = 4;

template <typename DenseValue, typename ScalarValue>
__device__ __forceinline__
const DenseValue* DenseVecPtr(const ScalarValue* B, int n, int n_idx, int col, int lane) {
  constexpr int V = int(sizeof(DenseValue) / sizeof(ScalarValue));
  int elem_off = col * n + n_idx;
  return reinterpret_cast<const DenseValue*>(B) + (elem_off / V) + lane;
}

template <typename ScalarValue,typename SparseValue,typename DenseValue,
          int kBlockItemsY,int kBlockItemsK,int kBlockItemsX,int kBlockWidth,
          int kResidueUnroll,int STAGE = 8>
struct SparseKernel {
  typedef int ScalarIndex;
  static constexpr int kSparseValuesPerLoad = int(sizeof(SparseValue) / sizeof(ScalarValue));
  static constexpr int kThreadItemsX = kBlockItemsX / kBlockWidth;
  static constexpr int kThreadItemsK = kBlockItemsK / kBlockWidth / kSparseValuesPerLoad;
  static constexpr int kDenseValuesPerLoad = int(sizeof(DenseValue) / sizeof(ScalarValue));
  static constexpr int kDenseThreadItemsX = kBlockItemsX / kDenseValuesPerLoad / kBlockWidth;
  static constexpr int kElementsPerScalar = 1;
  typedef typename Value2Index<SparseValue>::Index SparseIndex;
  typedef SPC::Barrier<kBlockItemsY,kBlockWidth> Barrier;
  typedef typename SPC::TypeUtils<DenseValue>::Accumulator Accumulator;
  static constexpr int kResidueOuterLimit = kBlockItemsK / kResidueUnroll;
  static constexpr int kResidueInnerLimit = kResidueUnroll;
  static constexpr int kDenseFragmentSize = kElementsPerScalar * kBlockItemsK * kBlockItemsX / kBlockWidth;
  static constexpr int kOutputFragmentSize = kBlockItemsX * kElementsPerScalar / kBlockWidth;
  static constexpr int kTileSize = kBlockItemsY * kBlockItemsK;

  static __device__ __forceinline__
  void Kernel4Residue(int m,int k,int n,
                      const ScalarValue* __restrict__ values,
                      const int * __restrict__ column_indices,
                      const int * __restrict__ row_offsets,
                      const int* __restrict__ row_indices,
                      const ScalarValue * B,ScalarValue *C) {
#ifdef THREADBLOCK_SWIZZLE
    int m_idx = blockIdx.y * kBlockItemsY + threadIdx.y;
    int n_idx = blockIdx.x * kBlockItemsX;
#else
    int m_idx = blockIdx.x * kBlockItemsY + threadIdx.y;
    int n_idx = blockIdx.y * kBlockItemsX;
#endif
    if (m_idx >= m) return;

    m_idx = Load(row_indices + m_idx);
    int row_offset = Load(row_offsets + m_idx);
    int nonzeros = Load(row_offsets + m_idx + 1) - row_offset;

    __shared__  ScalarValue values_tile_array[kTileSize];
    __shared__  ScalarIndex column_indices_tile_array[kTileSize];

    ScalarValue * values_tile = values_tile_array + kBlockItemsK * threadIdx.y;
    ScalarIndex * column_indices_tile = column_indices_tile_array + kBlockItemsK * threadIdx.y;

    static constexpr int kValueAligment = int(sizeof(SparseValue) / sizeof(ScalarValue));
    static constexpr uint32_t kAlignmentMask = ~(kValueAligment - 1);

    int values_to_mask_ = row_offset & (kValueAligment - 1);
    int aligned_nonzeros = nonzeros + values_to_mask_;
    bool atomicFlag = false;
    if (aligned_nonzeros >= kBlockItemsK) {
      nonzeros   = aligned_nonzeros % kBlockItemsK;
      row_offset = (row_offset & kAlignmentMask) + aligned_nonzeros / kBlockItemsK * kBlockItemsK;
      atomicFlag = true;
    }

    Barrier barrier(threadIdx.y);
    __align__(16) ScalarValue dense_matrix_fragment[kDenseFragmentSize] = {};
    __align__(16) ScalarValue output_fragment[kOutputFragmentSize] = {};

    SparseValue* sparse_values_tile = reinterpret_cast<SparseValue*>(values_tile) + threadIdx.x;
    SparseIndex* sparse_columns_tile = reinterpret_cast<SparseIndex*>(column_indices_tile) + threadIdx.x;

    if (kResidueUnroll > 1) {
      const ScalarValue kZeroValues[kSparseValuesPerLoad] = {};
      const ScalarIndex kZeroIndices[kSparseValuesPerLoad] = {};
      SparseValue* svt = sparse_values_tile;
      SparseIndex* sct = sparse_columns_tile;
      #pragma unroll
      for (int i = 0; i < (kBlockItemsK / kBlockWidth / kSparseValuesPerLoad); ++i) {
        Store(*reinterpret_cast<const SparseIndex*>(kZeroIndices), sct);
        Store(*reinterpret_cast<const SparseValue*>(kZeroValues), svt);
        svt += kBlockWidth; sct += kBlockWidth;
      }
      barrier.Sync();
    }

    constexpr int kResidueUpdateStrideValue = -int(sizeof(ScalarValue)) * (kSparseValuesPerLoad - 1);
    const int kResidueUpdateValue = int(threadIdx.x) * kResidueUpdateStrideValue;
    constexpr int kResidueUpdateStrideIndex = -int(sizeof(ScalarIndex)) * (kSparseValuesPerLoad - 1);
    const int kResidueUpdateIndex = int(threadIdx.x) * kResidueUpdateStrideIndex;

    const ScalarValue *sparse_values = reinterpret_cast<const ScalarValue*>(values + row_offset + threadIdx.x);
    const ScalarIndex *sparse_columns = reinterpret_cast<const ScalarIndex*>(column_indices + row_offset + threadIdx.x);

    ScalarIndex* sparse_columns_tile__ = OffsetCast<ScalarIndex>(sparse_columns_tile, kResidueUpdateIndex);
    ScalarValue* sparse_values_tile__  = OffsetCast<ScalarValue>(sparse_values_tile,  kResidueUpdateValue);

    constexpr int kScalarThreadItemsK = kBlockItemsK / kBlockWidth;
    int nonzeros_ = nonzeros;
    #pragma unroll
    for (int i = 0; i < kScalarThreadItemsK; ++i) {
      if (nonzeros_ <= int(threadIdx.x)) break;
      Store(Load(sparse_values),  sparse_values_tile__);
      Store(Load(sparse_columns), sparse_columns_tile__);
      sparse_values         += kBlockWidth;
      sparse_columns        += kBlockWidth;
      sparse_values_tile__  += kBlockWidth;
      sparse_columns_tile__ += kBlockWidth;
      nonzeros_             -= kBlockWidth;
    }
    asm(""); barrier.Sync();

    #pragma unroll
    for (int i = 0; i < kResidueOuterLimit; ++i) {
      if (nonzeros <= 0) break;
      #pragma unroll
      for (int j = 0; j < kResidueInnerLimit; ++j) {
        const int k_item_idx = i * kResidueInnerLimit + j;
        int col = column_indices_tile[k_item_idx];
        ScalarValue lhs = values_tile[k_item_idx];
        const DenseValue* matrix__ = DenseVecPtr<DenseValue,ScalarValue>(B, n, n_idx, col, threadIdx.x);
        #pragma unroll
        for (int l = 0; l < kDenseThreadItemsX; ++l) {
          ScalarValue *outputs = output_fragment + l * kDenseValuesPerLoad;
          SPC::VectorCompute<DenseValue>::FMA(lhs, Load(matrix__), reinterpret_cast<Accumulator*>(outputs));
          matrix__ += kBlockWidth;
        }
      }
      nonzeros -= kResidueInnerLimit;
    }
    asm("");

    const int output_offset = m_idx * n + n_idx;
    if (atomicFlag) {
      ScalarValue* out = C + output_offset + threadIdx.x * kDenseValuesPerLoad;
      #pragma unroll
      for (int i = 0; i < kDenseThreadItemsX; ++i) {
        #pragma unroll
        for (int j = 0; j < kDenseValuesPerLoad; ++j)
          atomicAdd(out + j, output_fragment[i * kDenseValuesPerLoad + j]);
        out += kBlockWidth * kDenseValuesPerLoad;
      }
    } else {
      DenseValue* out = reinterpret_cast<DenseValue*>(C + output_offset) + threadIdx.x;
      #pragma unroll
      for (int i = 0; i < kDenseThreadItemsX; ++i) {
        const DenseValue* of = reinterpret_cast<const DenseValue*>(output_fragment);
        *out = of[i];
        out += kBlockWidth;
      }
    }
  }

  static __device__ __forceinline__
  void CopyTileToSmem(int k_offset,
                      const ScalarValue* __restrict__ values,
                      const int* __restrict__ columns,
                      ScalarValue* __restrict__ values_tile,
                      ScalarIndex* __restrict__ columns_tile) {
    ScalarValue* vt = reinterpret_cast<ScalarValue*>(values_tile);
    ScalarIndex* ct = reinterpret_cast<ScalarIndex*>(columns_tile);
    const ScalarValue* vg = reinterpret_cast<const ScalarValue*>(values + k_offset);
    const ScalarIndex* cg = reinterpret_cast<const ScalarIndex*>(columns + k_offset);
    for (int t = threadIdx.x; t < kBlockItemsK; t += kBlockWidth) {
      vt[t] = vg[t];
      ct[t] = cg[t];
    }
  }

  static __device__ __forceinline__
  void Kernel4Block_OneTile(int m,int n,
                            const ScalarValue* __restrict__ values,
                            const int * __restrict__ column_indices,
                            const int * __restrict__ row_offsets,
                            const int* __restrict__ row_indices,
                            const int* __restrict__ st_offsets,
                            const ScalarValue * B,ScalarValue *C,
                            int m_tile_idx,int n_tile_idx) {

    int m_idx = m_tile_idx * kBlockItemsY + threadIdx.y;
    int n_idx = n_tile_idx * kBlockItemsX;
    if (m_idx >= m) return;

    int r_idx = Load(row_indices + m_idx);
    int row_offset = Load(st_offsets + m_idx);
    int nonzeros = min(Load(row_offsets + r_idx+1), Load(st_offsets + m_idx + 1)) - row_offset;

    __shared__  ScalarValue values_tile_array[kTileSize];
    __shared__  ScalarIndex column_indices_tile_array[kTileSize];

    ScalarValue * values_tile = values_tile_array + kBlockItemsK * threadIdx.y;
    ScalarIndex * column_indices_tile = column_indices_tile_array + kBlockItemsK * threadIdx.y;

    static constexpr int kValueAligment = int(sizeof(SparseValue) / sizeof(ScalarValue));
    static constexpr uint32_t kAlignmentMask = ~(kValueAligment - 1);

    int values_to_mask_ = row_offset & (kValueAligment - 1);
    int aligned_nonzeros = nonzeros + values_to_mask_;
    nonzeros   = aligned_nonzeros / kBlockItemsK * kBlockItemsK;
    row_offset = row_offset & kAlignmentMask;

    Barrier barrier(threadIdx.y);
    __align__(16) ScalarValue output_fragment[kOutputFragmentSize] = {};

    while (nonzeros >= kBlockItemsK) {
      CopyTileToSmem(0, values + row_offset, column_indices + row_offset, values_tile, column_indices_tile);
      barrier.Sync();

      for (int mask_idx = threadIdx.x; mask_idx < values_to_mask_; mask_idx += kBlockWidth) {
        reinterpret_cast<ScalarValue*>(values_tile)[mask_idx] = ScalarValue(0);
        reinterpret_cast<ScalarIndex*>(column_indices_tile)[mask_idx] = ScalarIndex(0);
      }
      barrier.Sync();

      #pragma unroll
      for (int kk = 0; kk < kBlockItemsK; ++kk) {
        ScalarValue lhs = values_tile[kk];
        int col = column_indices_tile[kk];
        const DenseValue* matrix__ = DenseVecPtr<DenseValue,ScalarValue>(B, n, n_idx, col, threadIdx.x);
        #pragma unroll
        for (int j = 0; j < kDenseThreadItemsX; ++j) {
          ScalarValue* outputs = output_fragment + j * kDenseValuesPerLoad;
          SPC::VectorCompute<DenseValue>::FMA(lhs, Load(matrix__), reinterpret_cast<Accumulator*>(outputs));
          matrix__ += kBlockWidth;
        }
      }

      row_offset += kBlockItemsK;
      nonzeros   -= kBlockItemsK;
      values_to_mask_ = 0;
      barrier.Sync();
    }

    const int output_offset = r_idx * n + n_idx;
    ScalarValue* out = C + output_offset + threadIdx.x * kDenseValuesPerLoad;
    #pragma unroll
    for (int i = 0; i < kDenseThreadItemsX; ++i) {
      #pragma unroll
      for (int j = 0; j < kDenseValuesPerLoad; ++j)
        atomicAdd(out + j, output_fragment[i * kDenseValuesPerLoad + j]);
      out += kBlockWidth * kDenseValuesPerLoad;
    }
  }

  static __device__ __forceinline__
  void Kernel4Block_StreamK(int m,int k,int n,
                            const ScalarValue* __restrict__ values,
                            const int * __restrict__ column_indices,
                            const int * __restrict__ row_offsets,
                            const int* __restrict__ row_indices,
                            const int* __restrict__ st_offsets,
                            const ScalarValue * B,ScalarValue *C) {
    const int tiles_m = (m + kBlockItemsY - 1) / kBlockItemsY;
    const int tiles_n = (n + kBlockItemsX - 1) / kBlockItemsX;
    const unsigned total_tiles = unsigned(tiles_m * tiles_n);
    if (threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0) {
      if (__rode_wq.total_tiles == 0) __rode_wq.total_tiles = total_tiles;
    }
    __syncthreads();

    for (;;) {
      unsigned t = atomicAdd(__rode_wq.counter, 1u);
      if (t >= __rode_wq.total_tiles) break;
      int m_tile = int(t / tiles_n);
      int n_tile = int(t - m_tile * tiles_n);
      Kernel4Block_OneTile(m,n,values,column_indices,row_offsets,row_indices,st_offsets,B,C,m_tile,n_tile);
    }
  }

  static __device__ __forceinline__
  void Kernel4Block(int m,int k,int n,
                    const ScalarValue* __restrict__ values,
                    const int * __restrict__ column_indices,
                    const int * __restrict__ row_offsets,
                    const int* __restrict__ row_indices,
                    const int* __restrict__ st_offsets,
                    const ScalarValue * B,ScalarValue *C) {
#ifdef THREADBLOCK_SWIZZLE
    int m_tile = blockIdx.y;
    int n_tile = blockIdx.x;
#else
    int m_tile = blockIdx.x;
    int n_tile = blockIdx.y;
#endif
    Kernel4Block_OneTile(m,n,values,column_indices,row_offsets,row_indices,st_offsets,B,C,m_tile,n_tile);
  }
};

template <typename ScalarValue,typename SparseValue,typename DenseValue,
          int kBlockItemsY,int kBlockItemsK,int kBlockItemsX,int kBlockWidth,
          int kResidueUnroll,int STAGE>
__global__ void __launch_bounds__(kBlockItemsY*kBlockWidth)
RoDeComputeKernel1(int m,int k,int n,
                   const ScalarValue* __restrict__ values,
                   const int * __restrict__ column_indices,
                   const int *__restrict__ row_offsets,
                   const int *__restrict__ row_indices,
                   const int* __restrict__ row_seg_st_offsets,
                   const ScalarValue * B,ScalarValue* C) {

  const int tiles_m = (m + kBlockItemsY - 1) / kBlockItemsY;
  const int tiles_n = (n + kBlockItemsX - 1) / kBlockItemsX;
  const int total_tiles = tiles_m * tiles_n;

  int sm = __rode_avail_sms;
  int blocks_in_grid = gridDim.x * gridDim.y;
  if (sm <= 0) sm = blocks_in_grid > 0 ? blocks_in_grid : 1;

  const int waves = (total_tiles + sm - 1) / sm;
  const float dp_eff = float(total_tiles) / float(waves * sm);

  const bool use_streamk =
      (__rode_force_streamk != 0) ||
      (waves >= __rode_wave_threshold) ||
      (waves == 2 && __rode_est_iters_per_tile >= __rode_min_iters_per_sk) ||
      (dp_eff < __rode_dp_eff_threshold);

  if (use_streamk) {
    SparseKernel<ScalarValue,SparseValue,DenseValue,kBlockItemsY,kBlockItemsK,kBlockItemsX,kBlockWidth,kResidueUnroll,STAGE>
      ::Kernel4Block_StreamK(m,k,n,values,column_indices,row_offsets,row_indices,row_seg_st_offsets,B,C);
  } else {
    SparseKernel<ScalarValue,SparseValue,DenseValue,kBlockItemsY,kBlockItemsK,kBlockItemsX,kBlockWidth,kResidueUnroll,STAGE>
      ::Kernel4Block(m,k,n,values,column_indices,row_offsets,row_indices,row_seg_st_offsets,B,C);
  }
}

template <typename ScalarValue,typename SparseValue,typename DenseValue,
          int kBlockItemsY,int kBlockItemsK,int kBlockItemsX,int kBlockWidth,
          int kResidueUnroll,int STAGE>
__global__ void __launch_bounds__(kBlockItemsY*kBlockWidth)
RoDeComputeKernel2(int m,int k,int n,
                   const ScalarValue* __restrict__ values,
                   const int * __restrict__ column_indices,
                   const int *__restrict__ row_offsets,
                   const int *__restrict__ row_indices,
                   const ScalarValue * B,ScalarValue* C) {
  SparseKernel<ScalarValue,SparseValue,DenseValue,kBlockItemsY,kBlockItemsK,kBlockItemsX,kBlockWidth,kResidueUnroll,STAGE>
  ::Kernel4Residue(m,k,n,values,column_indices,row_offsets,row_indices,B,C);
}

template <typename ScalarValue,typename SparseValue,typename DenseValue,
          int kBlockItemsY,int kBlockItemsK,int kBlockItemsX1,int kBlockItemsX2,
          int kBlockWidth,int kResidueUnroll,int STAGE = 8>
void RoDeSpmmKernel(int m1,int m2,int k,int n,
                    const ScalarValue* __restrict__ values,
                    const int * __restrict__ column_indices,
                    const int * __restrict__ row_offsets,
                    const int *__restrict__ row_indices1,
                    const int *__restrict__ row_indices2,
                    const int* __restrict__ row_seg_st_offsets,
                    const ScalarValue *B,ScalarValue* C,
                    cudaStream_t stream1,cudaStream_t stream2) {
#ifdef THREADBLOCK_SWIZZLE
  dim3 grid_dim1((n + kBlockItemsX1 - 1) / kBlockItemsX1,(m1 + kBlockItemsY - 1) / kBlockItemsY,1);
  dim3 grid_dim2((n + kBlockItemsX2 - 1) / kBlockItemsX2,(m2 + kBlockItemsY - 1) / kBlockItemsY,1);
#else
  dim3 grid_dim1( (m1 + kBlockItemsY - 1) / kBlockItemsY, (n + kBlockItemsX1 - 1)/kBlockItemsX1,1);
  dim3 grid_dim2( (m2 + kBlockItemsY - 1) / kBlockItemsY, (n + kBlockItemsX2 - 1)/kBlockItemsX2,1);
#endif
  dim3 block_dim( kBlockWidth, kBlockItemsY, 1);

  RoDeComputeKernel1<ScalarValue,SparseValue,DenseValue,kBlockItemsY,kBlockItemsK,kBlockItemsX1,kBlockWidth,kResidueUnroll,STAGE>
    <<<grid_dim1,block_dim,0,stream1>>>(m1,k,n,values,column_indices,row_offsets,row_indices1,row_seg_st_offsets,B,C);

  RoDeComputeKernel2<ScalarValue,SparseValue,DenseValue,kBlockItemsY,kBlockItemsK,kBlockItemsX2,kBlockWidth,kResidueUnroll,STAGE>
    <<<grid_dim2,block_dim,0,stream2>>>(m2,k,n,values,column_indices,row_offsets,row_indices2,B,C);
}

void RoDeSpmm_n32(int m1,int m2,int k,int n,
                  const float* __restrict__ values,
                  const int * __restrict__ column_indices,
                  const int * __restrict__ row_offsets,
                  const int *__restrict__ row_indices1,
                  const int *__restrict__ row_indices2,
                  const int * __restrict__ row_seg_st_offsets,
                  const float *B,float* C,
                  cudaStream_t stream1,cudaStream_t stream2) {
  RoDeSpmmKernel<float,float4,float4,4,32,32,32,8,4>(m1,m2,k,n,values,column_indices,row_offsets,row_indices1,row_indices2,row_seg_st_offsets,B,C,stream1,stream2);
}

void RoDeSpmm_n128(int m1,int m2,int k,int n,
                   const float* __restrict__ values,
                   const int * __restrict__ column_indices,
                   const int * __restrict__ row_offsets,
                   const int *__restrict__ row_indices1,
                   const int *__restrict__ row_indices2,
                   const int * __restrict__ row_seg_st_offsets,
                   const float *B,float* C,
                   cudaStream_t stream1,cudaStream_t stream2) {
  RoDeSpmmKernel<float,float4,float4,4,32,64,64,8,4>(m1,m2,k,n,values,column_indices,row_offsets,row_indices1,row_indices2,row_seg_st_offsets,B,C,stream1,stream2);
}

void RoDeSpmm_n32(int m1,int m2,int k,int n,
                  const double* __restrict__ values,
                  const int * __restrict__ column_indices,
                  const int * __restrict__ row_offsets,
                  const int *__restrict__ row_indices1,
                  const int *__restrict__ row_indices2,
                  const int * __restrict__ row_seg_st_offsets,
                  const double *B,double* C,
                  cudaStream_t stream1,cudaStream_t stream2) {
  RoDeSpmmKernel<double,double4,double4,4,32,32,32,8,4>(m1,m2,k,n,values,column_indices,row_offsets,row_indices1,row_indices2,row_seg_st_offsets,B,C,stream1,stream2);
}

void RoDeSpmm_n128(int m1,int m2,int k,int n,
                   const double* __restrict__ values,
                   const int * __restrict__ column_indices,
                   const int * __restrict__ row_offsets,
                   const int *__restrict__ row_indices1,
                   const int *__restrict__ row_indices2,
                   const int * __restrict__ row_seg_st_offsets,
                   const double *B,double* C,
                   cudaStream_t stream1,cudaStream_t stream2) {
  RoDeSpmmKernel<double,double4,double4,4,32,64,64,8,4,4>(m1,m2,k,n,values,column_indices,row_offsets,row_indices1,row_indices2,row_seg_st_offsets,B,C,stream1,stream2);
}
