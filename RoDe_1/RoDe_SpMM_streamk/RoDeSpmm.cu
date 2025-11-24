// Stream_RoDe_SpMM.cu

#include "RoDeSpmm.h"
#include "basic_utils.h"
#include "cuda_runtime.h"
#include "common_utils.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
using namespace SPC;

struct __align__(8) RoDeWorkQueue { unsigned total_tiles; unsigned* counter; };

__device__ RoDeWorkQueue __rode_wq;
__device__ int   __rode_wave_threshold     = 3;
__device__ int   __rode_force_streamk      = 0;
__device__ int   __rode_avail_sms          = 0;
__device__ float __rode_dp_eff_threshold   = 0.92f;
__device__ int   __rode_min_iters_per_sk   = 2;
__device__ int   __rode_est_iters_per_tile = 4;

template <typename ScalarValue,typename SparseValue,typename DenseValue,
          int kBlockItemsY,int kBlockItemsK,int kBlockItemsX,int kBlockWidth,
          int kResidueUnroll,int STAGE = 8>
struct SparseKernel {

  typedef int ScalarIndex;
  static constexpr int kSparseValuesPerLoad = sizeof(SparseValue) / sizeof(ScalarValue);
  static constexpr int kThreadItemsX = kBlockItemsX / kBlockWidth;
  static constexpr int kThreadItemsK = kBlockItemsK / kBlockWidth / kSparseValuesPerLoad;
  static constexpr int kDenseValuesPerLoad = sizeof(DenseValue) / sizeof(ScalarValue);
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

    static constexpr int kValueAligment = sizeof(SparseValue) / sizeof(ScalarValue);
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
    const DenseValue *dense_matrix = reinterpret_cast<const DenseValue*>(B + n_idx) + threadIdx.x;

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

    const ScalarIndex *dense_row_offsets = column_indices_tile;
    #pragma unroll
    for (int i = 0; i < kResidueOuterLimit; ++i) {
      if (nonzeros <= 0) break;
      #pragma unroll
      for (int j = 0; j < kResidueInnerLimit; ++j) {
        const int k_item_idx = i * kResidueInnerLimit + j;
        ScalarIndex scaled_indices = dense_row_offsets[0] * n * sizeof(ScalarValue);
        ScalarValue lhs_values = values_tile[k_item_idx];
        const DenseValue* matrix__ = SPC::OffsetCast<const DenseValue>(dense_matrix, scaled_indices);
        #pragma unroll
        for (int l = 0; l < kDenseThreadItemsX; ++l) {
          ScalarValue *outputs = output_fragment + l * kDenseValuesPerLoad;
          SPC::VectorCompute<DenseValue>::FMA(lhs_values, Load(matrix__), reinterpret_cast<Accumulator*>(outputs));
          matrix__ += kBlockWidth;
        }
        ++dense_row_offsets;
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
        for (int j = 0; j < kDenseValuesPerLoad; ++j) atomicAdd(out + j, output_fragment[i * kDenseValuesPerLoad + j]);
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

    constexpr int kTileSize2 = kBlockItemsY * kBlockItemsK;
    __shared__  ScalarValue values_tile_array[2 * kTileSize2];
    __shared__  ScalarIndex column_indices_tile_array[kTileSize2];

    ScalarValue * values_tile = values_tile_array + kBlockItemsK * threadIdx.y;
    ScalarIndex * column_indices_tile = column_indices_tile_array + kBlockItemsK * threadIdx.y;

    static constexpr int kValueAligment = sizeof(SparseValue) / sizeof(ScalarValue);
    static constexpr uint32_t kAlignmentMask = ~(kValueAligment - 1);
    int values_to_mask_ = row_offset & (kValueAligment - 1);
    int aligned_nonzeros = nonzeros + values_to_mask_;
    nonzeros   = aligned_nonzeros / kBlockItemsK * kBlockItemsK;
    row_offset = row_offset & kAlignmentMask;

    Barrier barrier(threadIdx.y);
    cooperative_groups::thread_block threadblock = cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<kBlockWidth> subwarp = cooperative_groups::tiled_partition<kBlockWidth>(threadblock);

    __align__(16) ScalarValue dense_matrix_fragment[kDenseFragmentSize] = {};
    __align__(16) ScalarValue output_fragment[kOutputFragmentSize] = {};
    const DenseValue *dense_matrix = reinterpret_cast<const DenseValue*>(B + n_idx) + threadIdx.x;

    const SparseValue *sparse_values  = reinterpret_cast<const SparseValue*>(values + row_offset);
    const SparseIndex *sparse_columns = reinterpret_cast<const SparseIndex*>(column_indices + row_offset);

    SparseValue* sparse_values_tile = reinterpret_cast<SparseValue*>(values_tile);
    SparseIndex* sparse_columns_tile = reinterpret_cast<SparseIndex*>(column_indices_tile);

    constexpr int Pipeline_steps = kBlockItemsK / STAGE;

    if (nonzeros >= kBlockItemsK) {
      cooperative_groups::memcpy_async(subwarp,sparse_columns_tile,kBlockItemsK / kSparseValuesPerLoad,sparse_columns,kBlockItemsK / kSparseValuesPerLoad);
      cooperative_groups::memcpy_async(subwarp,sparse_values_tile, kBlockItemsK / kSparseValuesPerLoad,sparse_values ,kBlockItemsK / kSparseValuesPerLoad);
      sparse_columns += kBlockItemsK / kSparseValuesPerLoad;
      sparse_values  += kBlockItemsK / kSparseValuesPerLoad;
      cooperative_groups::wait(subwarp);

      int mask_idx = threadIdx.x;
      #pragma unroll
      for (int i=0;i<( (kValueAligment-1 + kBlockWidth - 1)/kBlockWidth );++i) {
        if (mask_idx < (row_offset & (kValueAligment - 1))) {
          reinterpret_cast<ScalarValue*>(values_tile)[mask_idx] = 0.0f;
          reinterpret_cast<ScalarIndex*>(column_indices_tile)[mask_idx] = 0;
          mask_idx += kBlockWidth;
        }
      }
      cooperative_groups::wait(subwarp);

      nonzeros -= kBlockItemsK;

      ScalarValue * values_tile0 = values_tile_array + kBlockItemsK * threadIdx.y;
      const ScalarIndex *dense_row_offsets = column_indices_tile;
      DenseValue dense_fragment_regs[2][STAGE * kDenseThreadItemsX];

      #pragma unroll
      for (int k_st = 0; k_st < STAGE; ++k_st) {
        ScalarIndex row_idx = dense_row_offsets[k_st] * n * sizeof(ScalarValue);
        const DenseValue *dense_values = SPC::OffsetCast<const DenseValue>(dense_matrix, row_idx);
        #pragma unroll
        for (int e = 0; e < kDenseThreadItemsX; ++e)
          dense_fragment_regs[0][k_st * kDenseThreadItemsX + e] = dense_values[e * kBlockWidth];
      }
      dense_row_offsets += STAGE;

      int row_idxs[STAGE];
      #pragma unroll
      for (int k_st = 0; k_st < STAGE; ++k_st)
        row_idxs[k_st] = dense_row_offsets[k_st] * n * sizeof(ScalarValue);

      #pragma unroll
      for (int i = 1; i < Pipeline_steps; ++i) {
        #pragma unroll
        for (int k_st = 0; k_st < STAGE; ++k_st) {
          ScalarIndex row_idx = row_idxs[k_st];
          const DenseValue *dense_values = SPC::OffsetCast<const DenseValue>(dense_matrix, row_idx);
          #pragma unroll
          for (int e = 0; e < kDenseThreadItemsX; ++e)
            dense_fragment_regs[(i & 1)][k_st * kDenseThreadItemsX + e] = dense_values[e * kBlockWidth];
        }
        dense_row_offsets += STAGE;
        barrier.Sync();

        ScalarValue lhs[STAGE];
        int d_idx = (i - 1) * STAGE;
        #pragma unroll
        for (int k_st = 0; k_st < STAGE; ++k_st) {
          lhs[k_st] = values_tile0[d_idx + k_st];
          row_idxs[k_st] = dense_row_offsets[k_st] * n * sizeof(ScalarValue);
          #pragma unroll
          for (int j = 0; j < kDenseThreadItemsX; ++j) {
            DenseValue rhs = dense_fragment_regs[(i - 1) & 1][k_st * kDenseThreadItemsX + j];
            ScalarValue *outputs = output_fragment + j * kDenseValuesPerLoad;
            SPC::VectorCompute<DenseValue>::FMA(lhs[k_st], rhs, reinterpret_cast<Accumulator*>(outputs));
          }
        }
      }
      barrier.Sync();

      if (nonzeros >= kBlockItemsK) {
        cooperative_groups::memcpy_async(subwarp,sparse_columns_tile,kBlockItemsK / kSparseValuesPerLoad,sparse_columns,kBlockItemsK / kSparseValuesPerLoad);
        sparse_columns += kBlockItemsK / kSparseValuesPerLoad;
        cooperative_groups::memcpy_async(subwarp,sparse_values_tile + kTileSize / kSparseValuesPerLoad, kBlockItemsK / kSparseValuesPerLoad,sparse_values, kBlockItemsK / kSparseValuesPerLoad);
        sparse_values  += kBlockItemsK / kSparseValuesPerLoad;
      }

      ScalarValue lhs2[STAGE];
      int d_idx2 = (Pipeline_steps - 1) * (STAGE);
      #pragma unroll
      for (int k_st = 0; k_st < STAGE; ++k_st) {
        lhs2[k_st] = values_tile0[d_idx2 + k_st];
        #pragma unroll
        for (int j = 0; j < kDenseThreadItemsX; ++j) {
          DenseValue rhs = dense_fragment_regs[(Pipeline_steps - 1) & 1][k_st * kDenseThreadItemsX + j];
          ScalarValue *outputs = output_fragment + j * kDenseValuesPerLoad;
          SPC::VectorCompute<DenseValue>::FMA(lhs2[k_st], rhs, reinterpret_cast<Accumulator*>(outputs));
        }
      }
    }

    int col_off = 1;
    for (; nonzeros >= kBlockItemsK; nonzeros -= kBlockItemsK, col_off ^= 1) {
      cooperative_groups::wait(subwarp);

      ScalarValue * values_tile1 = values_tile_array + kBlockItemsK * threadIdx.y + col_off * kTileSize2;
      const ScalarIndex *dense_row_offsets = column_indices_tile;
      DenseValue dense_fragment_regs[2][STAGE * kDenseThreadItemsX];

      #pragma unroll
      for (int k_st = 0; k_st < STAGE; ++k_st) {
        ScalarIndex row_idx = dense_row_offsets[k_st] * n * sizeof(ScalarValue);
        const DenseValue *dense_values = SPC::OffsetCast<const DenseValue>(dense_matrix, row_idx);
        #pragma unroll
        for (int e = 0; e < kDenseThreadItemsX; ++e)
          dense_fragment_regs[0][k_st * kDenseThreadItemsX + e] = dense_values[e * kBlockWidth];
      }
      dense_row_offsets += STAGE;

      int row_idxs[STAGE];
      #pragma unroll
      for (int k_st = 0; k_st < STAGE; ++k_st)
        row_idxs[k_st] = dense_row_offsets[k_st] * n * sizeof(ScalarValue);

      #pragma unroll
      for (int i = 1; i < Pipeline_steps; ++i) {
        #pragma unroll
        for (int k_st = 0; k_st < STAGE; ++k_st) {
          ScalarIndex row_idx = row_idxs[k_st];
          const DenseValue *dense_values = SPC::OffsetCast<const DenseValue>(dense_matrix, row_idx);
          #pragma unroll
          for (int e = 0; e < kDenseThreadItemsX; ++e)
            dense_fragment_regs[(i & 1)][k_st * kDenseThreadItemsX + e] = dense_values[e * kBlockWidth];
        }
        dense_row_offsets += STAGE;
        Barrier(threadIdx.y).Sync();

        ScalarValue lhs[STAGE];
        int d_idx = (i - 1) * (STAGE);
        #pragma unroll
        for (int k_st = 0; k_st < STAGE; ++k_st) {
          lhs[k_st] = values_tile1[d_idx + k_st];
          row_idxs[k_st] = dense_row_offsets[k_st] * n * sizeof(ScalarValue);
          #pragma unroll
          for (int j = 0; j < kDenseThreadItemsX; ++j) {
            DenseValue rhs = dense_fragment_regs[(i - 1) & 1][k_st * kDenseThreadItemsX + j];
            ScalarValue *outputs = output_fragment + j * kDenseValuesPerLoad;
            SPC::VectorCompute<DenseValue>::FMA(lhs[k_st], rhs, reinterpret_cast<Accumulator*>(outputs));
          }
        }
      }
      Barrier(threadIdx.y).Sync();

      if (nonzeros >= 2 * kBlockItemsK) {
        cooperative_groups::memcpy_async(subwarp,sparse_columns_tile,kBlockItemsK / kSparseValuesPerLoad,sparse_columns,kBlockItemsK / kSparseValuesPerLoad);
        sparse_columns += kBlockItemsK / kSparseValuesPerLoad;
        cooperative_groups::memcpy_async(subwarp,sparse_values_tile + (col_off ^ 1) * kTileSize2 / kSparseValuesPerLoad, kBlockItemsK / kSparseValuesPerLoad,sparse_values, kBlockItemsK / kSparseValuesPerLoad);
        sparse_values  += kBlockItemsK / kSparseValuesPerLoad;
      }

      ScalarValue lhs3[STAGE];
      int d_idx3 = (Pipeline_steps - 1) * (STAGE);
      #pragma unroll
      for (int k_st = 0; k_st < STAGE; ++k_st) {
        lhs3[k_st] = values_tile1[d_idx3 + k_st];
        #pragma unroll
        for (int j = 0; j < kDenseThreadItemsX; ++j) {
          DenseValue rhs = dense_fragment_regs[(Pipeline_steps - 1) & 1][k_st * kDenseThreadItemsX + j];
          ScalarValue *outputs = output_fragment + j * kDenseValuesPerLoad;
          SPC::VectorCompute<DenseValue>::FMA(lhs3[k_st], rhs, reinterpret_cast<Accumulator*>(outputs));
        }
      }
    }

    const int output_offset = r_idx * n + n_idx;
    ScalarValue* out = C + output_offset + threadIdx.x * kDenseValuesPerLoad;
    #pragma unroll
    for (int i = 0; i < kDenseThreadItemsX; ++i) {
      #pragma unroll
      for (int j = 0; j < kDenseValuesPerLoad; ++j) atomicAdd(out + j, output_fragment[i * kDenseValuesPerLoad + j]);
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
  if (sm <= 0) sm = gridDim.x * gridDim.y > 0 ? gridDim.x * gridDim.y : 1;

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
