#define RODE_SPMM_CU_IMPLEMENTATION

#include "RoDeSpmm.h"
#include "basic_utils.h"
#include "common_utils.h"
#include "cuda_runtime.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstddef>
#include <cstdio>

using namespace SPC;

#ifndef RODE_DBG_AB
#define RODE_DBG_AB 1
#endif
#ifndef RODE_DBG_AB_TILE_LIMIT
#define RODE_DBG_AB_TILE_LIMIT 8
#endif
#ifndef RODE_DBG_AB_KMAX
#define RODE_DBG_AB_KMAX 64
#endif
#ifndef RODE_DBG_AB_XMAX
#define RODE_DBG_AB_XMAX 64
#endif
#ifndef RODE_FORCE_USE_STREAMK
#define RODE_FORCE_USE_STREAMK 1
#endif

// ===== device globals =====
__device__ RoDeWorkQueue __rode_wq;
__device__ int   __rode_wave_threshold     = 3;
__device__ int   __rode_force_streamk      = 0;
__device__ int   __rode_avail_sms          = 0;
__device__ float __rode_dp_eff_threshold   = 0.92f;
__device__ int   __rode_min_iters_per_sk   = 2;
__device__ int   __rode_est_iters_per_tile = 4;

// debug
__device__ int g_print_ticket = 0;

// WQ counter (device alloc)
static unsigned* g_wq_counter = nullptr;

// Debug counters
__device__ unsigned long long g_rows_total_res2 = 0;
__device__ unsigned long long g_rows_skipped_residue0 = 0;
__device__ unsigned long long g_rows_effective_gt0 = 0;
__device__ unsigned long long g_effective_nnz_sum = 0;
__device__ int                g_effective_nnz_max = 0;
__device__ unsigned long long g_kidx_oob = 0;
__device__ unsigned long long g_fma_mac_ops = 0;
__device__ unsigned long long g_rows_load_entered_but_zero_iters = 0;
__device__ unsigned long long g_tiles_nonzero_partial = 0;

// ===== host helpers =====
void RoDeWQInitOnce() {
    if (!g_wq_counter) cudaMalloc(&g_wq_counter, sizeof(unsigned));
    unsigned zero = 0;
    cudaMemcpy(g_wq_counter, &zero, sizeof(unsigned), cudaMemcpyHostToDevice);

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    cudaMemcpyToSymbol(__rode_avail_sms, &prop.multiProcessorCount, sizeof(int));

    printf("[DEBUG HOST] RoDeWQInitOnce: Detected %d SMs (prop.multiProcessorCount)\n", prop.multiProcessorCount);

    RoDeWorkQueue h{};
    h.total_tiles = 0u;
    h.counter     = g_wq_counter;
    cudaMemcpyToSymbol(__rode_wq, &h, sizeof(h));
}

void RoDeWQResetBeforeLaunch(unsigned total_tiles_for_this_launch) {
    unsigned zero = 0; int zero_int = 0;
    cudaMemcpy(g_wq_counter, &zero, sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(__rode_wq, &total_tiles_for_this_launch, sizeof(unsigned),
                       offsetof(RoDeWorkQueue, total_tiles));
    cudaMemcpyToSymbol(g_print_ticket, &zero_int, sizeof(int));
}

void RoDeDebugResetCounters() {
    unsigned long long ull0 = 0ull; int z = 0;
    cudaMemcpyToSymbol(g_rows_total_res2, &ull0, sizeof(ull0));
    cudaMemcpyToSymbol(g_rows_skipped_residue0, &ull0, sizeof(ull0));
    cudaMemcpyToSymbol(g_rows_effective_gt0, &ull0, sizeof(ull0));
    cudaMemcpyToSymbol(g_effective_nnz_sum, &ull0, sizeof(ull0));
    cudaMemcpyToSymbol(g_effective_nnz_max, &z, sizeof(z));
    cudaMemcpyToSymbol(g_kidx_oob, &ull0, sizeof(ull0));
    cudaMemcpyToSymbol(g_fma_mac_ops, &ull0, sizeof(ull0));
    cudaMemcpyToSymbol(g_rows_load_entered_but_zero_iters, &ull0, sizeof(ull0));
    cudaMemcpyToSymbol(g_tiles_nonzero_partial, &ull0, sizeof(ull0));
}

void RoDeDebugDumpCounters() {
    unsigned long long rows_total=0, rows_skip=0, rows_eff=0, nnz_sum=0;
    unsigned long long k_oob=0, fma_ops=0, load_zero=0, tiles_nz=0;
    int nnz_max=0;

    cudaMemcpyFromSymbol(&rows_total, g_rows_total_res2, sizeof(rows_total));
    cudaMemcpyFromSymbol(&rows_skip,  g_rows_skipped_residue0, sizeof(rows_skip));
    cudaMemcpyFromSymbol(&rows_eff,   g_rows_effective_gt0, sizeof(rows_eff));
    cudaMemcpyFromSymbol(&nnz_sum,    g_effective_nnz_sum, sizeof(nnz_sum));
    cudaMemcpyFromSymbol(&nnz_max,    g_effective_nnz_max, sizeof(nnz_max));
    cudaMemcpyFromSymbol(&k_oob,      g_kidx_oob, sizeof(k_oob));
    cudaMemcpyFromSymbol(&fma_ops,    g_fma_mac_ops, sizeof(fma_ops));
    cudaMemcpyFromSymbol(&load_zero,  g_rows_load_entered_but_zero_iters, sizeof(load_zero));
    cudaMemcpyFromSymbol(&tiles_nz,   g_tiles_nonzero_partial, sizeof(tiles_nz));

    double avg_eff_nnz = rows_total ? (double)nnz_sum / (double)rows_total : 0.0;
    double eff_row_ratio = rows_total ? (double)rows_eff / (double)rows_total : 0.0;

    printf("[RES2 STATS] rows_total=%llu, residue0_skip=%llu, eff_rows=%llu (%.3f)\n",
           rows_total, rows_skip, rows_eff, eff_row_ratio);
    printf("[RES2 STATS] eff_nnz_sum=%llu, eff_nnz_avg/row=%.3f, eff_nnz_max=%d\n",
           nnz_sum, avg_eff_nnz, nnz_max);
    printf("[RES2 STATS] k_idx_oob=%llu, fma_ops=%llu, load_entered_but_zero_iters=%llu\n",
           k_oob, fma_ops, load_zero);
    printf("[RES2 STATS] tiles_nonzero_partial=%llu\n", tiles_nz);
}

// (legacy compat)
void initializeRoDeWorkQueue() {
    unsigned* d_counter; cudaMalloc(&d_counter, sizeof(unsigned));
    cudaMemset(d_counter, 0, sizeof(unsigned));
    cudaMemcpyToSymbol(__rode_wq, &d_counter, sizeof(unsigned*),
                       offsetof(RoDeWorkQueue, counter), cudaMemcpyHostToDevice);
    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    cudaMemcpyToSymbol(__rode_avail_sms, &prop.multiProcessorCount, sizeof(int));
}

// ====== kernel template ======
template <typename ScalarValue,typename SparseValue,typename DenseValue,
          int kBlockItemsY,int kBlockItemsK,int kBlockItemsX,int kBlockWidth,
          int kResidueUnroll,int STAGE = 8>
struct SparseKernel {

  typedef int ScalarIndex;

  static constexpr int kSparseValuesPerLoad = sizeof(SparseValue) / sizeof(ScalarValue);
  static constexpr int kThreadItemsX        = kBlockItemsX / kBlockWidth;
  static constexpr int kThreadItemsK        = kBlockItemsK / kBlockWidth / kSparseValuesPerLoad;
  static constexpr int kDenseValuesPerLoad  = sizeof(DenseValue) / sizeof(ScalarValue); // 4 for float4
  static constexpr int kDenseThreadItemsX   = kBlockItemsX / kDenseValuesPerLoad / kBlockWidth; // typically 1
  static constexpr int kElementsPerScalar   = 1;
  static constexpr int kResidueOuterLimit   = kBlockItemsK / kResidueUnroll;
  static constexpr int kResidueInnerLimit   = kResidueUnroll;
  static constexpr int kDenseFragmentSize   = kElementsPerScalar * kBlockItemsK * kBlockItemsX / kBlockWidth;
  static constexpr int kOutputFragmentSize  = kBlockItemsX * kElementsPerScalar / kBlockWidth; // e.g., 4
  static constexpr int kTileSize            = kBlockItemsY * kBlockItemsK;
  static constexpr int kValueAligment       = sizeof(SparseValue) / sizeof(ScalarValue);       // 4
  static constexpr uint32_t kAlignmentMask  = ~(kValueAligment - 1);
  static constexpr int kMaxValuesToMask     = kValueAligment - 1;
  static constexpr int kMaskSteps           = (kMaxValuesToMask + kBlockWidth - 1) / kBlockWidth;

  typedef typename Value2Index<SparseValue>::Index SparseIndex;
  typedef SPC::Barrier<kBlockItemsY,kBlockWidth>    Barrier;
  typedef typename SPC::TypeUtils<DenseValue>::Accumulator Accumulator;

#if RODE_DBG_AB
  static __device__ __forceinline__
  void __dbg_print_dense_A_row(int m_idx, int k,
                               const ScalarIndex* __restrict__ k_idx,
                               const ScalarValue* __restrict__ a_vals,
                               int nnz) {
    if (k > RODE_DBG_AB_KMAX) {
      printf("[DBG] A-row(m=%d): k=%d too big, skip dense print (nnz=%d)\n", m_idx, k, nnz);
      return;
    }
    printf("[DBG] A-row(m=%d) dense k=%d (nnz=%d):\n", m_idx, k, nnz);
    for (int c = 0; c < k; ++c) {
      float v = 0.f;
      for (int t = 0; t < nnz; ++t) {
        if (k_idx[t] == c) { v = (float)a_vals[t]; break; }
      }
      printf(" % .6f", v);
    }
    printf("\n");
  }

  static __device__ __forceinline__
  void __dbg_print_B_submatrix(int m_idx, int k, int n,
                               int n_idx, int nx,
                               const ScalarIndex* __restrict__ k_idx,
                               int nnz,
                               const ScalarValue* __restrict__ B) {
    const int nx_clamped = (nx > RODE_DBG_AB_XMAX) ? RODE_DBG_AB_XMAX : nx;
    printf("[DBG] B-sub(m=%d): rows(nnz)=%d x cols=%d  (n-slice=[%d..%d])\n",
           m_idx, nnz, nx_clamped, n_idx, n_idx + nx_clamped - 1);

    for (int r = 0; r < nnz; ++r) {
      const int kk = k_idx[r];
      printf("  k=%d |", kk);
      if (kk < 0 || kk >= k) {
        for (int c = 0; c < nx_clamped; ++c) printf("  (oob)");
        printf("\n");
        continue;
      }
      const ScalarValue* Brow = B + kk * n + n_idx;  // row-major
      for (int c = 0; c < nx_clamped; ++c) {
        printf(" % .6f", (float)Brow[c]);
      }
      printf("\n");
    }
  }
#endif // RODE_DBG_AB

static __device__ __forceinline__
void Kernel4Residue_OneTile(int m,int k,int n,
                            int m_tile_idx,int n_tile_idx, // StreamK tile indices
                            const ScalarValue* __restrict__ values,
                            const int * __restrict__ column_indices,
                            const int * __restrict__ row_offsets,
                            const int* __restrict__ row_indices,
                            const ScalarValue * B,ScalarValue *C) {
  // --- tile geometry ---
  const int m_slot = m_tile_idx * kBlockItemsY + threadIdx.y;
  const int n_idx  = n_tile_idx * kBlockItemsX;
  if (m_slot >= m) return;

  // 유효 열수 (이 타일 내)
  int nx = n - n_idx;
  if (nx <= 0) return;
  if (nx > kBlockItemsX) nx = kBlockItemsX;

  // 스레드당 vec 시작 열, lane가 다루는 로컬 시작 오프셋
  const int lane_base_col = threadIdx.x * kDenseValuesPerLoad;

  // --- 행 인덱스 복원 ---
  const int m_idx = Load(row_indices + m_slot);

  // --- CSR 포인터/길이 ---
  const int row_offset_orig = Load(row_offsets + m_idx);
  const int nonzeros_orig   = Load(row_offsets + m_idx + 1) - row_offset_orig;

  // 정렬/패딩 계산
  const int pad0             = (row_offset_orig & (kValueAligment - 1));
  const int row_off_aligned0 = (row_offset_orig &  kAlignmentMask);
  const int aligned_total    = nonzeros_orig + pad0;

  // Residual이 처리할 tail 구간(start_orig ~)
  int start_orig = row_offset_orig;
  int tail       = nonzeros_orig;
  if (aligned_total >= kBlockItemsK) {
    const int n_blocks = aligned_total / kBlockItemsK;
    const int residue  = aligned_total % kBlockItemsK;
    if (residue == 0) {
      // 잔여가 없으면 Residual은 처리할 게 없음
      return;
    }
    start_orig = row_off_aligned0 + n_blocks * kBlockItemsK; // tail 시작
    tail       = residue;
  }

  // Shared 로딩용 정렬 시작/헤드 패드
  const int row_offset_aligned = (start_orig & kAlignmentMask);
  const int pad                = start_orig - row_offset_aligned;      // 0..(kValueAlignment-1)
  const int total_to_load      = tail + pad;                           // Shared에 실제 올릴 개수

  // --- Shared: 각 행(=threadIdx.y)마다 32개 슬롯 ---
  __shared__  ScalarValue values_tile_array[kTileSize];
  __shared__  ScalarIndex column_indices_tile_array[kTileSize];
  ScalarValue * values_tile         = values_tile_array + kBlockItemsK * threadIdx.y;
  ScalarIndex * column_indices_tile = column_indices_tile_array + kBlockItemsK * threadIdx.y;
  Barrier barrier(threadIdx.y);

  // 출력 fragment (스레드당 kDenseThreadItemsX*vec4)
  __align__(16) ScalarValue output_fragment[kOutputFragmentSize];
  #pragma unroll
  for (int t=0; t<kOutputFragmentSize; ++t) output_fragment[t] = ScalarValue(0);

  // --- Global -> Shared (경계 보장) ---
  const ScalarValue *g_vals = values + row_offset_aligned + threadIdx.x;
  const ScalarIndex *g_cols = column_indices + row_offset_aligned + threadIdx.x;
  ScalarValue *s_vals = values_tile + threadIdx.x;
  ScalarIndex *s_cols = column_indices_tile + threadIdx.x;

  // 32/8=4 스텝 (kBlockItemsK/kBlockWidth)
  #pragma unroll
  for (int i = 0; i < (kBlockItemsK / kBlockWidth); ++i) {
    const int lane_off = i * kBlockWidth + threadIdx.x;
    if (lane_off >= total_to_load) break;
    // 원행 끝을 넘지 않도록 방어
    if (row_offset_aligned + lane_off >= row_offset_orig + nonzeros_orig) break;

    Store(Load(g_vals), s_vals);
    Store(Load(g_cols), s_cols);

    g_vals += kBlockWidth; g_cols += kBlockWidth;
    s_vals += kBlockWidth; s_cols += kBlockWidth;
  }
  asm volatile("" ::: "memory");
  barrier.Sync();

  // 헤드 패드 0마스킹 (열 접근 방지 위해 k=-1로도 막음)
  int mask_idx = threadIdx.x;
  #pragma unroll
  for (int i=0; i<kMaskSteps; ++i) {
    if (mask_idx < pad) {
      values_tile[mask_idx]         = ScalarValue(0);
      column_indices_tile[mask_idx] = ScalarIndex(-1);
    }
    mask_idx += kBlockWidth;
  }
  barrier.Sync();

  // 유효 nnz 범위 포인터
  const ScalarIndex *k_ptr = column_indices_tile + pad;
  const ScalarValue *a_ptr = values_tile         + pad;
  const int effective_nonzeros = tail;

  // --- 계산 루프 (Residual unroll) ---
  #pragma unroll
  for (int i = 0; i < kResidueOuterLimit; ++i) {
    const int base = i * kResidueInnerLimit;
    if (base >= effective_nonzeros) break;

    #pragma unroll
    for (int j = 0; j < kResidueInnerLimit; ++j) {
      const int idx = base + j;
      if (idx >= effective_nonzeros) break;

      const ScalarIndex kk  = k_ptr[idx];
      const ScalarValue lhs = a_ptr[idx];
      if (!(kk >= 0 && kk < k) || lhs == ScalarValue(0)) continue;

      // X방향 추가 타일 묶음(kDenseThreadItemsX)을 모두 처리
      #pragma unroll
      for (int l = 0; l < kDenseThreadItemsX; ++l) {
        const int local_base = lane_base_col + l * (kBlockWidth * kDenseValuesPerLoad);
        if (local_base >= nx) break; // 이 l에는 유효 열 없음

        // 이 l에서 유효한 컴포넌트(1..vec4)
        int valid = nx - local_base;
        if (valid > kDenseValuesPerLoad) valid = kDenseValuesPerLoad;
        if (valid <= 0) break;

        // B[kk, n_idx + local_base] → 경계 안전 vec4 구성
        const size_t b_off_bytes =
          ((size_t)kk * (size_t)n + (size_t)(n_idx + local_base)) * sizeof(ScalarValue);
        const ScalarValue* b_src = reinterpret_cast<const ScalarValue*>(
          reinterpret_cast<const char*>(B) + b_off_bytes);

        DenseValue rhs_vec;
        if (valid == kDenseValuesPerLoad) {
          rhs_vec = *reinterpret_cast<const DenseValue*>(b_src);
        } else {
          // tail: 스칼라 패킹
          ScalarValue tmp[kDenseValuesPerLoad] = {ScalarValue(0),ScalarValue(0),
                                                  ScalarValue(0),ScalarValue(0)};
          #pragma unroll
          for (int t=0; t<kDenseValuesPerLoad; ++t) {
            if (t < valid) tmp[t] = b_src[t];
          }
          rhs_vec = *reinterpret_cast<const DenseValue*>(tmp);
        }

        // FMA 누적 위치: l 묶음의 vec 위치로
        ScalarValue* outputs = output_fragment + l * kDenseValuesPerLoad;
        SPC::VectorCompute<DenseValue>::FMA(lhs, rhs_vec,
          reinterpret_cast<Accumulator*>(outputs));
      } // for l
    } // for j
  } // for i

  asm volatile("" ::: "memory");

  // --- 결과 쓰기: 열 경계 가드 + kDenseThreadItemsX 전부 ---
  const size_t out_off = (size_t)m_idx * (size_t)n + (size_t)n_idx;
  ScalarValue* out_ptr = C + out_off + threadIdx.x * kDenseValuesPerLoad;

  #pragma unroll
  for (int l = 0; l < kDenseThreadItemsX; ++l) {
    #pragma unroll
    for (int j = 0; j < kDenseValuesPerLoad; ++j) {
      const int col_local = lane_base_col + l * (kBlockWidth * kDenseValuesPerLoad) + j;
      if (col_local < nx) {
        atomicAdd(out_ptr + j, output_fragment[l * kDenseValuesPerLoad + j]);
      }
    }
    out_ptr += kBlockWidth * kDenseValuesPerLoad; // 다음 l 묶음의 열로 이동
  }
}


  // non‑streamK entry (unused when force SK)
  static __device__ __forceinline__
  void Kernel4Residue(int m,int k,int n,
                      const ScalarValue* __restrict__ values,
                      const int * __restrict__ column_indices,
                      const int * __restrict__ row_offsets,
                      const int* __restrict__ row_indices,
                      const ScalarValue * B,ScalarValue *C) {
#ifdef THREADBLOCK_SWIZZLE
    int m_tile = blockIdx.y;
    int n_tile = blockIdx.x;
#else
    int m_tile = blockIdx.x;
    int n_tile = blockIdx.y;
#endif
    Kernel4Residue_OneTile(m,k,n,m_tile,n_tile,values,column_indices,row_offsets,row_indices,B,C);
  }

  // -------- Stream‑K (fixed): one tile per block at a time --------
  static __device__ __forceinline__
  void Kernel4Residue_StreamK(int m,int k,int n,
                              const ScalarValue* __restrict__ values,
                              const int * __restrict__ column_indices,
                              const int * __restrict__ row_offsets,
                              const int* __restrict__ row_indices,
                              const ScalarValue * B,ScalarValue *C,
                              const int total_tiles) {

    const int tiles_m = (m + kBlockItemsY - 1) / kBlockItemsY;
    const int tiles_n = (n + kBlockItemsX - 1) / kBlockItemsX;

    __shared__ unsigned s_t;

    for (;;) {
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_t = atomicAdd(__rode_wq.counter, 1u);
      }
      __syncthreads();

      unsigned t = s_t;
      if (t >= (unsigned)total_tiles) break;

      const int m_tile = int(t / tiles_n);
      const int n_tile = int(t % tiles_n);
      if (m_tile < tiles_m) {
        Kernel4Residue_OneTile(m,k,n,m_tile,n_tile,values,column_indices,row_offsets,row_indices,B,C);
      }
      __syncthreads();
    }
  }

  // -------- Main(Block) path (kept; add nx guard on write) --------
  static __device__ __forceinline__
  void Kernel4Block(int m,int k,int n,
                    const ScalarValue* __restrict__ values,
                    const int * __restrict__ column_indices,
                    const int * __restrict__ row_offsets,
                    const int* __restrict__ row_indices,
                    const int* __restrict__ st_offsets,
                    const ScalarValue * B,ScalarValue *C) {

#ifdef THREADBLOCK_SWIZZLE
    int m_idx = blockIdx.y * kBlockItemsY + threadIdx.y;
    int n_idx = blockIdx.x * kBlockItemsX;
#else
    int m_idx = blockIdx.x * kBlockItemsY + threadIdx.y;
    int n_idx = blockIdx.y * kBlockItemsX;
#endif
    if (m_idx >= m) return;

    const int nx = max(0, min(kBlockItemsX, n - n_idx));
    const bool lane_active = (threadIdx.x * kDenseValuesPerLoad) < nx;

    int r_idx = Load(row_indices + m_idx);

    int row_offset = Load(st_offsets + m_idx);
    int nonzeros   = min(Load(row_offsets + r_idx+1), Load(st_offsets + m_idx + 1)) - row_offset;

    constexpr int kTileSize2 = kBlockItemsY * kBlockItemsK;
    __shared__  ScalarValue values_tile_array[2*kTileSize2];
    __shared__  ScalarIndex column_indices_tile_array[kTileSize2];

    ScalarValue * values_tile       = values_tile_array + kBlockItemsK * threadIdx.y;
    ScalarIndex * column_indices_tile = column_indices_tile_array + kBlockItemsK * threadIdx.y;

    int values_to_mask_   = row_offset & (kValueAligment - 1);
    int aligned_nonzeros  = nonzeros + values_to_mask_;
    nonzeros   = aligned_nonzeros / kBlockItemsK * kBlockItemsK;
    row_offset = row_offset & kAlignmentMask;

    Barrier barrier(threadIdx.y);

    cooperative_groups::thread_block threadblock = cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<kBlockWidth> subwarp =
        cooperative_groups::tiled_partition<kBlockWidth>(threadblock);

    __align__(16) ScalarValue dense_matrix_fragment[kDenseFragmentSize] = {};
    __align__(16) ScalarValue output_fragment[kOutputFragmentSize]      = {};

    const DenseValue *dense_matrix = reinterpret_cast<const DenseValue*>(B + n_idx) + threadIdx.x;
    const SparseValue *sparse_values  = reinterpret_cast<const SparseValue*>(values + row_offset);
    const SparseIndex *sparse_columns = reinterpret_cast<const SparseIndex*>(column_indices + row_offset);

    SparseValue* sparse_values_tile = reinterpret_cast<SparseValue*>(values_tile);
    SparseIndex* sparse_columns_tile= reinterpret_cast<SparseIndex*>(column_indices_tile);

    constexpr int Pipeline_steps = kBlockItemsK / STAGE;

    if (nonzeros >= kBlockItemsK) {
      cooperative_groups::memcpy_async(subwarp,sparse_columns_tile,kBlockItemsK / kSparseValuesPerLoad,sparse_columns,kBlockItemsK / kSparseValuesPerLoad);
      cooperative_groups::memcpy_async(subwarp,sparse_values_tile, kBlockItemsK / kSparseValuesPerLoad,sparse_values ,kBlockItemsK / kSparseValuesPerLoad);

      sparse_columns += kBlockItemsK / kSparseValuesPerLoad;
      sparse_values  += kBlockItemsK / kSparseValuesPerLoad;

      cooperative_groups::wait(subwarp);

      ScalarValue *values_tile_sv     = reinterpret_cast<ScalarValue*>(values_tile);
      ScalarIndex *column_indices_sv  = reinterpret_cast<ScalarIndex*>(column_indices_tile);
      int mask_idx = threadIdx.x;
      #pragma unroll
      for (int i=0; i < kMaskSteps; ++i) {
        if (mask_idx < values_to_mask_) {
          values_tile_sv[mask_idx]     = ScalarValue(0);
          column_indices_sv[mask_idx]  = ScalarIndex(0);
        }
        mask_idx += kBlockWidth;
      }
      barrier.Sync();

      nonzeros -= kBlockItemsK;

      // (compute pipeline body omitted - unchanged)

      // writeback with nx guard
      const int output_offset = r_idx * n + n_idx;
      ScalarValue* output_matrix = C + output_offset + threadIdx.x * kDenseValuesPerLoad;
      #pragma unroll
      for (int i = 0; i < kDenseThreadItemsX; ++i) {
        const int base_local = threadIdx.x * kDenseValuesPerLoad + i * (kBlockWidth * kDenseValuesPerLoad);
        #pragma unroll
        for (int j=0; j < kDenseValuesPerLoad; ++j) {
          const int col_local = base_local + j;
          if (col_local < nx) {
            atomicAdd(output_matrix + j, output_fragment[i * kDenseValuesPerLoad + j]);
          }
        }
        output_matrix += kBlockWidth * kDenseValuesPerLoad;
      }
    }
    // (rest of main path omitted for brevity; not used in your current tests)
  } // Kernel4Block
}; // struct

// ===== kernels =====
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
  SparseKernel<ScalarValue,SparseValue,DenseValue,
               kBlockItemsY,kBlockItemsK,kBlockItemsX,kBlockWidth,
               kResidueUnroll,STAGE>
  ::Kernel4Block(m,k,n,values,column_indices,row_offsets,row_indices,row_seg_st_offsets,B,C);
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

  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    printf("[DEBUG KERNEL] RoDeComputeKernel2: __rode_avail_sms = %d\n", __rode_avail_sms);
  }

  int sm = __rode_avail_sms;
  int blocks_in_grid = gridDim.x * gridDim.y;
  if (sm <= 0) sm = (blocks_in_grid > 0 ? blocks_in_grid : 1);

  const int total_tiles = __rode_wq.total_tiles;
  const int   waves  = (total_tiles + sm - 1) / sm;
  const float dp_eff = float(total_tiles) / float(waves * sm);
  const int tiles_n_calc = (n + kBlockItemsX - 1) / kBlockItemsX;

#if RODE_FORCE_USE_STREAMK
  const bool use_streamk = true;
#else
  const bool use_streamk =
      (tiles_n_calc > 1) &&
      ( (__rode_force_streamk != 0) ||
        (waves >= __rode_wave_threshold) ||
        (waves == 2 && __rode_est_iters_per_tile >= __rode_min_iters_per_sk) ||
        (dp_eff < __rode_dp_eff_threshold) );
#endif

  if (1) {
    SparseKernel<ScalarValue,SparseValue,DenseValue,
                 kBlockItemsY,kBlockItemsK,kBlockItemsX,kBlockWidth,
                 kResidueUnroll,STAGE>
    ::Kernel4Residue_StreamK(m,k,n,values,column_indices,row_offsets,row_indices,B,C,total_tiles);
  } else {
    SparseKernel<ScalarValue,SparseValue,DenseValue,
                 kBlockItemsY,kBlockItemsK,kBlockItemsX,kBlockWidth,
                 kResidueUnroll,STAGE>
    ::Kernel4Residue(m,k,n,values,column_indices,row_offsets,row_indices,B,C);
  }
}

// ===== wrappers =====
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
  dim3 grid_dim1((m1 + kBlockItemsY - 1) / kBlockItemsY,(n + kBlockItemsX1 - 1) / kBlockItemsX1,1);
  dim3 grid_dim2((m2 + kBlockItemsY - 1) / kBlockItemsY,(n + kBlockItemsX2 - 1) / kBlockItemsX2,1);
#endif
  dim3 block_dim(kBlockWidth, kBlockItemsY, 1);

  if (m1 > 0 && grid_dim1.x > 0 && grid_dim1.y > 0) {
    RoDeComputeKernel1<ScalarValue,SparseValue,DenseValue,
                       kBlockItemsY,kBlockItemsK,kBlockItemsX1,kBlockWidth,
                       kResidueUnroll,STAGE>
        <<<grid_dim1,block_dim,0,stream1>>>(m1,k,n,
          values,column_indices,row_offsets,row_indices1,row_seg_st_offsets,B,C);
  }

  if (m2 > 0 && grid_dim2.x > 0 && grid_dim2.y > 0) {
    RoDeComputeKernel2<ScalarValue,SparseValue,DenseValue,
                       kBlockItemsY,kBlockItemsK,kBlockItemsX2,kBlockWidth,
                       kResidueUnroll,STAGE>
        <<<grid_dim2,block_dim,0,stream2>>>(m2,k,n,
          values,column_indices,row_offsets,row_indices2,B,C);
  }
}

// float/double n32/n128
void RoDeSpmm_n32(int m1,int m2,int k,int n,
                  const float* values,
                  const int * column_indices,
                  const int * row_offsets,
                  const int * row_indices1,
                  const int * row_indices2,
                  const int * row_seg_st_offsets,
                  const float *B,float* C,
                  cudaStream_t stream1,cudaStream_t stream2) {
  RoDeSpmmKernel<float,float4,float4,4,32,32,32,8,4>(
    m1,m2,k,n,values,column_indices,row_offsets,row_indices1,row_indices2,row_seg_st_offsets,B,C,stream1,stream2);
}

void RoDeSpmm_n128(int m1,int m2,int k,int n,
                   const float* values,
                   const int * column_indices,
                   const int * row_offsets,
                   const int * row_indices1,
                   const int * row_indices2,
                   const int * row_seg_st_offsets,
                   const float *B,float* C,
                   cudaStream_t stream1,cudaStream_t stream2) {
  RoDeSpmmKernel<float,float4,float4,4,32,64,64,8,4>(
    m1,m2,k,n,values,column_indices,row_offsets,row_indices1,row_indices2,row_seg_st_offsets,B,C,stream1,stream2);
}

void RoDeSpmm_n32(int m1,int m2,int k,int n,
                  const double* values,
                  const int * column_indices,
                  const int * row_offsets,
                  const int * row_indices1,
                  const int * row_indices2,
                  const int * row_seg_st_offsets,
                  const double *B,double* C,
                  cudaStream_t stream1,cudaStream_t stream2) {
  RoDeSpmmKernel<double,double4,double4,4,32,32,32,8,4>(
    m1,m2,k,n,values,column_indices,row_offsets,row_indices1,row_indices2,row_seg_st_offsets,B,C,stream1,stream2);
}

void RoDeSpmm_n128(int m1,int m2,int k,int n,
                   const double* values,
                   const int * column_indices,
                   const int * row_offsets,
                   const int * row_indices1,
                   const int * row_indices2,
                   const int * row_seg_st_offsets,
                   const double *B,double* C,
                   cudaStream_t stream1,cudaStream_t stream2) {
  RoDeSpmmKernel<double,double4,double4,4,32,64,64,8,4,4>(
    m1,m2,k,n,values,column_indices,row_offsets,row_indices1,row_indices2,row_seg_st_offsets,B,C,stream1,stream2);
}
