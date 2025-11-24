#include "cuda_runtime.h"
#include "matrix_utils.h"

#include "Sputnik_spmm.h"
#include "cuSPARSE_spmm.h"
#include "RoDe_SpMM/RoDeSpmm.h"

#include <sys/io.h>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <vector>
#include <cmath>

using namespace std;
using namespace SPC;

#define SEG_LENGTH 32
#define BN 16   // 16x16 실험

__global__ void MatrixDiff(int n,float* res,const float* __restrict__ A,const float* __restrict__ B) {
    if (threadIdx.x == 0 && blockIdx.x == 0) res[0] = 0.0f;
    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float diff = fabsf(A[idx] - B[idx]);
    if (diff > 1e-5f) {
        printf("Mismatch[%d]: cuSPARSE=%f  RoDe=%f  diff=%f\n", idx, A[idx], B[idx], diff);
    }
    float r = diff;
    r += __shfl_down_sync(0xffffffff, r, 16);
    r += __shfl_down_sync(0xffffffff, r, 8);
    r += __shfl_down_sync(0xffffffff, r, 4);
    r += __shfl_down_sync(0xffffffff, r, 2);
    r += __shfl_down_sync(0xffffffff, r, 1);
    if ((threadIdx.x & 31) == 0) atomicAdd(res, r);
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) printf("Matrix diff: %f\n", res[0]);
}

static void printCSR(const char* tag,
                     int m, int n, int nnz,
                     const int* row_offsets, const int* col_indices,
                     const float* values) {
    printf("\n==== %s: CSR (%d x %d), nnz=%d ====\n", tag, m, n, nnz);
    printf("RowOffsets: ");
    for (int i = 0; i <= m; ++i) printf("%d%s", row_offsets[i], (i==m? "\n" : " "));
    for (int r = 0; r < m; ++r) {
        int start = row_offsets[r], end = row_offsets[r+1];
        for (int p = start; p < end; ++p) {
            printf("  (%d, %d) = %.9f\n", r, col_indices[p], values[p]);
        }
    }
    printf("====================================\n");
}

static void printDenseRowMajor(const char* tag, int m, int n, const float* h) {
    printf("\n==== %s: dense (%d x %d), row-major ====\n", tag, m, n);
    for (int i = 0; i < m; ++i) {
        printf("[%2d] ", i);
        for (int j = 0; j < n; ++j) {
            printf("% .6f%s", h[i*n + j], (j==n-1? "" : " "));
        }
        printf("\n");
    }
    printf("====================================\n");
}

int main(int argc, char **argv) {
    // cudaSetDevice(0);

    string file_path;
    if (argc < 2) {
        cout << "No file path" << endl;
        return 0;
    } else {
        file_path = argv[1];
    }

    const int ITER = 1;

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    double gflops = 0.0;

    // Host CSR
    SPC::SparseMatrix sm1(file_path, SPC::SORTED, 1);

    int * row_offsets_h = sm1.RowOffsets();
    int * col_indices_h = sm1.ColumnIndices();
    float* values_h     = sm1.Values();

    // Segment 메타는 만들되, 이번 실험에서는 Residue를 "모든 행"으로 돌린다
    sm1.RowDivide2Segment(SEG_LENGTH, 4, 32);
    SPC::CudaSparseMatrix<float> c_sm(sm1);

    int m = c_sm.Rows();        // 16
    int k = c_sm.Columns();     // 16
    int n = BN;                 // 16

    printCSR("A (Host)", m, k, c_sm.Nonzeros(), row_offsets_h, col_indices_h, values_h);

    absl::BitGen bitgen;
    SPC::CudaMatrix<float> d_B(k, n, &bitgen);

    std::vector<float> hB(k * n);
    cudaMemcpy(hB.data(), d_B.Values(), sizeof(float) * k * n, cudaMemcpyDeviceToHost);
    printDenseRowMajor("B (Host copied from Device)", k, n, hB.data());

    float* d_C;   cudaMalloc((void**)&d_C,  sizeof(float) * m * n); // cuSPARSE
    float* d_C1;  cudaMalloc((void**)&d_C1, sizeof(float) * m * n); // Sputnik
    float* d_C2;  cudaMalloc((void**)&d_C2, sizeof(float) * m * n); // RoDe
    float* diff;  cudaMalloc((void**)&diff, sizeof(float) * 1);

    cudaMemset(d_C,  0, sizeof(float) * m * n);
    cudaMemset(d_C1, 0, sizeof(float) * m * n);
    cudaMemset(d_C2, 0, sizeof(float) * m * n);

    float tot_ms;
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    // ---------- Sputnik ----------
    cudaDeviceSynchronize();
    cudaEventRecord(event1, 0);
    for (int i = 0; i < ITER; ++i) {
        SPC::SputnikSpmm(m, c_sm.Columns(), n, c_sm.Nonzeros(),
                         c_sm.RowIndices(), c_sm.Values(), c_sm.RowOffsets(), c_sm.ColumnIndices(),
                         d_B.Values(), d_C1, stream1);
    }
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&tot_ms, event1, event2);
    cudaDeviceSynchronize();
    gflops = (double)ITER * (double)c_sm.Nonzeros() * 2.0 * n / tot_ms / 1e6;
    printf("Sputnik  : %f ms, %f GFLOP/s\n", tot_ms, gflops);
    {
        std::vector<float> hC1(m*n);
        cudaMemcpy(hC1.data(), d_C1, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
        printDenseRowMajor("C (Sputnik)", m, n, hC1.data());
    }

    // ---------- cuSPARSE ----------
    cuSparse_SPMM<float> cu_sp;
    cu_sp.Preprocess(m, c_sm.Columns(), c_sm.Nonzeros(),
                     c_sm.RowOffsets(), c_sm.ColumnIndices(), c_sm.Values());
    cudaDeviceSynchronize();
    cudaEventRecord(event1, 0);
    for (int i = 0; i < ITER; ++i) {
        cu_sp.Process(n, d_B.Values(), d_C);
    }
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&tot_ms, event1, event2);
    cudaDeviceSynchronize();
    gflops = (double)ITER * (double)c_sm.Nonzeros() * 2.0 * n / tot_ms / 1e6;
    printf("cuSPARSE : %f ms, %f GFLOP/s\n", tot_ms, gflops);
    {
        std::vector<float> hC(m*n);
        cudaMemcpy(hC.data(), d_C, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
        printDenseRowMajor("C (cuSPARSE)", m, n, hC.data());
    }

    // ---------- RoDe (Residue=모든 행, Stream-K 강제) ----------
    RoDeWQInitOnce();
    RoDeDebugResetCounters();

    // row_indices2 = [0..m-1] 를 디바이스에 만들어 전달
    std::vector<int> h_row_ids(m);
    for (int r = 0; r < m; ++r) h_row_ids[r] = r;
    int* d_row_ids = nullptr;
    cudaMalloc(&d_row_ids, sizeof(int) * m);
    cudaMemcpy(d_row_ids, h_row_ids.data(), sizeof(int) * m, cudaMemcpyHostToDevice);

    // 총 타일 수 (m2_residue = m)
    const int kBlockItemsY_n32 = 4;
    const int kBlockItemsX_n32 = 32;   // RoDeSpmm_n32 정의에 맞춤
    const int m2_residue = m;          // ★ 모든 행을 Residue로
    const int tiles_m_res = (m2_residue + kBlockItemsY_n32 - 1) / kBlockItemsY_n32;
    const int tiles_n_res = (n + kBlockItemsX_n32 - 1) / kBlockItemsX_n32;
    const unsigned total_residue_tiles = (unsigned)(tiles_m_res * tiles_n_res);
    RoDeWQResetBeforeLaunch(total_residue_tiles);

    // 실행
    cudaMemset(d_C2, 0, sizeof(float) * m * n);
    cudaDeviceSynchronize();
    cudaEventRecord(event1, 0);
    for (int i = 0; i < ITER; ++i) {
        RoDeSpmm_n32(
            /*m1=*/0,                       // 이번 실험은 Main 비활성 (Residue만)
            /*m2=*/m2_residue,
            c_sm.Columns(), n,
            c_sm.Values(), c_sm.ColumnIndices(), c_sm.RowOffsets(),
            /*row_indices1=*/c_sm.seg_row_indices,    // 사용 안됨
            /*row_indices2=*/d_row_ids,               // ★ 모든 행
            c_sm.seg_st_offsets,
            d_B.Values(), d_C2, stream1, stream2
        );
    }
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&tot_ms, event1, event2);
    cudaDeviceSynchronize();
    gflops = (double)ITER * (double)c_sm.Nonzeros() * 2.0 * n / tot_ms / 1e6;
    printf("RoDe     : %f ms, %f GFLOP/s\n", tot_ms, gflops);

    RoDeDebugDumpCounters();

    {
        std::vector<float> hC2(m*n);
        cudaMemcpy(hC2.data(), d_C2, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
        printDenseRowMajor("C (RoDe)", m, n, hC2.data());
    }

    // ---------- 검증 ----------
    printf("\nValidating results (cuSPARSE vs RoDe)...\n");
    const int total_elements = m * n;
    if (total_elements > 0) {
        int blockSize = 256;
        int gridSize = (total_elements + blockSize - 1) / blockSize;
        MatrixDiff<<<gridSize, blockSize>>>(total_elements, diff, d_C, d_C2);
        cudaDeviceSynchronize();
        float h_diff = 0.0f;
        cudaMemcpy(&h_diff, diff, sizeof(float), cudaMemcpyDeviceToHost);
        printf("Validation complete. Total accumulated difference: %f\n", h_diff);
        if (h_diff > 1e-3f) printf("WARNING: Significant difference detected!\n");
        else                 printf("SUCCESS: Results match cuSPARSE.\n");
    }

    // cleanup
    cudaFree(d_row_ids);
    cudaFree(d_C);
    cudaFree(d_C1);
    cudaFree(d_C2);
    cudaFree(diff);
    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    return 0;
}
