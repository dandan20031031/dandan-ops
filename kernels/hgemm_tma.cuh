// Extracted from hgemm_v10_tma.cu; arithmetic and TMA pipeline preserved.
// Inference epilogue rounds FP32 accumulators directly to FP16.
// Original SHA256: d5f1860c939cd1ced07be02a66521316fdaf0eedffc52c94c70ad0c1355a4b6a
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
namespace dandan_tma {

#ifndef BLOCK_M
#define BLOCK_M 128
#endif
#ifndef BLOCK_N
#define BLOCK_N 64
#endif
#ifndef BLOCK_K
#define BLOCK_K 64
#endif
#ifndef NUM_WARP_M
#define NUM_WARP_M 8
#endif
#ifndef NUM_WARP_N
#define NUM_WARP_N 1
#endif
#ifndef NUM_STAGES
#define NUM_STAGES 2
#endif

#ifndef SUPER_M
#define SUPER_M 16
#endif
#ifndef SUPER_N
#define SUPER_N 2
#endif

#define WARP_SIZE 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_M (BLOCK_M / NUM_WARP_M)
#define WARP_N (BLOCK_N / NUM_WARP_N)
#define TB_SIZE (NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
#define A_SZ (BLOCK_M * BLOCK_K * 2)   
#define B_SZ (BLOCK_N * BLOCK_K * 2)   

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__ void ldmatrix_x4(int reg[4], int addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
               : "r"(addr));
}

__device__ void mma_m16n8k16(const int A[4], const int B[2], float C[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
              : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]));
}


__device__ static void mbar_init(int bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
__device__ static void mbar_arrive(int bar) {
  asm volatile("{\n\t.reg .b64 st;\n\tmbarrier.arrive.shared::cta.b64 st, [%0];\n\t}"
               :: "r"(bar) : "memory");
}
__device__ static void mbar_expect_tx(int bar, int bytes) {
  asm volatile("{\n\t.reg .b64 st;\n\tmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n\t}"
               :: "r"(bar), "r"(bytes) : "memory");
}
__device__ static void mbar_wait(int bar, int phase) {
  asm volatile("{\n\t.reg .pred p;\n"
               "WAIT_%=:\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
               "@!p bra WAIT_%=;\n\t}"
               :: "r"(bar), "r"(phase) : "memory");
}

__device__ static void tma_load_2d(int dst, const CUtensorMap *tmap, int x, int y, int bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"((unsigned long long)tmap), "r"(x), "r"(y), "r"(bar)
               : "memory");
}

template <int TBM, int TBN, int TBK,
          int TWM_N_, int TWN_N_, int TSTAGES,
          int TSM, int TSN>
__global__
__launch_bounds__(TWM_N_ * TWN_N_ * WARP_SIZE)
void hgemm_v10_tma_kernel(const __grid_constant__ CUtensorMap tmapA,
                          const __grid_constant__ CUtensorMap tmapB,
                          half *C, int M, int N, int K) {
  const int tid     = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  constexpr int TWARP_M = TBM / TWM_N_;
  constexpr int TWARP_N = TBN / TWN_N_;

  
  const int grid_m = M / TBM;
  const int grid_n = N / TBN;
  constexpr int SB_TILES = TSM * TSN;
  const int sb_n_count = cdiv(grid_n, TSN);
  const int sb_id = blockIdx.x / SB_TILES;
  const int in_id = blockIdx.x % SB_TILES;
  const int sb_m  = sb_id / sb_n_count;
  const int sb_n  = sb_id % sb_n_count;
  const int bid_m = sb_m * TSM + in_id / TSN;
  const int bid_n = sb_n * TSN + in_id % TSN;
  if (bid_m >= grid_m || bid_n >= grid_n) return;
  const int off_m = bid_m * TBM;
  const int off_n = bid_n * TBN;

  const int warp_id_m = warp_id / TWN_N_;
  const int warp_id_n = warp_id % TWN_N_;
  C += (off_m + warp_id_m * TWARP_M) * N + (off_n + warp_id_n * TWARP_N);

  
  
  extern __shared__ __align__(16) half smem_raw[];
  const int raw  = (int)__cvta_generic_to_shared(smem_raw);
  const int base = (raw + 1023) & ~1023;
  const int A0   = base;                        
  const int B0   = A0 + TSTAGES * A_SZ;         
  const int bar0 = B0 + TSTAGES * B_SZ;         

  if (tid == 0) {
    for (int s = 0; s < TSTAGES; s++) {
      mbar_init(bar0 + s * 8, 1);                 
      mbar_init(bar0 + (TSTAGES + s) * 8, 8);     
    }
  }
  __syncthreads();

  float acc[TWARP_M / MMA_M][TWARP_N / MMA_N][4] = {};
  int A_rmem[TBK / MMA_K][TWARP_M / MMA_M][4];
  int B_rmem[TBK / MMA_K][TWARP_N / MMA_N][2];

  
  const int A_thr = A0
    + ((warp_id_m * TWARP_M) + (lane_id % 16)) * TBK * 2
    + ((lane_id / 16) ^ (((warp_id_m * TWARP_M) + (lane_id % 16)) % 8)) * 16;
  const int B_thr = B0
    + ((warp_id_n * TWARP_N) + (lane_id % 8) + (lane_id / 16) * 8) * TBK * 2
    + (((lane_id % 16) / 8) ^ (((warp_id_n * TWARP_N) + (lane_id % 8) + (lane_id / 16) * 8) % 8)) * 16;

  const int k_iters = K / TBK;

  
  auto issue_tma = [&](int f) {
    const int s = f % TSTAGES;
    mbar_expect_tx(bar0 + s * 8, A_SZ + B_SZ);
    tma_load_2d(A0 + s * A_SZ, &tmapA, f * TBK, off_m, bar0 + s * 8);
    tma_load_2d(B0 + s * B_SZ, &tmapB, f * TBK, off_n, bar0 + s * 8);
  };

  
  if (tid == 0)
    for (int f = 0; f < (TSTAGES - 1 < k_iters ? TSTAGES - 1 : k_iters); f++)
      issue_tma(f);

  int ph_f[TSTAGES] = {};   
  int ph_e[TSTAGES] = {};   

  for (int it = 0; it < k_iters; it++) {
    const int s = it % TSTAGES;
    
    const int f = it + TSTAGES - 1;
    if (tid == 0 && f < k_iters) {
      if (f >= TSTAGES) {                       
        const int se = f % TSTAGES;
        mbar_wait(bar0 + (TSTAGES + se) * 8, ph_e[se]);
        ph_e[se] ^= 1;
      }
      issue_tma(f);
    }
    
    mbar_wait(bar0 + s * 8, ph_f[s]);
    ph_f[s] ^= 1;
    
    for (int k = 0; k < TBK / MMA_K; k++)
      for (int m = 0; m < TWARP_M / MMA_M; m++) {
        int addr = A_thr + s * A_SZ + (m * MMA_M * TBK * 2);
        ldmatrix_x4(A_rmem[k][m], addr ^ (k * 32));
      }
    for (int k = 0; k < TBK / MMA_K; k++)
      for (int n = 0; n < TWARP_N / MMA_N; n += 2) {
        int addr = B_thr + s * B_SZ + (n * MMA_N * TBK * 2);
        int regs[4];
        ldmatrix_x4(regs, addr ^ (k * 32));
        B_rmem[k][n][0] = regs[0]; B_rmem[k][n][1] = regs[1];
        B_rmem[k][n+1][0] = regs[2]; B_rmem[k][n+1][1] = regs[3];
      }
    for (int k = 0; k < TBK / MMA_K; k++)
      for (int m = 0; m < TWARP_M / MMA_M; m++)
        for (int n = 0; n < TWARP_N / MMA_N; n++)
          mma_m16n8k16(A_rmem[k][m], B_rmem[k][n], acc[m][n]);
    
    if (lane_id == 0)
      mbar_arrive(bar0 + (TSTAGES + s) * 8);
  }

  
  for (int m = 0; m < TWARP_M / MMA_M; m++)
    for (int n = 0; n < TWARP_N / MMA_N; n++) {
      const int row = m * MMA_M + (lane_id / 4);
      const int col = n * MMA_N + (lane_id % 4) * 2;
      half *p0 = C + row * N + col;
      half *p8 = C + (row + 8) * N + col;
      *reinterpret_cast<half2*>(p0) = __floats2half2_rn(acc[m][n][0], acc[m][n][1]);
      *reinterpret_cast<half2*>(p8) = __floats2half2_rn(acc[m][n][2], acc[m][n][3]);
    }
}


}
#undef BLOCK_M
#undef BLOCK_N
#undef BLOCK_K
#undef NUM_WARP_M
#undef NUM_WARP_N
#undef NUM_STAGES
#undef SUPER_M
#undef SUPER_N
#undef WARP_SIZE
#undef MMA_M
#undef MMA_N
#undef MMA_K
#undef WARP_M
#undef WARP_N
#undef TB_SIZE
#undef A_SZ
#undef B_SZ
