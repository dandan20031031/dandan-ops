 #include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>   
#include <cuda_fp16.h>       
#include <cstdint>          
#include <math.h>

// FlashAttention v2.1 Tensor Core   mma.sync + ldmatrix 

// ---- ldmatrix.x4 4   8x8 tile       4   b32   b32 = 2 half ----
__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t saddr)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(saddr));
}

// ---- ldmatrix.x4.trans            V   B operand   ----
__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t saddr)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(saddr));
}

// ---- mma.m16n8k16 D = A B + C fp16    fp32    ----
__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,   // C/D     4   fp32
    const uint32_t a0, const uint32_t a1,         // A     4   b32
    const uint32_t a2, const uint32_t a3,
    const uint32_t b0, const uint32_t b1)         // B     2   b32
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
}

// shared      ldmatrix    32    
__device__ __forceinline__ uint32_t smem_u32addr(const void* p)
{
    return (uint32_t)__cvta_generic_to_shared(p);
}

//    fp32       half2   b32    = lo fragment     2t     
__device__ __forceinline__ uint32_t f2h2(float lo, float hi)
{
    __half2 h2 = __floats2half2_rn(lo, hi);
    return reinterpret_cast<uint32_t&>(h2);
}

// ------------------------------------------------------------
// K/V       half   +8 padding    
//    72 half = 144B 16B          16B      
//     zfill    0   =0   S        -inf     
// ------------------------------------------------------------
template<int Bc, int HEAD_DIM>
__device__ __forceinline__ void kv_async_copy_v21(
    const __half* k, const __half* v,
    __half* Ks, __half* Vs,
    size_t bh, int kv_start, int N, int buf)
{
    int tx = threadIdx.x;
    constexpr int DP = HEAD_DIM + 8;      // padded    half   
    constexpr int PKG = HEAD_DIM / 8;     //    16B    8 half/  

    const uint4* kg = reinterpret_cast<const uint4*>(
        k + bh * N * HEAD_DIM + (size_t)kv_start * HEAD_DIM);
    const uint4* vg = reinterpret_cast<const uint4*>(
        v + bh * N * HEAD_DIM + (size_t)kv_start * HEAD_DIM);
    __half* kbase = Ks + (size_t)buf * Bc * DP;
    __half* vbase = Vs + (size_t)buf * Bc * DP;

    constexpr int NVEC = Bc * PKG;
    for (int i = tx; i < NVEC; i += blockDim.x) {
        int  row = i / PKG;
        int  p   = i % PKG;
        bool ok  = (kv_start + row) < N;
        uint4* kd = reinterpret_cast<uint4*>(kbase + row * DP + p * 8);
        uint4* vd = reinterpret_cast<uint4*>(vbase + row * DP + p * 8);
        __pipeline_memcpy_async(kd, &kg[i], 16, ok ? 0 : 16);
        __pipeline_memcpy_async(vd, &vg[i], 16, ok ? 0 : 16);
    }
}

// ------------------------------------------------------------
// v2.1 kernel
// Br = 128 4 warp   warp 32   = 2   m16    warp      
// Bc = 32 S   4   n8 PV   A fragment k16 = 2   n8     
// ------------------------------------------------------------
template<bool CAUSAL, int Br, int Bc, int HEAD_DIM>
__global__ void __launch_bounds__(Br) flashattention_v2_1(
    const __half* q,   // [B, H, N, D]
    const __half* k,   // [B, H, N, D]
    const __half* v,   // [B, H, N, D]
    __half* o,         // [B, H, N, D]     
    int N)
{
    static_assert(Br % 128 == 0,  "v2.1   4 warp 32     Br    128    ");
    static_assert(Bc % 16 == 0,   "PV   A fragment k16      Bc   16    ");
    static_assert(HEAD_DIM % 16 == 0, "ldmatrix       D   16    ");

    constexpr int D  = HEAD_DIM;
    constexpr int DP = D + 8;        // shared    +8 padding half   
    constexpr int KM = D / 16;       // QK^T   k16      D 
    constexpr int NM = Bc / 8;       // QK^T   n8      kv 
    constexpr int KP = Bc / 16;      // PV   k16      kv 
    constexpr int ND = D / 8;        // PV   n8      D = Oacc   tile  
    constexpr int M2 = 2;            //   warp   m16    

    int qb   = blockIdx.x;           // Q    
    int bh   = blockIdx.y;           // batch*head
    int tx   = threadIdx.x;
    int warp = tx >> 5;              // warp    [0,4)
    int lane = tx & 31;
    int g    = lane >> 2;            // fragment    [0,8)      = g   g+8
    int t    = lane & 3;             // fragment    [0,4)      = 2t   2t+1
    int tl   = lane >> 3;            // ldmatrix   tile    [0,4)
    int tr   = lane & 7;             // ldmatrix   tile     [0,8)

    int q_start = qb * Br;

    // ---- causal        v2.0    ----
    const int kv_limit = CAUSAL ? min(q_start + Br, N) : N;
    const int Tc = (kv_limit + Bc - 1) / Bc;

    // softmax scale   log2(e)       softmax    exp2f
    const float sl2e = (1.0f / sqrtf((float)HEAD_DIM)) * 1.44269504f;

    const size_t base = (size_t)bh * N * D;

    // ----    shared Ks[2][Bc][DP] | Vs[2][Bc][DP] | Qs[Br][DP] ----
    //      _h    extern __shared__   nvcc        
    //      v1/v2   float smem[]    
    extern __shared__ __half smem_h[];
    __half* Ks = smem_h;                                   // [2][Bc][DP]
    __half* Vs = Ks + 2 * Bc * DP;                         // [2][Bc][DP]
    __half* Qs = Vs + 2 * Bc * DP;                         // [Br][DP]

    // ---- Q     shared                ----
    for (int i = tx; i < Br * (D / 8); i += blockDim.x) {
        int r = i / (D / 8), p = i % (D / 8);
        uint4* dst = reinterpret_cast<uint4*>(Qs + r * DP + p * 8);
        if (q_start + r < N)
            *dst = *(const uint4*)(q + base + (size_t)(q_start + r) * D + p * 8);
        else
            *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();

    // ---- Q fragment       ldmatrix      32   b32 ----
    // A fragment   4   tile tile0=( 0-8, 0-8) tile1=( 8-16, 0-8)
    //                         tile2=( 0-8, 8-16) tile3=( 8-16, 8-16)
    // lane l   tile l/8    l%8      tile       +8 tile       +8
    uint32_t qf[M2][KM][4];
    #pragma unroll
    for (int mi = 0; mi < M2; mi++)
        #pragma unroll
        for (int kk = 0; kk < KM; kk++) {
            int row = warp * 32 + mi * 16 + ((tl & 1) ? tr + 8 : tr);
            int col = kk * 16 + ((tl & 2) ? 8 : 0);
            ldmatrix_x4(qf[mi][kk][0], qf[mi][kk][1],
                        qf[mi][kk][2], qf[mi][kk][3],
                        smem_u32addr(&Qs[row * DP + col]));
        }

    // ----       m      running max / l        ----
    float m_reg[M2][2], l_par[M2][2];        // [m16  ][   0=  g, 1=  g+8]
    #pragma unroll
    for (int mi = 0; mi < M2; mi++) {
        m_reg[mi][0] = m_reg[mi][1] = -INFINITY;
        l_par[mi][0] = l_par[mi][1] = 0.0f;
    }

    // ---- Oacc mma       2 m16   8 n8   4 fp32 = 64   ----
    float acc_o[M2][ND][4];
    #pragma unroll
    for (int mi = 0; mi < M2; mi++)
        #pragma unroll
        for (int nn = 0; nn < ND; nn++) {
            acc_o[mi][nn][0] = 0.0f; acc_o[mi][nn][1] = 0.0f;
            acc_o[mi][nn][2] = 0.0f; acc_o[mi][nn][3] = 0.0f;
        }

    // ---- prologue     0   ----
    kv_async_copy_v21<Bc, D>(k, v, Ks, Vs, bh, 0, N, 0);
    __pipeline_commit();

    // =================            K/V   =================
    for (int j = 0; j < Tc; j++) {
        int kv_start = j * Bc;
        int cur = j & 1;

        if (j + 1 < Tc) {
            kv_async_copy_v21<Bc, D>(k, v, Ks, Vs, bh, kv_start + Bc, N, cur ^ 1);
            __pipeline_commit();
        }
        __pipeline_wait_prior(j + 1 < Tc ? 1 : 0);
        __syncthreads();

        __half* Kj = Ks + (size_t)cur * Bc * DP;
        __half* Vj = Vs + (size_t)cur * Bc * DP;

        // causal     mask      v2.0      
        const bool diag_block = CAUSAL && (kv_start + Bc - 1 > q_start);
        const bool kv_full    = kv_start + Bc <= N;   //           

        // ---- 1. K fragment 8   ldmatrix.x4 normal   ----
        //    x4   2   n8     b0/b1 = 4   b32
        //    tile0=(n 0-8, k 0-8) tile1=(n 0-8, k 8-16)
        //       tile2=(n 8-16, k 0-8) tile3=(n 8-16, k 8-16)
        //  tile      k+8 tile      n+8      Q   /        
        uint32_t kf[KM][NM][2];
        #pragma unroll
        for (int kk = 0; kk < KM; kk++)
            #pragma unroll
            for (int p = 0; p < NM / 2; p++) {
                int nrow = p * 16 + ((tl >> 1) ? tr + 8 : tr);
                int kcol = kk * 16 + ((tl & 1) ? 8 : 0);
                ldmatrix_x4(kf[kk][2 * p][0],     kf[kk][2 * p][1],
                            kf[kk][2 * p + 1][0], kf[kk][2 * p + 1][1],
                            smem_u32addr(&Kj[nrow * DP + kcol]));
            }

        // ---- 2. S = Q @ K^T 32   mma 2 m16   4 n8   4 k16 ----
        float s[M2][NM][4];
        #pragma unroll
        for (int mi = 0; mi < M2; mi++)
            #pragma unroll
            for (int ni = 0; ni < NM; ni++) {
                s[mi][ni][0] = s[mi][ni][1] = 0.0f;
                s[mi][ni][2] = s[mi][ni][3] = 0.0f;
                #pragma unroll
                for (int kk = 0; kk < KM; kk++)
                    mma_m16n8k16(s[mi][ni][0], s[mi][ni][1],
                                 s[mi][ni][2], s[mi][ni][3],
                                 qf[mi][kk][0], qf[mi][kk][1],
                                 qf[mi][kk][2], qf[mi][kk][3],
                                 kf[kk][ni][0], kf[kk][ni][1]);
            }

        // ---- 3. mask    kv + causal     +     scale log2e ----
        // c0/c1    g lo  c2/c3    g+8 hi    = kv_start + ni*8 + 2t (+1)
        if (!kv_full || diag_block) {
            #pragma unroll
            for (int mi = 0; mi < M2; mi++) {
                int row_lo = q_start + warp * 32 + mi * 16 + g;
                int row_hi = row_lo + 8;
                #pragma unroll
                for (int ni = 0; ni < NM; ni++) {
                    int col_lo = kv_start + ni * 8 + 2 * t;
                    int col_hi = col_lo + 1;
                    if (col_lo >= N || (diag_block && col_lo > row_lo))
                        s[mi][ni][0] = -INFINITY;
                    if (col_hi >= N || (diag_block && col_hi > row_lo))
                        s[mi][ni][1] = -INFINITY;
                    if (col_lo >= N || (diag_block && col_lo > row_hi))
                        s[mi][ni][2] = -INFINITY;
                    if (col_hi >= N || (diag_block && col_hi > row_hi))
                        s[mi][ni][3] = -INFINITY;
                }
            }
        }
        #pragma unroll
        for (int mi = 0; mi < M2; mi++)
            #pragma unroll
            for (int ni = 0; ni < NM; ni++) {
                s[mi][ni][0] *= sl2e; s[mi][ni][1] *= sl2e;
                s[mi][ni][2] *= sl2e; s[mi][ni][3] *= sl2e;
            }

        // ---- 4. rowmax     8   + quad allreduce xor 2   xor 1 ----
        //   g   16       lane 4g..4g+3       
        float m_new[M2][2];
        #pragma unroll
        for (int mi = 0; mi < M2; mi++)
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                float m = -INFINITY;
                #pragma unroll
                for (int ni = 0; ni < NM; ni++) {
                    m = fmaxf(m, s[mi][ni][2 * h]);
                    m = fmaxf(m, s[mi][ni][2 * h + 1]);
                }
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                m_new[mi][h] = fmaxf(m_reg[mi][h], m);
            }

        // ---- 5. m_d    + rescale    ----
        // m_new == -inf     mask   exp2f(-inf-(-inf)) = NaN   0   
        float m_d[M2][2], so[M2][2];
        #pragma unroll
        for (int mi = 0; mi < M2; mi++)
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                m_d[mi][h] = (m_new[mi][h] == -INFINITY) ? 0.0f : m_new[mi][h];
                so[mi][h]  = exp2f(m_reg[mi][h] - m_d[mi][h]);   //         
            }

        // ---- 6. P = exp2f(S - m_d)      s      A fragment ----
        //      a0= 2kk   c0/c1   g  a1= 2kk   c2/c3   g+8  
        //           a2/a3 =   2kk+1                     
        #pragma unroll
        for (int mi = 0; mi < M2; mi++)
            #pragma unroll
            for (int ni = 0; ni < NM; ni++) {
                s[mi][ni][0] = exp2f(s[mi][ni][0] - m_d[mi][0]);
                s[mi][ni][1] = exp2f(s[mi][ni][1] - m_d[mi][0]);
                s[mi][ni][2] = exp2f(s[mi][ni][2] - m_d[mi][1]);
                s[mi][ni][3] = exp2f(s[mi][ni][3] - m_d[mi][1]);
            }
        // l               shuffle epilogue            
        //    l            acc_o *= so           
        #pragma unroll
        for (int mi = 0; mi < M2; mi++)
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                l_par[mi][h] *= so[mi][h];    //                acc_o *= so    
                #pragma unroll
                for (int ni = 0; ni < NM; ni++)
                    l_par[mi][h] += s[mi][ni][2 * h] + s[mi][ni][2 * h + 1];
            }

        uint32_t pf[M2][KP][4];
        #pragma unroll
        for (int mi = 0; mi < M2; mi++)
            #pragma unroll
            for (int kk = 0; kk < KP; kk++) {
                int ni = 2 * kk;
                pf[mi][kk][0] = f2h2(s[mi][ni][0],     s[mi][ni][1]);
                pf[mi][kk][1] = f2h2(s[mi][ni][2],     s[mi][ni][3]);
                pf[mi][kk][2] = f2h2(s[mi][ni + 1][0], s[mi][ni + 1][1]);
                pf[mi][kk][3] = f2h2(s[mi][ni + 1][2], s[mi][ni + 1][3]);
            }

        // ---- 7. Oacc rescale        v2.0   shared       ----
        #pragma unroll
        for (int mi = 0; mi < M2; mi++)
            #pragma unroll
            for (int nn = 0; nn < ND; nn++) {
                acc_o[mi][nn][0] *= so[mi][0];
                acc_o[mi][nn][1] *= so[mi][0];
                acc_o[mi][nn][2] *= so[mi][1];
                acc_o[mi][nn][3] *= so[mi][1];
            }

        // ---- 8. V fragment 8   ldmatrix.x4.trans ----
        // tile0=(k 0-8, n 0-8) tile1=(k 8-16, n 0-8)
        // tile2=(k 0-8, n 8-16) tile3=(k 8-16, n 8-16)
        //  tile      k+8 tile      n+8 trans    n       k 
        uint32_t vf[KP][ND][2];
        #pragma unroll
        for (int kk = 0; kk < KP; kk++)
            #pragma unroll
            for (int p = 0; p < ND / 2; p++) {
                int krow = kk * 16 + ((tl & 1) ? tr + 8 : tr);
                int dcol = p * 16 + ((tl >> 1) ? 8 : 0);
                ldmatrix_x4_trans(vf[kk][2 * p][0],     vf[kk][2 * p][1],
                                  vf[kk][2 * p + 1][0], vf[kk][2 * p + 1][1],
                                  smem_u32addr(&Vj[krow * DP + dcol]));
            }

        // ---- 9. PV 32   mma A = P fragment           ----
        #pragma unroll
        for (int mi = 0; mi < M2; mi++)
            #pragma unroll
            for (int nn = 0; nn < ND; nn++)
                #pragma unroll
                for (int kk = 0; kk < KP; kk++)
                    mma_m16n8k16(acc_o[mi][nn][0], acc_o[mi][nn][1],
                                 acc_o[mi][nn][2], acc_o[mi][nn][3],
                                 pf[mi][kk][0], pf[mi][kk][1],
                                 pf[mi][kk][2], pf[mi][kk][3],
                                 vf[kk][nn][0], vf[kk][nn][1]);

        //      so     m_reg          
        #pragma unroll
        for (int mi = 0; mi < M2; mi++) {
            m_reg[mi][0] = m_new[mi][0];
            m_reg[mi][1] = m_new[mi][1];
        }

        //             cur buffer          
        __syncthreads();
    }

    // ---- epilogue l             +     +    ----
    #pragma unroll
    for (int mi = 0; mi < M2; mi++)
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            float l = l_par[mi][h];
            l += __shfl_xor_sync(0xffffffffu, l, 2);
            l += __shfl_xor_sync(0xffffffffu, l, 1);
            l_par[mi][h] = l;
        }

    #pragma unroll
    for (int mi = 0; mi < M2; mi++)
        #pragma unroll
        for (int nn = 0; nn < ND; nn++) {
            int row_lo = q_start + warp * 32 + mi * 16 + g;   // c0/c1   
            float inv0 = 1.0f / l_par[mi][0];
            float inv1 = 1.0f / l_par[mi][1];
            if (row_lo < N) {
                size_t off = base + (size_t)row_lo * D + nn * 8 + 2 * t;
                o[off]     = __float2half(acc_o[mi][nn][0] * inv0);
                o[off + 1] = __float2half(acc_o[mi][nn][1] * inv0);
            }
            if (row_lo + 8 < N) {
                size_t off = base + (size_t)(row_lo + 8) * D + nn * 8 + 2 * t;
                o[off]     = __float2half(acc_o[mi][nn][2] * inv1);
                o[off + 1] = __float2half(acc_o[mi][nn][3] * inv1);
            }
        }
}
