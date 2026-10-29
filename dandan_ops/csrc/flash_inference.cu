#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <climits>
#include <limits>
#include "../../kernels/attention_decode.cuh"

namespace {
void check_inputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                  int qrank, int krank) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "attention expects CUDA tensors");
    TORCH_CHECK((q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) &&
                k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
                "attention expects matching FP16/BF16 tensors");
    TORCH_CHECK(q.dim() == qrank && k.dim() == krank && v.sizes() == k.sizes(), "invalid attention ranks or K/V shape");
    const auto D = q.size(-1);
    TORCH_CHECK((D == 64 || D == 128 || D == 256) && k.size(-1) == D, "attention requires head_dim=64,128,256");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "attention inputs must be contiguous");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "attention tensors must share a device");
    TORCH_CHECK(q.numel() <= INT_MAX && k.numel() <= INT_MAX, "attention tensor index range exceeded");
    TORCH_CHECK(!c10::GradMode::is_enabled() || !(q.requires_grad() || k.requires_grad() || v.requires_grad()),
                "attention CUDA backend is inference-only");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "attention inputs require 16-byte alignment");
    int major = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, q.get_device()));
    TORCH_CHECK(major >= 8, "attention requires SM80+");
}
void check_options(double scale, int64_t window, double softcap) {
    TORCH_CHECK(std::isfinite(scale) && std::abs(scale) <= std::numeric_limits<float>::max(), "invalid attention scale");
    TORCH_CHECK(window == -1 || (window > 0 && window <= INT_MAX), "window_size must be positive or -1");
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0 && softcap <= std::numeric_limits<float>::max(), "invalid softcap");
}
void check_heads(int64_t B, int64_t Hq, int64_t Hkv) {
    TORCH_CHECK(B > 0 && Hq > 0 && Hkv > 0 && Hq % Hkv == 0 && B * Hq <= 65535, "invalid batch/GQA head dimensions");
}
void check_metadata(const at::Tensor& t, const at::Tensor& q, int rank) {
    TORCH_CHECK(t.device() == q.device() && t.scalar_type() == at::kInt && t.dim() == rank && t.is_contiguous(),
                "attention metadata must be contiguous int32 on the query device");
    TORCH_CHECK(t.numel() <= INT_MAX, "attention metadata index range exceeded");
}

template<typename T, int D, bool CAUSAL, bool VARLEN>
void launch_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
    int B, int Hq, int Hkv, int Q, int K, int start, float scale, int window, float softcap,
    const int* cu_q, const int* cu_k, const int* starts, cudaStream_t stream) {
    constexpr int Br = 64, Bc = D == 256 ? 16 : 32;
    constexpr int smem = (4 * Bc + Br) * (D + 8) * sizeof(T);
    auto kernel = dandan_flash::prefill<CAUSAL, Br, Bc, D, T, VARLEN>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    dandan_flash::prefill<CAUSAL, Br, Bc, D, T, VARLEN><<<dim3((Q + Br - 1) / Br, B * Hq), 128, smem, stream>>>(
        reinterpret_cast<const T*>(q.const_data_ptr()), reinterpret_cast<const T*>(k.const_data_ptr()),
        reinterpret_cast<const T*>(v.const_data_ptr()), reinterpret_cast<T*>(out.mutable_data_ptr()),
        Q, K, Hq, Hkv, start, scale, window, softcap, cu_q, cu_k, starts, q.size(0));
}

template<typename T, int D, bool PAGED>
void launch_decode(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
    int B, int Hq, int Hkv, int K, int visible, float scale, int window, float softcap,
    const int* tables, const int* lengths, int width, int block_size, int num_blocks, cudaStream_t stream) {
    const int splits = std::max(1, (visible + 255) / 256);
    TORCH_CHECK(splits <= 65535, "attention split grid exceeds CUDA limit");
    at::Tensor scratch;
    float* sp = nullptr;
    if (splits > 1) {
        scratch = at::empty({B * Hq, splits, D + 2}, q.options().dtype(at::kFloat));
        sp = scratch.data_ptr<float>();
    }
    dandan_flash::decode<T, D, PAGED><<<dim3(B * Hq, splits), 128, 0, stream>>>(
        reinterpret_cast<const T*>(q.const_data_ptr()), reinterpret_cast<const T*>(k.const_data_ptr()),
        reinterpret_cast<const T*>(v.const_data_ptr()), reinterpret_cast<T*>(out.mutable_data_ptr()),
        sp, Hq, Hkv, K, visible, splits, scale, window, softcap, tables, lengths, width, block_size, num_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (splits > 1) dandan_flash::merge_decode<T, D><<<B * Hq, 128, 0, stream>>>(
        sp, reinterpret_cast<T*>(out.mutable_data_ptr()), splits);
}

template<typename T, int D>
void dispatch(int mode, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
    int B, int Hq, int Hkv, int Q, int K, int start, bool causal, float scale, int window, float softcap,
    const int* meta0, const int* meta1, const int* starts, int width, cudaStream_t stream) {
    if (mode == 2) {
        launch_decode<T,D,true>(q,k,v,out,B,Hq,Hkv,K,K,scale,window,softcap,
                                meta0,meta1,width,k.size(1),k.size(0),stream);
    } else if (mode == 0 && Q == 1) {
        launch_decode<T,D,false>(q,k,v,out,B,Hq,Hkv,K,causal ? start+1 : K,scale,window,softcap,
                                 nullptr,nullptr,0,0,0,stream);
    } else if (mode == 1) {
        if (causal) launch_prefill<T,D,true,true>(q,k,v,out,B,Hq,Hkv,Q,K,start,scale,window,softcap,meta0,meta1,starts,stream);
        else launch_prefill<T,D,false,true>(q,k,v,out,B,Hq,Hkv,Q,K,start,scale,window,softcap,meta0,meta1,starts,stream);
    } else {
        if (causal) launch_prefill<T,D,true,false>(q,k,v,out,B,Hq,Hkv,Q,K,start,scale,window,softcap,nullptr,nullptr,nullptr,stream);
        else launch_prefill<T,D,false,false>(q,k,v,out,B,Hq,Hkv,Q,K,start,scale,window,softcap,nullptr,nullptr,nullptr,stream);
    }
}
void launch(int mode, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
    int B, int Hq, int Hkv, int Q, int K, int start, bool causal, double scale, int window, double softcap,
    const int* meta0 = nullptr, const int* meta1 = nullptr, const int* starts = nullptr, int width = 0) {
    const auto stream = c10::cuda::getCurrentCUDAStream().stream();
    #define CALL(T,D) dispatch<T,D>(mode,q,k,v,out,B,Hq,Hkv,Q,K,start,causal,float(scale),window,float(softcap),meta0,meta1,starts,width,stream)
    #define DTYPE(T) if (q.size(-1)==64) { CALL(T,64); } else if (q.size(-1)==128) { CALL(T,128); } else { CALL(T,256); }
    if (q.scalar_type() == at::kHalf) { DTYPE(half); } else { DTYPE(__nv_bfloat16); }
    #undef DTYPE
    #undef CALL
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor flash_inference(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    int64_t query_start, bool causal, double scale, int64_t window, double softcap) {
    c10::cuda::CUDAGuard guard(q.device());
    check_inputs(q,k,v,4,4); check_options(scale,window,softcap);
    TORCH_CHECK(q.numel() > 0 && k.numel() > 0, "attention dimensions must be non-empty");
    check_heads(q.size(0),q.size(1),k.size(1));
    TORCH_CHECK(q.size(0) == k.size(0), "attention batches must match");
    TORCH_CHECK(query_start >= 0 && query_start <= k.size(2)-q.size(2), "invalid query_start");
    TORCH_CHECK(causal || window == -1, "sliding window requires causal attention");
    auto out = at::empty_like(q);
    launch(0,q,k,v,out,q.size(0),q.size(1),k.size(1),q.size(2),k.size(2),query_start,causal,scale,window,softcap);
    return out;
}
at::Tensor flash_varlen(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& cu_q, const at::Tensor& cu_k, const at::Tensor& starts,
    int64_t max_q, bool causal, double scale, int64_t window, double softcap) {
    c10::cuda::CUDAGuard guard(q.device());
    check_inputs(q,k,v,3,3); check_options(scale,window,softcap);
    check_metadata(cu_q,q,1); check_metadata(cu_k,q,1); check_metadata(starts,q,1);
    const auto B = starts.numel();
    check_heads(B,q.size(1),k.size(1));
    TORCH_CHECK(cu_q.numel()==B+1 && cu_k.numel()==B+1 && max_q>=0 && max_q<=q.size(0), "invalid varlen metadata shapes");
    TORCH_CHECK(causal || window == -1, "sliding window requires causal attention");
    auto out = at::empty_like(q);
    if (q.numel()) {
        TORCH_CHECK(max_q>0 && k.numel()>0, "nonempty queries require nonempty keys and max_q>0");
        launch(1,q,k,v,out,B,q.size(1),k.size(1),max_q,k.size(0),0,causal,scale,window,softcap,
               cu_q.data_ptr<int>(),cu_k.data_ptr<int>(),starts.data_ptr<int>());
    }
    return out;
}
at::Tensor paged_decode(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& tables, const at::Tensor& lengths, int64_t max_context,
    double scale, int64_t window, double softcap) {
    c10::cuda::CUDAGuard guard(q.device());
    check_inputs(q,k,v,3,4); check_options(scale,window,softcap);
    check_metadata(tables,q,2); check_metadata(lengths,q,1);
    check_heads(q.size(0),q.size(1),k.size(2));
    TORCH_CHECK(k.size(0)>0 && k.size(1)>0 && tables.size(0)==q.size(0) && lengths.numel()==q.size(0), "invalid paged cache/metadata shapes");
    TORCH_CHECK(max_context>=0 && max_context<=INT_MAX-255 && max_context<=tables.size(1)*k.size(1), "invalid max_context");
    auto out = at::empty_like(q);
    launch(2,q,k,v,out,q.size(0),q.size(1),k.size(2),1,max_context,0,true,scale,window,softcap,
           tables.data_ptr<int>(),lengths.data_ptr<int>(),nullptr,tables.size(1));
    return out;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(dandan_ops, m) {
    m.def("attention_contract_version() -> int", []() -> int64_t { return 1; });
    m.def("flash_inference(Tensor q, Tensor k, Tensor v, int query_start, bool causal, float scale=0.08838834764831845, int window_size=-1, float softcap=0.) -> Tensor");
    m.def("flash_varlen(Tensor q, Tensor k, Tensor v, Tensor cu_q, Tensor cu_k, Tensor starts, int max_q, bool causal, float scale, int window_size=-1, float softcap=0.) -> Tensor");
    m.def("paged_decode(Tensor q, Tensor k, Tensor v, Tensor block_tables, Tensor context_lens, int max_context, float scale, int window_size=-1, float softcap=0.) -> Tensor");
}
TORCH_LIBRARY_IMPL(dandan_ops, CUDA, m) {
    m.impl("flash_inference", &flash_inference);
    m.impl("flash_varlen", &flash_varlen);
    m.impl("paged_decode", &paged_decode);
}
