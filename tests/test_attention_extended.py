"""Independent math oracles and state/layout contracts for inference attention."""
import pytest
import torch
import dandan_ops as ops

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def oracle(q, k, v, start=0, causal=True, window=None, cap=None, scale=None):
    # Explicit FP32 score/softmax math, independent of SDPA dispatch, packed
    # offsets, physical page addressing and Tensor Core fragment mapping.
    ratio = q.shape[1] // k.shape[1]
    scores = q.float() @ k.float().repeat_interleave(ratio, 1).transpose(-1, -2)
    scores *= q.shape[-1]**-0.5 if scale is None else scale
    if cap is not None:
        scores = cap * torch.tanh(scores / cap)
    if causal:
        pos = torch.arange(q.shape[2], device=q.device)[:, None] + start
        col = torch.arange(k.shape[2], device=q.device)[None, :]
        mask = col <= pos
        if window is not None:
            mask &= col >= pos - window + 1
        scores.masked_fill_(~mask, -float("inf"))
    return (scores.softmax(-1) @ v.float().repeat_interleave(ratio, 1)).to(q.dtype)


def close(actual, expected):
    tol = 1.6e-2 if actual.dtype == torch.bfloat16 else 2e-3
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


@gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("length,klen,start,window,cap,causal", [
    (1, 1, 0, None, None, True),
    (1, 513, 512, None, None, True),
    (1, 513, 256, 3, 4., True),
    (5, 17, 0, 1, None, True),
    (17, 65, 48, 7, 3., True),
    (65, 257, 192, 33, None, True),
    (127, 257, 0, None, 7., True),
    (17, 65, 0, None, 3., False),
])
@torch.inference_mode()
def test_dense_dimensions_types_and_masks(dtype, d, length, klen, start, window, cap, causal):
    torch.manual_seed(d + length)
    q = torch.randn(2, 4, length, d, device="cuda", dtype=dtype)
    k = torch.randn(2, 2, klen, d, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    y = ops.flash_attention(q,k,v,causal,backend="cuda",query_start=start,window_size=window,softcap=cap)
    close(y, oracle(q,k,v,start,causal,window,cap))
    assert ops.flash_attention_execution_info(q,k,v)["implementation"] == "cuda"


@pytest.mark.parametrize("backend", ["auto", "torch"])
def test_reference_window_softcap_and_gradients(backend):
    q = torch.randn(1, 4, 3, 8, requires_grad=True)
    k = torch.randn(1, 2, 9, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    y = ops.flash_attention(q,k,v,True,query_start=6,window_size=2,softcap=3.,backend=backend)
    close(y,oracle(q,k,v,6,True,2,3.))
    y.sum().backward()
    assert all(x.grad is not None and x.grad.isfinite().all() for x in (q,k,v))


@pytest.mark.parametrize("kwargs", [
    {"window_size":0}, {"window_size":-1}, {"window_size":True}, {"window_size":1.5},
    {"softcap":0}, {"softcap":-1}, {"softcap":float("inf")}, {"softcap":1e-100},
])
def test_options_reject_invalid_values(kwargs):
    q = torch.randn(1,2,3,8)
    with pytest.raises(ValueError):
        ops.flash_attention(q,q,q,causal=True,**kwargs)


def test_noncausal_window_is_explicitly_rejected():
    q = torch.randn(1,2,3,8)
    with pytest.raises(ValueError,match="causal"):
        ops.flash_attention(q,q,q,window_size=2)


def varlen_inputs(device, dtype, d):
    lengths, kv_lengths, starts = [0,1,17,65], [0,3,65,129], [0,2,48,64]
    plan = ops.prepare_varlen_attention(lengths,kv_lengths,starts,device=device)
    q = torch.randn(sum(lengths),4,d,device=device,dtype=dtype)
    k = torch.randn(sum(kv_lengths),2,d,device=device,dtype=dtype)
    return q,k,torch.randn_like(k),plan


def varlen_oracle(q,k,v,plan,causal,window,cap):
    qs=ks=0
    outputs=[]
    for qlen,klen,start in zip(plan.q_lengths,plan.kv_lengths,plan._query_starts):
        if qlen:
            qb=q[qs:qs+qlen].transpose(0,1)[None]
            kb=k[ks:ks+klen].transpose(0,1)[None]
            vb=v[ks:ks+klen].transpose(0,1)[None]
            outputs.append(oracle(qb,kb,vb,start,causal,window,cap)[0].transpose(0,1))
        qs+=qlen; ks+=klen
    return torch.cat(outputs) if outputs else q.clone()


@gpu
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@pytest.mark.parametrize("d",[64,128,256])
@pytest.mark.parametrize("causal,window,cap",[(True,None,None),(True,9,4.),(False,None,3.)])
@torch.inference_mode()
def test_varlen_cuda_reads_packed_sequences(dtype,d,causal,window,cap):
    q,k,v,plan=varlen_inputs("cuda",dtype,d)
    y=ops.flash_attention_varlen(q,k,v,plan,causal=causal,window_size=window,softcap=cap,backend="cuda")
    close(y,varlen_oracle(q,k,v,plan,causal,window,cap))
    # Different requests must never read each other's packed KV spans.
    k2=k.clone();v2=v.clone();k2[3:68]+=3;v2[3:68]-=3
    y2=ops.flash_attention_varlen(q,k2,v2,plan,causal=causal,window_size=window,softcap=cap,backend="cuda")
    torch.testing.assert_close(y[:1],y2[:1],atol=0,rtol=0)
    torch.testing.assert_close(y[18:],y2[18:],atol=0,rtol=0)


def test_varlen_cpu_reference_and_contracts():
    q,k,v,plan=varlen_inputs("cpu",torch.float32,8)
    close(ops.flash_attention_varlen(q,k,v,plan,window_size=3,softcap=2.),varlen_oracle(q,k,v,plan,True,3,2.))
    with pytest.raises(ValueError,match="token counts"):
        ops.flash_attention_varlen(q[:-1],k,v,plan)
    with pytest.raises(ValueError,match="equally sized"):
        ops.prepare_varlen_attention([1,2],[3])
    with pytest.raises(ValueError,match="query_starts"):
        ops.prepare_varlen_attention([3],[2])
    with pytest.raises(ValueError,match="integer"):
        ops.prepare_varlen_attention([1.5])
    plan._buffers[0].add_(1)
    with pytest.raises(ValueError,match="modified"):
        ops.flash_attention_varlen(q,k,v,plan)


@pytest.mark.parametrize("device",["cpu",pytest.param("cuda",marks=gpu)])
def test_empty_varlen(device):
    plan=ops.prepare_varlen_attention([0,0],device=device)
    q=torch.empty(0,4,64,device=device,dtype=torch.float16)
    k=torch.empty(0,2,64,device=device,dtype=torch.float16)
    y=ops.flash_attention_varlen(q,k,k,plan)
    assert y.shape==q.shape and y.data_ptr()==q.data_ptr() # both empty


def paged_inputs(device,dtype,d):
    # Read-only prefix page 7 is shared; other pages are shuffled and disjoint.
    tables=[[7,2,9]+[-1]*31,[7,8]+[-1]*32,[-1]*34,[4,5,6]+[-1]*31]
    lengths=[35,19,0,513]
    # Last row spans split boundaries, including a partial physical page.
    tables[3]=list(range(10,43))+[-1]
    plan=ops.prepare_paged_attention(tables,lengths,num_blocks=48,block_size=16,
                                     device=device,max_context=544)
    q=torch.randn(4,4,d,device=device,dtype=dtype)
    k=torch.randn(48,16,2,d,device=device,dtype=dtype)
    return q,k,torch.randn_like(k),plan


def paged_oracle(q,k,v,plan,window=None,cap=None):
    out=[]
    for b,length in enumerate(plan.context_lens):
        if not length:
            out.append(torch.zeros_like(q[b:b+1]));continue
        tokens=[]; values=[]
        for pos in range(length):
            page=plan._host_tables[b][pos//plan._block_size]
            tokens.append(k[page,pos%plan._block_size]);values.append(v[page,pos%plan._block_size])
        kb=torch.stack(tokens).transpose(0,1)[None];vb=torch.stack(values).transpose(0,1)[None]
        out.append(oracle(q[b:b+1,None].transpose(1,2),kb,vb,length-1,True,window,cap).squeeze(2))
    return torch.cat(out)


@gpu
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@pytest.mark.parametrize("d",[64,128,256])
@pytest.mark.parametrize("window,cap",[(None,None),(7,4.),(1,None)])
@torch.inference_mode()
def test_paged_decode_physical_pages_padding_and_isolation(dtype,d,window,cap):
    q,k,v,plan=paged_inputs("cuda",dtype,d)
    q[2].fill_(float("nan"))  # inactive rows must never propagate padding data
    original_k=k.clone();original_v=v.clone()
    y=ops.paged_attention(q,k,v,plan,window_size=window,softcap=cap,backend="cuda")
    close(y,paged_oracle(q,k,v,plan,window,cap))
    assert torch.count_nonzero(y[2])==0
    torch.testing.assert_close(k,original_k,atol=0,rtol=0);torch.testing.assert_close(v,original_v,atol=0,rtol=0)
    # Poison unused storage and another request's pages.
    k[0].fill_(float("nan"));v[0].fill_(float("nan"));k[10:43]+=3;v[10:43]-=3
    k[9,3:].fill_(float("nan"));v[9,3:].fill_(float("nan"))
    y2=ops.paged_attention(q,k,v,plan,window_size=window,softcap=cap,backend="cuda")
    torch.testing.assert_close(y[:3],y2[:3],atol=0,rtol=0)


def test_paged_cpu_update_is_atomic_and_keeps_buffers():
    q,k,v,plan=paged_inputs("cpu",torch.float32,8)
    close(ops.paged_attention(q,k,v,plan),paged_oracle(q,k,v,plan))
    ptrs=tuple(t.data_ptr() for t in plan._buffers)
    old=ops.paged_attention(q,k,v,plan)
    bad=[list(x) for x in plan._host_tables];bad[0][0]=-1
    with pytest.raises(ValueError,match="physical cache"):
        plan.update(bad,plan.context_lens)
    torch.testing.assert_close(ops.paged_attention(q,k,v,plan),old,atol=0,rtol=0)
    tables=[list(x) for x in plan._host_tables];tables[0][0]=1
    plan.update(tables,[1,0,0,17])
    assert tuple(t.data_ptr() for t in plan._buffers)==ptrs
    close(ops.paged_attention(q,k,v,plan,softcap=3.),paged_oracle(q,k,v,plan,cap=3.))


def test_inactive_reference_rows_have_zero_gradient():
    q=torch.full((2,4,8),float("nan"),requires_grad=True)
    cache=torch.zeros(1,16,2,8)
    plan=ops.prepare_paged_attention([[-1],[-1]],[0,0],num_blocks=1,block_size=16)
    y=ops.paged_attention(q,cache,cache,plan)
    assert torch.count_nonzero(y)==0
    y.sum().backward()
    assert torch.count_nonzero(q.grad)==0


@pytest.mark.parametrize("tables,lengths,kwargs",[
    ([[-1]],[1],{}), ([[3]],[1],{}), ([[0]],[17],{}), ([[0]],[-1],{}),
    ([[0],[0,1]],[1,1],{}), ([[0]],[1.5],{}), ([[True]],[1],{}),
    ([[0]],[1],{"max_context":0}), ([[0]],[1],{"max_context":17}),
])
def test_paged_bad_metadata(tables,lengths,kwargs):
    with pytest.raises(ValueError):
        ops.prepare_paged_attention(tables,lengths,num_blocks=3,block_size=16,**kwargs)


@gpu
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@torch.inference_mode()
def test_paged_graph_updates_lengths_tables_and_padding(dtype):
    q,k,v,plan=paged_inputs("cuda",dtype,256)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):ops.paged_attention(q,k,v,plan,backend="cuda")
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        y=ops.paged_attention(q,k,v,plan,backend="cuda")
    # Update on the replay stream; the graph reads buffers at fixed addresses.
    ptrs=tuple(t.data_ptr() for t in plan._buffers)
    tables=[list(x) for x in plan._host_tables];tables[0][0]=1;tables[2][0]=3
    plan.update(tables,[1,0,7,257]);q.add_(0.25)
    graph.replay();torch.cuda.synchronize()
    close(y,paged_oracle(q,k,v,plan))
    assert ptrs==tuple(t.data_ptr() for t in plan._buffers)
    plan.update([[-1]*34]*4,[0]*4)
    graph.replay();torch.cuda.synchronize()
    assert torch.count_nonzero(y)==0


@gpu
@pytest.mark.parametrize("kind",["dense","varlen"])
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@torch.inference_mode()
def test_extended_graph_and_strided_inputs(kind,dtype):
    if kind=="dense":
        q=torch.randn(1,4,256,17,device="cuda",dtype=dtype).transpose(-1,-2)
        k=torch.randn(1,2,256,65,device="cuda",dtype=dtype).transpose(-1,-2);v=torch.randn_like(k)
        fn=lambda:ops.flash_attention(q,k,v,True,query_start=48,window_size=3,softcap=4.,backend="cuda")
        ref=lambda:oracle(q,k,v,48,True,3,4.)
        assert ops.flash_attention_execution_info(q,k,v)["input_copies"]
    else:
        q,k,v,plan=varlen_inputs("cuda",dtype,256)
        q=q.transpose(0,1).contiguous().transpose(0,1)
        fn=lambda:ops.flash_attention_varlen(q,k,v,plan,window_size=7,backend="cuda")
        ref=lambda:varlen_oracle(q,k,v,plan,True,7,None)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):y=fn()
    q.add_(0.25);graph.replay();torch.cuda.synchronize()
    close(y,ref())


@gpu
def test_plan_integrity_and_cuda_grad_contracts():
    q,k,v,plan=paged_inputs("cuda",torch.float16,64)
    q.requires_grad_()
    with pytest.raises(ValueError,match="forward inference"):
        ops.paged_attention(q,k,v,plan,backend="cuda")
    y=ops.paged_attention(q,k,v,plan)
    y.sum().backward();assert q.grad is not None
    plan._buffers[0].add_(1)
    with pytest.raises(ValueError,match="modified"):
        ops.paged_attention(q,k,v,plan)


@gpu
@pytest.mark.parametrize("page_size",[1,7,32,128])
@torch.inference_mode()
def test_page_sizes_zero_capacity_and_unaligned_cache(page_size):
    d=64
    q=torch.randn(1,4,d,device="cuda",dtype=torch.float16)
    k=torch.randn(7*page_size*2*d+1,device="cuda",dtype=q.dtype)[1:].view(7,page_size,2,d)
    v=torch.randn_like(k)
    plan=ops.prepare_paged_attention([[6,4,2]],[2*page_size+1],num_blocks=7,block_size=page_size,device="cuda")
    assert ops.paged_attention_execution_info(q,k,v,plan)["input_copies"]==[1]
    close(ops.paged_attention(q,k,v,plan,backend="cuda"),paged_oracle(q,k,v,plan))
    plan=ops.prepare_paged_attention([[-1,-1,-1]],[0],num_blocks=7,block_size=page_size,device="cuda",max_context=0)
    assert torch.count_nonzero(ops.paged_attention(q,k,v,plan,backend="cuda"))==0


@gpu
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@torch.inference_mode()
def test_qwen35_dense_actual_heads_and_long_context(dtype):
    q=torch.randn(1,16,129,256,device="cuda",dtype=dtype)
    k=torch.randn(1,4,1025,256,device="cuda",dtype=dtype);v=torch.randn_like(k)
    close(ops.flash_attention(q,k,v,True,query_start=896,backend="cuda"),oracle(q,k,v,896))
    q=q[:,:,:1].contiguous()
    k=torch.randn(1,4,4097,256,device="cuda",dtype=dtype);v=torch.randn_like(k)
    close(ops.flash_attention(q,k,v,True,query_start=4096,backend="cuda"),oracle(q,k,v,4096))


@gpu
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@pytest.mark.parametrize("length",[1,17])
@pytest.mark.parametrize("scale",[0.,-0.1,0.17])
@torch.inference_mode()
def test_extended_scales_and_large_logits(dtype,length,scale):
    q=torch.full((1,4,length,256),32.,device="cuda",dtype=dtype)
    k=torch.full((1,2,65,256),32.,device="cuda",dtype=dtype);v=torch.randn_like(k)
    close(ops.flash_attention(q,k,v,True,scale=scale,query_start=65-length,softcap=4.,window_size=3,backend="cuda"),
          oracle(q,k,v,65-length,True,3,4.,scale))


@gpu
def test_varlen_grad_fallback_and_plan_device_validation():
    q,k,v,plan=varlen_inputs("cuda",torch.float16,64)
    q.requires_grad_();k.requires_grad_();v.requires_grad_()
    with pytest.raises(ValueError,match="forward inference"):
        ops.flash_attention_varlen(q,k,v,plan,backend="cuda")
    y=ops.flash_attention_varlen(q,k,v,plan)
    y.float().square().sum().backward()
    assert all(t.grad is not None and t.grad.isfinite().all() for t in (q,k,v))
    cpu=ops.prepare_varlen_attention(plan.q_lengths,plan.kv_lengths,plan._query_starts)
    with pytest.raises(ValueError,match="share device"):
        ops.flash_attention_varlen(q,k,v,cpu)
    with pytest.raises(ValueError,match="CPU integer"):
        ops.prepare_varlen_attention(torch.ones(2,device="cuda",dtype=torch.int32))
