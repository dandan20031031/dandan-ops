"""Optional engine integration; engine files are imported read-only."""
import copy
import os
from pathlib import Path
import sys

import pytest
import torch
import dandan_ops as ops
from dandan_ops.integrations.myvllm import convert_signed_int4, EngineQuantLinear, install_int4_projections

engine = Path(os.environ.get("MYVLLM_ROOT", Path(__file__).resolve().parents[2] / "myvllm"))
if not (engine / "mini_engine").is_dir():
    pytest.skip("optional myvllm checkout not found", allow_module_level=True)
sys.path.insert(0, str(engine))
from mini_engine.config import QuantConfig
from mini_engine.model import NativeQwen3, Qwen3Config
from mini_engine.quantization.backends import Int4WeightOnlyBackend


@pytest.mark.parametrize("k", [1,7,8,63,64,65,127,128,129,255,256,257,1024])
@pytest.mark.parametrize("kind", ["random","zero","constant"])
def test_bridge_preserves_exact_fp32_weight(k, kind):
    torch.manual_seed(k)
    w = torch.randn(7,k)
    if kind == "zero": w.zero_()
    if kind == "constant": w.fill_(-0.7)
    source = Int4WeightOnlyBackend(128).pack(w)
    p,s,z = convert_signed_int4(source.weight, source.scale, k)
    torch.testing.assert_close(ops.dequantize_int4(p,s,z,k),
        Int4WeightOnlyBackend(128).dequantize(source), rtol=0, atol=0)


def test_bridge_all_signed_nibbles_and_bad_metadata():
    nibble = torch.arange(16,dtype=torch.uint8).repeat(8).view(1,-1)
    raw = nibble[:,::2] | (nibble[:,1::2] << 4)
    p,s,z = convert_signed_int4(raw,torch.ones(1,1),128)
    signed = torch.where(nibble >= 8, nibble.short()-16, nibble.short()).float()
    torch.testing.assert_close(ops.dequantize_int4(p,s,z,128),signed,atol=0,rtol=0)
    with pytest.raises(ValueError,match="group"):
        convert_signed_int4(raw,torch.ones(1,1),128,64)
    with pytest.raises(ValueError,match="shape"):
        convert_signed_int4(raw,torch.ones(1,2),128)
    with pytest.raises(ValueError,match="positive"):
        convert_signed_int4(raw,torch.zeros(1,1),128)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_engine_module_state_move_shape_and_strict_dispatch():
    config = QuantConfig("int4",activation_dtype="fp16",group_size=128)
    linear = torch.nn.Linear(129,67,bias=True,device="cuda",dtype=torch.float16)
    mod = EngineQuantLinear(linear,config)
    x = torch.randn(2,3,129,device="cuda",dtype=torch.float16)
    with torch.inference_mode():
        expected = (x.float() @ mod.dequantize().half().float().t()).half() + linear.bias
        torch.testing.assert_close(mod(x),expected,atol=2e-3,rtol=2e-3)
        assert mod(torch.empty(0,129,device="cuda",dtype=torch.float16)).shape == (0,67)
    s = mod.op.scales.clone()
    mod.cpu().half().cuda().bfloat16()
    assert mod.op.scales.dtype == torch.float32 and torch.equal(mod.op.scales,s)
    with pytest.raises(ValueError,match="CUDA FP16"):
        mod(x.bfloat16())
    mod.half()
    clone = EngineQuantLinear(linear,config)
    clone.load_state_dict(mod.state_dict())
    with torch.inference_mode():
        torch.testing.assert_close(mod(x),clone(x),rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_tiny_qwen_cached_decode_batch_and_full_recompute():
    torch.manual_seed(17)
    cfg = Qwen3Config(53,32,64,2,4,2,16,max_position_embeddings=128)
    model = NativeQwen3(cfg,"math").cuda().half().eval()
    embedding = model.model.embed_tokens.weight
    norm = model.model.norm.weight
    quant = QuantConfig("int4",activation_dtype="fp16",group_size=128)
    report = install_int4_projections(model,quant)
    assert len(report["replaced_layers"]) == 14 and report["accelerated_quant_gemm"]
    assert model.lm_head.weight is embedding and model.model.norm.weight is norm
    assert not any(".op.weight" in key for key in model.state_dict())
    a_cache, b_cache, full_cache = (model.create_cache(16,4) for _ in range(3))
    blocks = [5,1,6,0,8,9,10,11]
    ids = torch.tensor([1,2,3,4,5],device="cuda")
    with torch.inference_mode():
        a = model(ids,a_cache,blocks)
        for _ in range(3):
            token = a[-1].argmax().view(1)
            position = ids.numel()
            ids = torch.cat([ids,token])
            a = model(token,a_cache,blocks,position)
            full = model(ids,full_cache,blocks)[-1:]
            # Same FP16 operands; downstream norm/attention reductions can
            # amplify low-bit differences, so compare model logits separately.
            torch.testing.assert_close(a,full,atol=2e-2,rtol=2e-2)
        state = copy.deepcopy(model.state_dict())
        clone = NativeQwen3(cfg,"math").cuda().half().eval()
        install_int4_projections(clone,quant)
        clone.load_state_dict(state)
        fresh = clone.create_cache(16,4)
        torch.testing.assert_close(clone(ids,fresh,blocks),model(ids,b_cache,blocks),atol=0,rtol=0)
    with pytest.raises(ValueError,match="already"):
        install_int4_projections(model,quant)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_model_quantization_failure_does_not_partially_mutate():
    model = NativeQwen3(Qwen3Config(17,16,32,1,2,1,8),"math").cuda().half()
    before = dict(model.named_modules())
    # A late projection contains invalid values; earlier prepared replacements
    # must not have been attached to the live model.
    with torch.no_grad(): model.model.layers[0].mlp.down_proj.weight[0,0] = float("nan")
    with pytest.raises(ValueError,match="finite"):
        install_int4_projections(model,QuantConfig("int4",activation_dtype="fp16"))
    assert all(model.get_submodule(name) is module for name,module in before.items() if name)
    assert model.quantization_report is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_elementwise_library_in_tiny_qwen_prefill_and_cached_decode(monkeypatch, dtype):
    """Replace only this process's small ops; engine source and cache stay native."""
    import mini_engine.model.native_qwen3 as module
    torch.manual_seed(27)
    cfg = Qwen3Config(53, 32, 64, 2, 4, 2, 16, max_position_embeddings=128)
    model = NativeQwen3(cfg, "math").cuda().to(dtype).eval()
    blocks = [5, 1, 6, 0, 8, 9, 10, 11]
    prompt = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    reference_cache = model.create_cache(16, 4)
    reference = [model(prompt, reference_cache, blocks).clone()]
    tokens = []
    for position in range(5, 8):
        token = reference[-1][-1].argmax().view(1)
        tokens.append(token.clone())
        reference.append(model(token, reference_cache, blocks, position).clone())
    def forced(fn):
        def call(*args, backend="torch", **kwargs):
            return fn(*args, backend="cuda", **kwargs)
        return call
    for name in ("rms_norm", "residual_rms_norm", "silu_and_mul"):
        monkeypatch.setattr(module, name, forced(getattr(ops, name)))
    cache = model.create_cache(16, 4)
    actual = [model(prompt, cache, blocks).clone()]
    for position, token in zip(range(5, 8), tokens):
        assert torch.equal(actual[-1][-1].argmax().view(1), token)
        actual.append(model(token, cache, blocks, position).clone())
    tolerance = 3e-2 if dtype == torch.bfloat16 else 3e-3
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
        assert torch.equal(a[-1].argmax(), b[-1].argmax())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("quantized", [False, True])
@torch.inference_mode()
def test_d128_flash_prefill_with_engine_paged_decode(monkeypatch, quantized):
    """Exercise the actual model layout and cached continuation without editing
    the concurrently developed engine. Patch only this test's prefill call;
    decode still uses the engine's existing paged reference implementation.
    """
    import mini_engine.model.native_qwen3 as native

    torch.manual_seed(71)
    cfg = Qwen3Config(53, 64, 128, 2, 4, 2, 128, max_position_embeddings=64)
    model = NativeQwen3(cfg, "math").cuda().half().eval()
    if quantized:
        install_int4_projections(model, QuantConfig("int4", activation_dtype="fp16", group_size=128))
    blocks = [5, 1, 6, 0]
    prompt = torch.arange(17, device="cuda", dtype=torch.long) % cfg.vocab_size
    ref_cache = model.create_cache(8, 8)
    expected = [model(prompt, ref_cache, blocks)]
    tokens = []
    for step in range(3):
        token = expected[-1][-1].argmax().view(1)
        tokens.append(token)
        expected.append(model(token, ref_cache, blocks, prompt.numel() + step))

    calls = []
    def flash_prefill(q, k, v, backend="auto", *, query_start=0):
        assert q.shape[1] == 4 and k.shape[1] == 2  # Keep native GQA heads.
        calls.append(q.shape[2])
        return ops.flash_attention(q, k, v, causal=True, backend="cuda", query_start=query_start)

    monkeypatch.setattr(native, "prefill_attention", flash_prefill)
    cache = model.create_cache(8, 8)
    actual = model(prompt, cache, blocks)
    torch.testing.assert_close(actual, expected[0], atol=2e-2, rtol=2e-2)
    assert actual[-1].argmax() == expected[0][-1].argmax()
    for step, token in enumerate(tokens):
        actual = model(token, cache, blocks, prompt.numel() + step)
        torch.testing.assert_close(actual, expected[step+1], atol=2e-2, rtol=2e-2)
        assert actual[-1].argmax() == expected[step+1][-1].argmax()
    assert calls == [17] * cfg.num_hidden_layers
