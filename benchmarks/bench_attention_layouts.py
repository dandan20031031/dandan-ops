"""Packed prefill and direct paged decode; Graph GPU time, repeated inputs.

The paged baseline gathers static physical pages and calls SDPA per request;
gather GPU work is timed. Metadata preparation/update, cache writes, Python,
capture and engine scheduling are excluded. No end-to-end claim is implied.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import dandan_ops as ops
if __package__:
    from .bench_hgemm import measure
else:
    from bench_hgemm import measure


def gathered_sdpa(q,k,v,tables,lengths,page_size):
    outputs=[]
    for row,length in enumerate(lengths):
        used=(length+page_size-1)//page_size
        kb=k[tables[row,:used]].reshape(-1,k.shape[2],k.shape[3])[:length].transpose(0,1)[None]
        vb=v[tables[row,:used]].reshape(-1,v.shape[2],v.shape[3])[:length].transpose(0,1)[None]
        outputs.append(F.scaled_dot_product_attention(q[row:row+1,:,None],kb,vb,enable_gqa=True).squeeze(2))
    return torch.cat(outputs)


@torch.inference_mode()
def run(args):
    torch.manual_seed(31)
    dtype=getattr(torch,args.dtype)
    tol=1.6e-2 if dtype==torch.bfloat16 else 2e-3
    rows=[]
    qlens=[1,17,129];klens=[17,65,257]
    plan=ops.prepare_varlen_attention(qlens,klens,[k-q for q,k in zip(qlens,klens)],device="cuda")
    q=torch.randn(sum(qlens),args.Hq,args.D,device="cuda",dtype=dtype)
    k=torch.randn(sum(klens),args.Hkv,args.D,device="cuda",dtype=dtype);v=torch.randn_like(k)
    paths={"custom_packed":lambda:ops.flash_attention_varlen(q,k,v,plan,backend="cuda"),
           "per_sequence_sdpa":lambda:ops.flash_attention_varlen(q,k,v,plan,backend="torch")}
    expected=paths["per_sequence_sdpa"]();actual=paths["custom_packed"]()
    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)
    rows.append({"operation":"packed_prefill","q_lengths":qlens,"kv_lengths":klens,
                 "timing":measure(paths,args.warmup,args.repeats,args.batch),
                 "max_abs":float((actual.float()-expected.float()).abs().max())})
    lens=[17,257,1025];page_size=16;width=65
    tables=torch.randperm(3*width,device="cpu").reshape(3,width)
    plan=ops.prepare_paged_attention(tables,lens,num_blocks=3*width,block_size=page_size,device="cuda")
    # Static gather indices are uploaded before timing, as is the custom plan.
    device_tables=tables.to("cuda")
    q=torch.randn(3,args.Hq,args.D,device="cuda",dtype=dtype)
    k=torch.randn(3*width,page_size,args.Hkv,args.D,device="cuda",dtype=dtype);v=torch.randn_like(k)
    paths={"custom_direct_paged":lambda:ops.paged_attention(q,k,v,plan,backend="cuda"),
           "gather_then_sdpa":lambda:gathered_sdpa(q,k,v,device_tables,lens,page_size)}
    expected=paths["gather_then_sdpa"]();actual=paths["custom_direct_paged"]()
    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)
    rows.append({"operation":"paged_decode","context_lens":lens,"page_size":page_size,
                 "timing":measure(paths,args.warmup,args.repeats,args.batch),
                 "max_abs":float((actual.float()-expected.float()).abs().max())})
    root=Path(__file__).resolve().parents[1]
    sources=["dandan_ops/attention.py","dandan_ops/csrc/flash_inference.cu",
             "kernels/attention_decode.cuh","kernels/flash_tensorcore_inference.cuh"]
    report={"gpu":torch.cuda.get_device_name(),"sm":torch.cuda.get_device_capability(),
            "torch":str(torch.__version__),"torch_cuda":torch.version.cuda,"config":vars(args),
            "timing_mode":__doc__,"source_sha256":{s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources},
            "results":rows}
    path=Path(args.json);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    for row in rows:
        print(row["operation"],{name:round(result["median_us"],2) for name,result in row["timing"].items()})


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--D",type=int,choices=[64,128,256],default=256)
    p.add_argument("--dtype",choices=["float16","bfloat16"],default="float16")
    p.add_argument("--Hq",type=int,default=16);p.add_argument("--Hkv",type=int,default=4)
    p.add_argument("--warmup",type=int,default=5);p.add_argument("--repeats",type=int,default=21)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--json",default="build/benchmarks/bench_attention_layouts.json")
    args=p.parse_args()
    if min(args.Hq,args.Hkv,args.warmup,args.repeats,args.batch)<1 or args.Hq%args.Hkv:
        p.error("positive dimensions/repeats and Hq divisible by Hkv required")
    run(args)


if __name__=="__main__":
    main()
