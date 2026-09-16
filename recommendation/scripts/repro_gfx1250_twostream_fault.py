#!/usr/bin/env python3
"""Same-process, two-stream contention test -- docs/mi450_b0/mi450_b0.md section 3.6.

Produces the `hard fault in _hstu_attn_bwd` row of the 3.6 table: a victim HSTU
layer on the default stream, and an aggressor HSTU fwd+bwd on a second
torch.cuda.Stream() in a daemon thread.

It answers whether the contention needs two PROCESSES or merely two concurrent
STREAMS. It needs only two streams -- so the corruption is reachable inside a
single process, and stream serialisation is an available workaround.

    AGGRESSOR=1 (default)  aggressor thread on, expect an aperture violation
    AGGRESSOR=0            control, was 20/20 clean
    REPEATS=20             victim calls to compare

Run with AMDGCN_USE_BUFFER_OPS=0 and PYTORCH_CUDA_ALLOC_CONF= empty, on an IDLE GPU.
"""
import hashlib, os, sys, torch, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: import fbgemm_gpu  # noqa
except ImportError: pass
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.hstu_compute import hstu_compute_output, hstu_preprocess_and_attention
D,NH,AD,HD,MSL,EPS = 512,4,128,128,4096,1e-6
dev=torch.device("cuda"); torch.manual_seed(0)
REPEATS=int(os.environ.get("REPEATS","20")); AGG=os.environ.get("AGGRESSOR","1")=="1"
def md5(t): return hashlib.md5(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:12]
def mkw(g,n=D):
    return {"in_w":torch.ones(n,device=dev),"in_b":torch.zeros(n,device=dev),
        "uvqk_w":torch.randn(n,2*NH*(HD+AD),device=dev,generator=g)*0.02,
        "uvqk_b":torch.zeros(2*NH*(HD+AD),device=dev),
        "out_w":torch.ones(NH*HD,device=dev),"out_b":torch.zeros(NH*HD,device=dev),
        "out_proj":torch.randn(NH*HD+n,n,device=dev,generator=g)*0.02}
def mkdata(g,N,B):
    wt=torch.rand(B,device=dev,generator=g)+0.1
    L=(wt/wt.sum()*N).long().clamp(1,MSL)
    while int(L.sum())!=N:
        d=N-int(L.sum()); room=(MSL-L) if d>0 else (L-1)
        i=torch.nonzero(room>0).flatten()[0]; t=min(abs(d),int(room[i])); L[i]+=t if d>0 else -t
    so=torch.cat([torch.zeros(1,device=dev,dtype=torch.long),L.cumsum(0)]).long()
    return torch.randn(N,D,device=dev,generator=g), so, int(L.max())
def layer(x,w,so,msl,kern):
    attn,u,_,_=hstu_preprocess_and_attention(x=x,norm_weight=w["in_w"],norm_bias=w["in_b"],
        norm_eps=EPS,num_heads=NH,attn_dim=AD,hidden_dim=HD,uvqk_weight=w["uvqk_w"],
        uvqk_bias=w["uvqk_b"],max_seq_len=msl,seq_offsets=so,attn_alpha=1.0/(AD**0.5),
        causal=True,num_targets=None,max_attn_len=0,contextual_seq_len=0,
        recompute_uvqk_in_backward=False,recompute_normed_x_in_backward=False,
        sort_by_length=False,kernel=kern)
    return hstu_compute_output(attn=attn,u=u,x=x,norm_weight=w["out_w"],norm_bias=w["out_b"],
        norm_eps=EPS,output_weight=w["out_proj"],num_heads=NH,linear_dim=HD,dropout_ratio=0.0,
        training=False,concat_u=False,concat_x=True,mul_u_activation_type="silu",
        group_norm=False,recompute_y_in_backward=False,kernel=kern)
g=torch.Generator(device=dev).manual_seed(0)
wv=mkw(g); xv,sov,mslv=mkdata(g,100_000,1024)
ga=torch.Generator(device=dev).manual_seed(7)
wa=mkw(ga); xa,soa,msla=mkdata(ga,1_200_000,1024)
for v in wa.values(): v.requires_grad_(True)
xa.requires_grad_(True)
agg_stream=torch.cuda.Stream(); stop=threading.Event()
def aggressor():
    with torch.cuda.stream(agg_stream):
        while not stop.is_set():
            y=layer(xa,wa,soa,msla,HammerKernel.TRITON); y.sum().backward()
            for vv in wa.values(): vv.grad=None
            xa.grad=None
print(f"# pid={os.getpid()} aggressor={'SAME-PROCESS 2nd stream' if AGG else 'none'} repeats={REPEATS}")
th=None
if AGG:
    th=threading.Thread(target=aggressor,daemon=True); th.start(); time.sleep(25)
for name,kern in (("PYTORCH",HammerKernel.PYTORCH),("TRITON",HammerKernel.TRITON)):
    h={}
    for _ in range(REPEATS):
        with torch.no_grad(): y=layer(xv,wv,sov,mslv,kern)
        k=md5(y); h[k]=h.get(k,0)+1
    print(f"{name:8s} x{REPEATS}: {len(h)} distinct | "+", ".join(f'{a}x{b}' for a,b in h.items())
          +("  <== NONDETERMINISTIC" if len(h)>1 else ""))
stop.set()
