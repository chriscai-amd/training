#!/usr/bin/env python3
"""E2 route (b): synthetic-input GEMM capture of the PATCHED testbed (no dataset needed).

Builds the real yambda-5b DlrmHSTU dense model (get_hstu_configs("yambda-5b"),
3 HSTU layers, max_seq_len 4096, bf16 autocast, TRITON hammer kernel) with
is_dense=True, so there are no embedding tables. Then it feeds main_forward
synthetic SequenceEmbeddings in exactly the layout preprocess() would produce
from the EmbeddingCollection: fp32 [sum_len, 512] per feature, jagged UIH
lengths, contextual uid/cross features of length 1, and one candidate. It runs
STEPS forward+backward steps of sum(aux_losses).backward(), as the trainer does.

GEMM shapes, layouts and dtypes depend only on the model code and the token
count T. They do not depend on embedding values or row counts. So under
HIPBLASLT_LOG_MASK=32 this issues the same hipBLASLt call KEYS as training,
and eval/capture_to_yaml.py (a COPY with STEPS=<steps>, RANKS=1) turns the logs
into arbor_hipblaslt/recapture_union.txt (roadmap E2).

What it does NOT reproduce: the embedding lookup and its backward (TBE, not
hipBLASLt), DMP/DDP collectives (not hipBLASLt), the optimizer step (fused
Adam, not hipBLASLt), and the data's exact jagged T. The capture normalises
T > 100000 to "T", so T only needs to be in the training range.

GPU WORK. Under this campaign's rules it is HUMAN-RUN ONLY (it is not
validator/bench.sh and has no hazard watch). Use scripts/e2_synthetic_capture.sh.
Run it on the image's STOCK hipBLASLt (no ARM_LIB_DIR): only the call keys
matter, and the stock library has no 10.2 tall-K kernels.

--cpu-dry-run checks the plumbing on CPU at a tiny T with the PYTORCH kernel
and records the aten mm-family calls. It does not exercise the Triton-only
paths (triton_addmm_bwd for G20, hstu_linear for G19), so it is a smoke test,
not evidence.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generative_recommenders.dlrm_v4.configs import get_hstu_configs  # noqa: E402
from generative_recommenders.common import HammerKernel  # noqa: E402
from generative_recommenders.modules.dlrm_hstu import (  # noqa: E402
    DlrmHSTU,
    SequenceEmbedding,
)


def cumsum0(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.new_zeros(1), torch.cumsum(x, 0)])


def make_batch(cfg, model, batch, uih_min, uih_max, dim, device, gen):
    uih_len = torch.randint(uih_min, uih_max + 1, (batch,), generator=gen).to(device)
    ones = torch.ones(batch, dtype=torch.int64, device=device)
    total_uih = int(uih_len.sum())

    def emb(n):
        return torch.randn(n, dim, generator=gen).to(device).requires_grad_(True)

    seq = {}
    ctx = set(cfg.contextual_feature_to_max_length.keys())
    for name in cfg.user_embedding_feature_names:
        if name in ctx:
            seq[name] = SequenceEmbedding(lengths=ones, embedding=emb(batch))
        else:
            seq[name] = SequenceEmbedding(lengths=uih_len, embedding=emb(total_uih))
    for name in cfg.item_embedding_feature_names:
        seq[name] = SequenceEmbedding(lengths=ones, embedding=emb(batch))

    nbits = len(cfg.action_weights)
    payload = {}
    for uih_name, cand_name in cfg.merge_uih_candidate_feature_mapping:
        if (cand_name not in cfg.item_embedding_feature_names
                and uih_name not in cfg.user_embedding_feature_names):
            if uih_name == cfg.uih_weight_feature_name:
                lo, hi = 0, 2 ** nbits
            elif uih_name == cfg.uih_action_time_feature_name:
                lo, hi = 1_600_000_000, 1_700_000_000
            else:
                lo, hi = 0, 2
            payload[uih_name] = torch.randint(lo, hi, (total_uih,), generator=gen).to(device)
            payload[cand_name] = torch.randint(lo, hi, (batch,), generator=gen).to(device)
    # candidates are queried after their history
    payload[cfg.candidates_querytime_feature_name] = payload[cfg.candidates_querytime_feature_name] + 200_000_000
    payload["uih_offsets"] = cumsum0(uih_len)
    payload["candidate_offsets"] = cumsum0(ones)
    return seq, payload, int(uih_len.max()), uih_len, 1, ones


class AtenMMRecorder(torch.utils._python_dispatch.TorchDispatchMode):
    OPS = {"mm", "addmm", "bmm", "baddbmm", "linear", "_addmm_activation"}

    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.__name__.split(".")[0]
        if name in self.OPS:
            self.calls.append({
                "op": name,
                "args": [
                    {"shape": list(a.shape), "stride": list(a.stride()), "dtype": str(a.dtype)}
                    for a in args if isinstance(a, torch.Tensor)
                ],
            })
        return func(*args, **(kwargs or {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "2")))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("BATCH_SIZE", "1024")))
    # yambda: history_length = min_history = 4086 interleaved, max_seq_len 4096.
    # Uniform [0, 4086] gives mean ~2043 tokens/sample, T ~2.09M at batch 1024,
    # inside the captured 1.59-2.53M range.
    ap.add_argument("--uih-min", type=int, default=int(os.environ.get("UIH_MIN", "0")))
    ap.add_argument("--uih-max", type=int, default=int(os.environ.get("UIH_MAX", "4086")))
    ap.add_argument("--layers", type=int, default=int(os.environ.get("HSTU_NUM_LAYERS", "3")))
    ap.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "4096")))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--out", default=None, help="json summary (dry run: aten calls)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    cfg = get_hstu_configs("yambda-5b", max_seq_len=a.max_seq_len, hstu_attn_num_layers=a.layers)
    model = DlrmHSTU(hstu_configs=cfg, embedding_tables={}, is_inference=False,
                     is_dense=True, bf16_training=True)
    if a.cpu_dry_run:
        device = torch.device("cpu")
        model.set_hammer_kernel(HammerKernel.PYTORCH)
    else:
        device = torch.device("cuda")
        model.set_hammer_kernel(HammerKernel[os.environ.get("HSTU_HAMMER_KERNEL", "TRITON").upper()])
    model = model.to(device).train()

    summary = {"steps": a.steps, "batch": a.batch, "uih": [a.uih_min, a.uih_max],
               "layers": a.layers, "max_seq_len": a.max_seq_len, "T_per_step": []}
    rec = AtenMMRecorder() if a.cpu_dry_run else None
    for step in range(a.steps):
        seq, payload, max_uih, uih_len, max_c, n_c = make_batch(
            cfg, model, a.batch, a.uih_min, a.uih_max, cfg.hstu_embedding_table_dim, device, gen)
        T = int(uih_len.sum()) + a.batch * (1 + len(cfg.contextual_feature_to_max_length))
        summary["T_per_step"].append(T)
        model.zero_grad(set_to_none=True)
        if rec is not None:
            rec.__enter__()
        try:
            out = model.main_forward(seq, payload, max_uih, uih_len, max_c, n_c)
            aux_losses = out[2]
            loss = sum(aux_losses.values())
            loss.backward()
        finally:
            if rec is not None:
                rec.__exit__(None, None, None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"[e2-synthetic] step {step} T={T} loss={float(loss):.5f} finite={bool(torch.isfinite(loss))}",
              flush=True)
    if rec is not None:
        summary["aten_calls"] = rec.calls
        print(f"[e2-synthetic] dry run: {len(rec.calls)} aten mm-family calls")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
