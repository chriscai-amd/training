# NaN source and saved-capture review — 2026-09-18

This review narrows the next experiments; **it does not identify the NaN root
cause**. It used repository source, logs, and CPU-only reads of saved tensors.
No GPU kernels were launched. The active fresh-test stack is
`recommendation-gfx1250-20260910:triton-7ff97e`, with Triton
`3.8.0+git7ff97e31` / commit
`7ff97e310935b4a79794878dbc911f9af25d38d9`.

The fresh post-AC baseline `~/mi450_logs/nan_post_ac_20260918_0347/baseline01.log`
completed 20 steps with finite loss and exit 0. Its step-8 large NE and high
training-window AUC recur without a NaN; the metric explanation below is
independent of a new NaN reproduction.

## What the old captures establish

Files inspected with `torch.load(..., map_location="cpu")`, with GPU visibility
disabled in the analysis process:

- `/home/chcai/dlrm_data/nan_capture/postonset_0917e/step10_pre_bwd.pt`
- `/home/chcai/dlrm_data/nan_capture/postonset_0917e/step11_pre_bwd.pt`
- `/home/chcai/dlrm_data/nan_capture/postonset_0917e/step12_pre_bwd.pt`

All three are already after that run's NaN onset. Each contains:

| Quantity | Measured value |
|---|---|
| Effective batch | 1024 |
| UIH `item_id`, `artist_id`, `album_id` lengths | Every row has length **1697**; total **1,737,728** per feature |
| Contextual feature lengths | Every row has length 1 |
| Candidate feature lengths | Every row has length 1 |
| Predictions | All 1024 non-finite |
| Labels | All 1024 finite and **0** |
| Supervision weights | All 1024 finite and **1** |
| Dense parameters | All **69** captured tensors entirely non-finite |
| Omitted large parameters | 11; embedding-table contents were not captured |

Further measurement of step 10 shows extreme repetition:

| Feature | Entries | Unique IDs | Details |
|---|---:|---:|---|
| `uid` | 1024 | 1 | ID 49544 |
| `user_x_artist` | 1024 | 1 | ID 21734862 |
| `user_x_album` | 1024 | 1 | ID 4298714 |
| UIH `item_id` | 1,737,728 | 104 | IDs 3734–2308494 |
| UIH `artist_id` | 1,737,728 | 90 | IDs 4353–321454 |
| UIH `album_id` | 1,737,728 | 101 | IDs 11309–832359 |
| `item_candidate_id` | 1024 | 1 | ID 26732 |

This is a full batch with heavily repeated history, not a partial batch. A
random-index TBE microbenchmark does not cover this duplication pattern.

`MIN_HISTORY=4086` is an anchor-eligibility floor on raw prior events. It does
**not** guarantee 4086 selected UIH events: the default interleaved history
builder filters into three behavior pools, independently caps each pool at
1362, and does not redistribute unused capacity. See
[`yambda.py`](../../generative_recommenders/dlrm_v4/datasets/yambda.py),
`_gather_interleaved_history` and the anchor-positions builder. The stronger
“full-length by construction” claim in the investigation notes and gin comments
is contradicted by both source and these captures.

## Loss and propagation boundaries

[`multitask_module.py`](../../generative_recommenders/modules/multitask_module.py)
computes binary cross entropy with logits in fp32, divides by
`mt_weights.sum(-1).clamp(min=1.0)`, then multiplies by causal weight 0.2.
Labels come from an integer bitmask comparison and default supervision weights
are ones. There is no apparent zero-denominator mechanism in this loss path.
The only configured task is `listen_plus`, so “only listen_plus is bad; other
terms finite: []” does not narrow the producer among tasks.

The streaming loop in
[`train/utils.py`](../../generative_recommenders/dlrm_v4/train/utils.py)
calls forward, backward, dense gradient clipping, and optimizer step in that
order. It is separate from the optional TorchRec training-pipeline path.
The sparse RowWiseAdagrad update is fused into TBE backward by
`make_optimizer_and_shard`, so a bad embedding weight written at step N can
first become visible in step N+1's forward.

Dense clipping is another propagation mechanism: with `GRAD_CLIP_NORM=1`, a
single NaN gradient makes the total norm and clipping coefficient NaN; the
default `clip_grad_norm_` then multiplies all dense gradients by that coefficient.
Adam can consequently poison every dense parameter. Thus the all-NaN dense
state and absorbing loss in the captures do not prove a wild store, nor do they
identify which gradient or forward value first went bad. Check dense gradients
**before clipping** when a forward-only probe is clean. Using
`error_if_nonfinite=True` diagnostically would stop at this boundary, before
clipping spreads the corruption, but still would not identify the first kernel.

## The large NE is explained by metric semantics

The installed TorchRec sources are under
`/opt/venv/lib/python3.12/site-packages/torchrec/metrics/` inside the pinned
container. `ne.py` clamps mean label to `eta=1e-12`; with all-negative labels,
normalized entropy is approximately `mean_BCE / -ln(1-eta)`. It can therefore be
around `7e11` while ordinary loss and predictions are healthy.

The fresh baseline's step-7 and step-8 losses are 0.13599 and 0.13577. Undoing
the 0.2 causal weight gives average BCE 0.6794. Dividing by
`-ln(1-1e-12)` gives **679,415,029,828.8**, consistent with its logged step-8
`window_ne=6.7942e11` at the available print precision. This is strong evidence
for a single-class metric denominator, not a numerical precursor to NaN.

There is also a window-size difference: `train_ranker.py` configures 2500
samples, but `rec_metric.py:WindowBuffer` evicts **whole batches**, so NE retains
only the last two 1024-sample batches. `auc.py` trims individual samples and
retains 2500. At step 8 its window can still include 452 samples from step 6,
while NE uses only steps 7–8. A high AUC at step 8 and an almost-zero NE
denominator are therefore not contradictory. AUC explicitly returns 0.5 when
its entire window has only one class. The exact step-8 AUC cannot be replayed
from the post-onset captures; those do not contain its predictions.

Finally, `MetricsLogger.compute_and_log`'s `REACHED AUC>=...` message only
latches a TensorBoard/time-to-target reporting flag. It neither stops training
nor emits MLPerf `run_stop`. Streaming convergence decisions consume holdout
evaluation results separately. The training-window message is misleading as
convergence evidence, but the source does not support the notes' stronger
claim that this message itself causes a false early termination.

## Ranked next experiments on the pinned stack

Keep batch 1024, `START_TS=0`, embedding scale 0.25, seed 1, and
`AMDGCN_USE_BUFFER_OPS=0`, with unique run/checkpoint names. Preserve the default
stage clamp and separated-RNG workaround. Compare repeat counts and validated
steps, since the old short-run NaN was nondeterministic and on Triton 3.6.0.

| Priority | Experiment | What it can establish |
|---:|---|---|
| 1 | Repeat the uninstrumented short baseline; then run the repaired module probe with parameter checks, real batch/N summaries, and gradient checks before clipping if needed | Establish whether the early NaN reproduces on this image/Triton, then locate the first bad boundary. A clean short run does not test the old step-1340 long-run failure. |
| 2 | If the first bad boundary is embedding output, repeat with **`EC_INDEX_DEDUP=0`** | Tests FBGEMM `jagged_unique_indices`, reverse-index reconstruction, and their interaction with highly duplicated early batches. It also changes allocation/timing, so suppression alone is not proof of a bad dedup kernel. |
| 3 | If the first bad boundary is dense HSTU output, use `scripts/bisect_hooks` with **`BISECT_PYTORCH_OPS=out`**, then a separate `ln,pos,jagged,mm` arm as indicated by the probe | Replaces bounded op groups while retaining the other Triton paths. The `out` group covers fused norm/multiply/dropout and output projection; it is a useful split before replacing attention. |
| 4 | Once a NaN hit rate exists, add **`AMD_SERIALIZE_KERNEL=3`** as a separate arm; a later narrow fence between forward and backward can refine it | Tests scheduling sensitivity suggested by prior standalone overlap faults. Suppression does not identify a particular kernel or prove that those hard faults and this NaN share a cause. |
| 5 | If embedding communication actually executes on the single-rank path, test **`SPARSE_A2A_FWD=fp32 SPARSE_A2A_BWD=fp32`**; if embedding state is implicated, isolate fused TBE update from gather explicitly | Discriminates the communication codec or fused optimizer path. The qcomm-enabled startup message alone does not prove a codec executes on one GPU. `SPARSE_LR=0` does not remove TBE backward and `0 * NaN` remains NaN, so zero LR is not an optimizer bypass. |

Avoid starting with an all-PyTorch attention arm at production batch: the
reference `pytorch_hstu_mha` materializes dense `[B,H,S,S]` attention tensors, so
its memory demand and scheduling differ substantially. Use recorded real
shapes and a targeted attention replay if the probe implicates that boundary.
The old captures can provide index/shape cases, but replaying their already
poisoned dense state cannot establish the original cause.
