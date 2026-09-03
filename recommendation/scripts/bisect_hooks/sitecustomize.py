# Per-op kernel bisect hook for the gfx1250 memory-access fault (docs/mi450.md [6]).
#
# The trainer spawns its rank processes with start_method="spawn", so a monkeypatch
# applied in the launcher process is lost. sitecustomize is imported by every
# interpreter on PYTHONPATH, including the spawned children, so the override lands
# in the process that actually runs the model.
#
# Nothing heavy is imported here: torch cannot be imported during site
# initialization (it hangs). Instead `__import__` is wrapped, and each op module
# is patched the moment it finishes loading -- which is still before the
# `from ... import fn` in its call sites binds a name, so call sites pick up the
# override.
#
# Usage (prepend this directory to PYTHONPATH):
#   HSTU_HAMMER_KERNEL=PYTORCH BISECT_TRITON_OPS=attn  -> only the attention op
#                                                         group runs on Triton.
#   HSTU_HAMMER_KERNEL=TRITON  BISECT_PYTORCH_OPS=attn -> everything on Triton
#                                                         except that group.
#
# RECORD_HSTU_MHA=/path/to/calls.pt additionally dumps the shape/layout metadata
# of every hstu_mha call (no tensor payloads, a few KB) so the standalone
# reproducer can replay the trainer's real call sequence without the dataset.
import builtins
import inspect
import os
import sys

_FORCE_TRITON = {g for g in os.environ.get("BISECT_TRITON_OPS", "").split(",") if g}
_FORCE_PYTORCH = {g for g in os.environ.get("BISECT_PYTORCH_OPS", "").split(",") if g}
_RECORD_PATH = os.environ.get("RECORD_HSTU_MHA", "")
_RECORD_LIMIT = int(os.environ.get("RECORD_HSTU_MHA_LIMIT", "6000"))

# group -> [(module path, function name), ...]
_GROUPS = {
    "attn": [
        ("generative_recommenders.ops.hstu_compute", "hstu_preprocess_and_attention"),
        ("generative_recommenders.ops.hstu_attention", "hstu_mha"),
    ],
    # Sub-groups of "attn": "prep" takes the fused Triton LN+addmm+attention
    # path, "mha" only swaps the attention op and leaves uvqk on PyTorch.
    "prep": [
        ("generative_recommenders.ops.hstu_compute", "hstu_preprocess_and_attention"),
    ],
    "mha": [
        ("generative_recommenders.ops.hstu_attention", "hstu_mha"),
    ],
    "out": [
        ("generative_recommenders.ops.hstu_compute", "hstu_compute_output"),
    ],
    "uqvk": [
        ("generative_recommenders.ops.hstu_compute", "hstu_compute_uqvk"),
    ],
    "jagged": [
        ("generative_recommenders.ops.jagged_tensors", "concat_2D_jagged"),
        ("generative_recommenders.ops.jagged_tensors", "split_2D_jagged"),
        (
            "generative_recommenders.ops.jagged_tensors",
            "jagged_dense_bmm_broadcast_add",
        ),
    ],
    "pos": [
        ("generative_recommenders.ops.position", "add_timestamp_positional_embeddings"),
    ],
    "ln": [
        ("generative_recommenders.ops.layer_norm", "layer_norm"),
        ("generative_recommenders.ops.layer_norm", "swish_layer_norm"),
        ("generative_recommenders.ops.layer_norm", "rms_norm"),
    ],
    "mm": [
        ("generative_recommenders.ops.mm", "addmm"),
    ],
}

# module path -> [(function name, kernel name), ...]
_WANTED = {}
for _group, _kernel_name in [(g, "TRITON") for g in _FORCE_TRITON] + [
    (g, "PYTORCH") for g in _FORCE_PYTORCH
]:
    for _mod_path, _fn_name in _GROUPS[_group]:
        _WANTED.setdefault(_mod_path, []).append((_fn_name, _kernel_name))


def _patch_module(mod_path):
    mod = sys.modules.get(mod_path)
    common = sys.modules.get("generative_recommenders.common")
    if mod is None or common is None:
        return False
    # A module mid-import is already in sys.modules but has no functions yet.
    if any(not hasattr(mod, fn) for fn, _ in _WANTED[mod_path]):
        return False
    for fn_name, kernel_name in _WANTED[mod_path]:
        original = getattr(mod, fn_name)
        if getattr(original, "_bisect_wrapped", False):
            continue
        kernel = common.HammerKernel[kernel_name]

        # Bind through the real signature: some call sites (ops.mm.addmm) pass
        # `kernel` positionally, so blindly adding it as a kwarg is a TypeError.
        sig = inspect.signature(original)

        def override(*args, _fn=original, _k=kernel, _sig=sig, **kwargs):
            bound = _sig.bind_partial(*args, **kwargs)
            bound.arguments["kernel"] = _k
            return _fn(*bound.args, **bound.kwargs)

        override.__name__ = fn_name
        override._bisect_wrapped = True
        setattr(mod, fn_name, override)
        # Safety net for call sites that already did `from ... import fn`.
        for other in list(sys.modules.values()):
            if other is None or other is mod:
                continue
            if getattr(other, fn_name, None) is original:
                setattr(other, fn_name, override)
        print(f"[bisect] {mod_path}.{fn_name} -> {kernel_name}", flush=True)
    return True


_RECORD_MOD = "generative_recommenders.ops.hstu_attention"
_records = []


def _install_recorder():
    mod = sys.modules.get(_RECORD_MOD)
    if mod is None or not hasattr(mod, "hstu_mha"):
        return False
    original = mod.hstu_mha
    if getattr(original, "_mha_recorded", False):
        return True
    # Fetch these from sys.modules rather than importing: this runs inside the
    # __import__ wrapper below, so a nested import statement recurses.
    atexit = sys.modules.get("atexit")
    torch = sys.modules.get("torch")
    if atexit is None or torch is None:
        return False

    def _dump():
        torch.save(_records, _RECORD_PATH)
        print(f"[record] {len(_records)} hstu_mha calls -> {_RECORD_PATH}", flush=True)

    def recorder(*args, _fn=original, **kwargs):
        # Positional-arg call sites exist; bind the ones the repro needs.
        names = (
            "max_seq_len",
            "alpha",
            "q",
            "k",
            "v",
            "seq_offsets",
            "causal",
            "dropout_pr",
            "training",
            "num_targets",
            "attn_scale",
            "max_attn_len",
            "contextual_seq_len",
        )
        a = dict(zip(names, args))
        a.update(kwargs)
        if len(_records) < _RECORD_LIMIT:
            nt = a.get("num_targets")
            _records.append(
                {
                    "max_seq_len": a["max_seq_len"],
                    "alpha": a["alpha"],
                    "shape": tuple(a["q"].shape),
                    "v_shape": tuple(a["v"].shape),
                    "dtype": a["q"].dtype,
                    "q_stride": tuple(a["q"].stride()),
                    "v_stride": tuple(a["v"].stride()),
                    "seq_offsets": a["seq_offsets"].detach().cpu(),
                    "num_targets": None if nt is None else nt.detach().cpu(),
                    "max_attn_len": a.get("max_attn_len", 0),
                    "contextual_seq_len": a.get("contextual_seq_len", 0),
                    "dropout_pr": a.get("dropout_pr", 0.0),
                    "training": a.get("training", True),
                    "requires_grad": bool(a["q"].requires_grad),
                }
            )
            if len(_records) % 200 == 0:
                _dump()
        return _fn(*args, **kwargs)

    recorder.__name__ = "hstu_mha"
    recorder._mha_recorded = True
    setattr(mod, "hstu_mha", recorder)
    for other in list(sys.modules.values()):
        if other is None or other is mod:
            continue
        if getattr(other, "hstu_mha", None) is original:
            setattr(other, "hstu_mha", recorder)
    atexit.register(_dump)
    print(f"[record] recording hstu_mha calls to {_RECORD_PATH}", flush=True)
    return True


if _WANTED or _RECORD_PATH:
    _real_import = builtins.__import__
    _remaining = set(_WANTED)
    _record_pending = [bool(_RECORD_PATH)]

    def _bisect_import(name, *args, **kwargs):
        mod = _real_import(name, *args, **kwargs)
        if _remaining:
            for mod_path in list(_remaining):
                if mod_path in sys.modules and _patch_module(mod_path):
                    _remaining.discard(mod_path)
        # The recorder must wrap the outermost hstu_mha, so install it only
        # after any bisect override for that module is already in place.
        if _record_pending[0] and _RECORD_MOD not in _remaining:
            if _install_recorder():
                _record_pending[0] = False
        return mod

    builtins.__import__ = _bisect_import
