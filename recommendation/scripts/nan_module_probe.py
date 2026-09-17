"""Name the first module that emits a non-finite value, without adding a sync.

WHY THIS SHAPE. Run e added a per-step `.item()` and its nan onset moved from
step 11 to step 5 -- instrumentation changed the result, so the result could not
be trusted. This probe therefore performs ZERO device->host syncs on the step
path. Every check accumulates into a device-resident int32 tensor; the host copy
is a non_blocking D2H issued once per step and READ ON THE FOLLOWING STEP, by
which time it has long since landed. The nan is absorbing (240+ consecutive nan
steps observed), so being one step behind costs nothing.

THE DISCRIMINATOR. Flagging "this module's output is non-finite" is not enough:
once a nan exists every downstream module and every ancestor container reports
it too. What identifies the culprit is a module whose OUTPUT is non-finite while
its INPUTS are all finite -- it manufactured the value rather than propagating
it. So both sides are tracked, and the report ranks by (first bad step, forward
execution order) and marks the modules where out_bad fired with in_bad clear.

Execution order is captured host-side on the first step -- forward hooks fire in
deterministic order, so recording the call sequence costs nothing on device.

Enable with NAN_MODULE_PROBE=1. Optional: NAN_MODULE_PROBE_PARAMS=1 also checks
each leaf module's parameters (OFF by default -- isfinite over a 2.3M x 512
embedding table every step is not free and would itself perturb timing).
"""
import os
import torch

_MAX_T = 16  # cap tensors inspected per module, guards against huge containers


class ModuleProbe:
    def __init__(self, model):
        self.names = []
        self.idx = {}
        for name, _ in model.named_modules():
            nm = name or "<root>"
            self.idx[nm] = len(self.names)
            self.names.append(nm)
        n = self.n = len(self.names)
        dev = None
        for p in model.parameters():
            dev = p.device
            break
        self.dev = dev if dev is not None else torch.device("cuda")
        # layout: [0:n) in_bad counts, [n:2n) out_bad counts, [2n:3n) first bad step
        self.flags = torch.zeros(3 * n, dtype=torch.int32, device=self.dev)
        self.host = torch.zeros(3 * n, dtype=torch.int32).pin_memory()
        self.step = torch.zeros((), dtype=torch.int32, device=self.dev)
        self.exec_rank = {}
        self._rank = 0
        self.reported = False
        self._warned = False
        self.check_params = os.environ.get("NAN_MODULE_PROBE_PARAMS") == "1"
        self.handles = []
        self.leaf = {}
        for name, m in model.named_modules():
            nm = name or "<root>"
            self.leaf[nm] = next(m.children(), None) is None
            self.handles.append(m.register_forward_hook(self._mk(nm)))
        print(
            f"[module-probe] armed on {n} modules "
            f"({sum(self.leaf.values())} leaves) dev={self.dev} "
            f"params={self.check_params}",
            flush=True,
        )

    # -- tensor discovery -------------------------------------------------
    def _tensors(self, obj, out=None, depth=0):
        if out is None:
            out = []
        if depth > 3 or len(out) >= _MAX_T:
            return out
        if torch.is_tensor(obj):
            if obj.is_floating_point() and obj.numel():
                out.append(obj)
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                self._tensors(o, out, depth + 1)
        elif isinstance(obj, dict):
            for o in obj.values():
                self._tensors(o, out, depth + 1)
        else:
            # KeyedJaggedTensor / JaggedTensor and friends expose .values()
            for attr in ("values", "_values"):
                v = getattr(obj, attr, None)
                if v is None:
                    continue
                try:
                    v = v() if callable(v) else v
                except Exception:
                    continue
                if torch.is_tensor(v):
                    self._tensors(v, out, depth + 1)
        return out

    def _bad(self, obj):
        """0-dim int32 count of non-finite tensors, or None if nothing to check."""
        ts = self._tensors(obj)
        if not ts:
            return None
        acc = None
        for t in ts:
            b = (~torch.isfinite(t).all()).to(torch.int32)
            acc = b if acc is None else acc + b
        return acc

    def _mk(self, name):
        i = self.idx[name]
        n = self.n

        def hook(mod, inputs, output):
            try:
                self._hook(mod, inputs, output, i, n, name)
            except Exception as e:  # an instrument must never kill the run
                if not self._warned:
                    self._warned = True
                    print(f"[module-probe] hook disabled after error: {e}", flush=True)

        return hook

    def _hook(self, mod, inputs, output, i, n, name):
        if True:
            if name not in self.exec_rank:
                self.exec_rank[name] = self._rank
                self._rank += 1
            ib = self._bad(inputs)
            if ib is not None:
                self.flags[i] += ib
            if self.check_params and self.leaf[name]:
                for p in mod.parameters(recurse=False):
                    if p.is_floating_point():
                        self.flags[i] += (~torch.isfinite(p).all()).to(torch.int32)
            ob = self._bad(output)
            if ob is not None:
                self.flags[n + i] += ob
                cur = self.flags[2 * n + i]
                self.flags[2 * n + i] = torch.where(
                    (cur == 0) & (ob > 0), self.step, cur
                )

    # -- step path (no syncs) ---------------------------------------------
    def set_step(self, gs):
        self.step.fill_(int(gs))

    def pump(self):
        """Async D2H. Read on the NEXT step -- never synchronize here."""
        self.host.copy_(self.flags, non_blocking=True)

    def report_if_bad(self):
        if self.reported:
            return
        n = self.n
        out_bad = self.host[n : 2 * n]
        if int(out_bad.sum()) == 0:  # host tensor: no device sync
            return
        self.reported = True
        in_bad = self.host[0:n]
        first = self.host[2 * n : 3 * n]
        rows = []
        for k in range(n):
            if int(out_bad[k]) == 0:
                continue
            rows.append(
                (
                    int(first[k]),
                    self.exec_rank.get(self.names[k], 1 << 30),
                    self.names[k],
                    int(in_bad[k]),
                    int(out_bad[k]),
                    self.leaf[self.names[k]],
                )
            )
        rows.sort(key=lambda r: (r[0], r[1]))
        print("\n[module-probe] !! NON-FINITE MODULE REPORT", flush=True)
        print(
            "[module-probe] %-6s %-6s %-7s %-7s %-5s %s"
            % ("step", "order", "in_bad", "out_bad", "leaf", "module"),
            flush=True,
        )
        for st, rk, nm, ib, ob, lf in rows[:60]:
            mark = "  <== ORIGIN" if ib == 0 else ""
            print(
                "[module-probe] %-6d %-6d %-7d %-7d %-5s %s%s"
                % (st, rk, ib, ob, "Y" if lf else "n", nm, mark),
                flush=True,
            )
        origins = [r for r in rows if r[3] == 0 and r[5]]
        if origins:
            st, rk, nm, _, _, _ = origins[0]
            print(
                f"[module-probe] EARLIEST LEAF ORIGIN (finite in, non-finite out): "
                f"{nm}  step={st} order={rk}",
                flush=True,
            )
        else:
            print(
                "[module-probe] no leaf had finite inputs and non-finite output -- "
                "the value likely originates inside a fused kernel whose inputs "
                "were already non-finite, or upstream of the first hooked module.",
                flush=True,
            )


_PROBE = None


def install(model):
    global _PROBE
    if _PROBE is None:
        _PROBE = ModuleProbe(model)
    return _PROBE


def get():
    return _PROBE
