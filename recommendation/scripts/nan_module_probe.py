"""Attribute the first observed non-finite output or weight without a GPU sync.

Enable with NAN_MODULE_PROBE=1. NAN_MODULE_PROBE_PARAMS=1 also checks direct
parameters and FBGEMM TBE weight buffers before forward. Optimizer state is not
checked. Full weight scans perturb scheduling; this is a component probe, not
an uninstrumented reproduction-rate measurement.

Pre-hooks include kwargs and retain input/weight state for the matching call.
The first bad output or weight latches that state, step, and output emission order
on device. A pinned snapshot is read/reused only after a CUDA event's query()
reports completion; a later CPU iteration alone is not a completion guarantee.
No synchronize(), GPU item(), or GPU-to-CPU value read occurs on the step path.
"""

import dataclasses
import numbers
import os

import torch


class ModuleProbe:
    # Per-module first-bad record. STEP=0 means no bad output/weight observed yet.
    STEP, ORDER, INPUT_BAD, WEIGHT_BAD, INPUT_N, OUTPUT_N, COVERAGE, WEIGHT_N, OUTPUT_BAD = range(9)
    INPUT_COMPLETE, OUTPUT_COMPLETE, WEIGHTS_CHECKED = 1, 2, 4

    @staticmethod
    def _module_children(mod):
        # Inspect the registered graph directly, bypassing custom traversal.
        for name, child in mod._modules.items():
            if child is not None:
                yield name, child
        # TorchRec ShardedEmbeddingCollection keeps the compute lookups in a
        # plain list, outside _modules. Its registered `embeddings` children
        # expose checkpoint parameters but never execute a forward themselves.
        for index, child in enumerate(vars(mod).get("_lookups", ())):
            if isinstance(child, torch.nn.Module):
                yield f"_lookups.{index}", child

    @classmethod
    def _walk_modules(cls, model):
        seen = set()

        def walk(name, mod):
            if id(mod) in seen:
                return
            seen.add(id(mod))
            yield name, mod
            for child_name, child in cls._module_children(mod):
                yield from walk(f"{name}.{child_name}" if name else child_name, child)

        return list(walk("", model))

    def __init__(self, model):
        modules = self._walk_modules(model)
        self.names = [name or "<root>" for name, _ in modules]
        self.n = len(self.names)
        self.dev = next(
            (
                p.device
                for _, mod in modules
                for p in mod._parameters.values()
                if torch.is_tensor(p) and p.is_cuda
            ),
            torch.device("cuda"),
        )
        self.flags = torch.zeros((self.n, 9), dtype=torch.int32, device=self.dev)
        self.host = torch.zeros((self.n, 9), dtype=torch.int32).pin_memory()
        self.step = torch.zeros((), dtype=torch.int32, device=self.dev)
        self.copy_done = torch.cuda.Event()
        self._copy_pending = False
        self._disabled = False
        self._active = False
        self._rank = 0
        self._gs = 0
        self.reported = False
        self.handles = []
        self.leaf = [next(self._module_children(m), None) is None for _, m in modules]
        self.check_params = os.environ.get("NAN_MODULE_PROBE_PARAMS") == "1"
        self.chunk_elements = max(
            1, int(os.environ.get("NAN_MODULE_PROBE_CHUNK_ELEMENTS", 16 * 1024 * 1024))
        )
        self.log_shapes = os.environ.get("NAN_MODULE_PROBE_SHAPES") == "1"
        self._coverage_warnings = set()
        self._pending = [[] for _ in modules]
        self._weights = []
        self._executed = set()
        self._checked_state = set()
        self._tbe_buffer_counts = [0 for _ in modules]
        self._coverage_snapshot = None
        self._last_coverage_report = None
        state_count = 0
        tbe_count = 0
        for i, (_, mod) in enumerate(modules):
            # Use actual registered parameters: TorchRec wrappers override
            # named_parameters() to manufacture aliases of their child's TBE.
            weights = [
                (name, p)
                for name, p in mod._parameters.items()
                if torch.is_tensor(p) and p.numel() and p.is_floating_point()
            ]
            if hasattr(mod, "split_embedding_weights"):
                tbe_buffers = []
                for name in ("weights_dev", "weights_host", "weights_uvm"):
                    value = getattr(mod, name, None)
                    if torch.is_tensor(value) and value.numel() and value.is_floating_point():
                        tbe_buffers.append(value)
                        tbe_count += 1
                        self._tbe_buffer_counts[i] += 1
                        print(
                            f"[module-probe] TBE weight buffer {self.names[i]}.{name} "
                            f"numel={value.numel()} dtype={value.dtype} "
                            f"device={value.device} check_requested={self.check_params}",
                            flush=True,
                        )
                if tbe_buffers:
                    # Views cover actual table rows, excluding any unused
                    # physical-buffer padding. This does not copy the tables.
                    weights = [(name, p) for name, p in weights if not any(p is b for b in tbe_buffers)]
                    weights.extend(
                        (f"table[{j}]", value)
                        for j, value in enumerate(mod.split_embedding_weights())
                        if torch.is_tensor(value) and value.numel() and value.is_floating_point()
                    )
            self._weights.append(weights)
            state_count += len(weights)
            self.handles.append(
                mod.register_forward_pre_hook(self._mk_pre(i), with_kwargs=True)
            )
            self.handles.append(
                mod.register_forward_hook(self._mk_post(i), with_kwargs=True)
            )
        print(
            f"[module-probe] armed on {self.n} modules ({sum(self.leaf)} leaves) "
            f"dev={self.dev} params={self.check_params} "
            f"weight_tensors_available={state_count} "
            f"weight_tensors_configured={state_count if self.check_params else 0} "
            "weight_tensors_inspected=0 (awaiting executed hooks) "
            f"tbe_buffers={tbe_count} chunk_elements={self.chunk_elements}; "
            "optimizer state is not inspected",
            flush=True,
        )

    def _disable(self, where, error):
        if self._disabled:
            return
        self._disabled = True
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        print(f"[module-probe] DISABLED at {where}: {error}", flush=True)

    def _tensors(self, obj):
        """Return deduplicated floating tensors and whether traversal was complete."""
        tensors, seen = [], set()
        complete = True

        def visit(value, depth=0):
            nonlocal complete
            if value is None or isinstance(value, (str, bytes, numbers.Number, torch.dtype, torch.device)):
                return
            if id(value) in seen:
                return
            seen.add(id(value))
            # TorchRec LazyAwaitable.__getattr__ calls wait(). Even asking
            # whether it has .values or dataclass fields would materialize it.
            if self._is_awaitable(value):
                complete = False
            elif depth > 12:
                complete = False
            elif torch.is_tensor(value):
                if value.is_floating_point() and value.numel():
                    tensors.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, depth + 1)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item, depth + 1)
            elif dataclasses.is_dataclass(type(value)):
                for field in dataclasses.fields(value):
                    visit(getattr(value, field.name), depth + 1)
            elif callable(getattr(value, "values", None)):
                # KJT/JT integer values cannot be NaN; optional weights can.
                # Do not visit both values and _values (the same storage).
                visit(value.values(), depth + 1)
                weights = getattr(value, "weights_or_none", None)
                if callable(weights):
                    visit(weights(), depth + 1)
            elif torch.is_tensor(getattr(value, "_values", None)):
                visit(value._values, depth + 1)
                visit(getattr(value, "_weights", None), depth + 1)
            else:
                complete = False

        visit(obj)
        return tensors, complete

    @staticmethod
    def _is_awaitable(value):
        return any(base.__name__.endswith("Awaitable") for base in type(value).__mro__)

    def _chunks(self, tensor):
        """Bound temporary boolean allocations, including noncontiguous tensors."""
        if tensor.numel() <= self.chunk_elements:
            yield tensor
            return
        if tensor.is_contiguous():
            flat = tensor.view(-1)
            for start in range(0, flat.numel(), self.chunk_elements):
                yield flat[start : start + self.chunk_elements]
            return
        # Avoid reshape()/contiguous(), which could copy an entire weight table.
        dim = next(i for i, size in enumerate(tensor.shape) if size > 1)
        stride = max(1, self.chunk_elements // (tensor.numel() // tensor.shape[dim]))
        for start in range(0, tensor.shape[dim], stride):
            yield from self._chunks(
                tensor.narrow(dim, start, min(stride, tensor.shape[dim] - start))
            )

    def _bad_tensors(self, tensors):
        acc = torch.zeros((), dtype=torch.int32, device=self.dev)
        for tensor in tensors:
            if not tensor.is_cuda:
                raise RuntimeError(
                    "CPU floating tensor encountered; refusing a host/device reduction "
                    "transfer on the probe step path"
                )
            tensor_bad = torch.zeros((), dtype=torch.int32, device=self.dev)
            for chunk in self._chunks(tensor):
                bad = (~torch.isfinite(chunk).all()).to(
                    device=self.dev, dtype=torch.int32, non_blocking=True
                )
                tensor_bad = torch.maximum(tensor_bad, bad)
            acc += tensor_bad
        return acc

    def _warn_coverage(self, i, side):
        key = (i, side)
        if key not in self._coverage_warnings:
            self._coverage_warnings.add(key)
            print(
                f"[module-probe] incomplete {side} tensor traversal: {self.names[i]}",
                flush=True,
            )

    def _shape_log(self, args, kwargs):
        if not self.log_shapes or not 1 <= self._gs <= 12:
            return
        for name, value in list(enumerate(args)) + list(kwargs.items()):
            if self._is_awaitable(value):
                continue
            accessor = getattr(value, "values", None)
            if not torch.is_tensor(value) and callable(accessor):
                values = accessor()
                lengths = getattr(value, "_lengths", None)
                if torch.is_tensor(values):
                    cached = vars(value)
                    keys = cached.get("_keys")
                    lengths_per_key = cached.get("_length_per_key")
                    keys_ready = isinstance(keys, list) and all(isinstance(k, str) for k in keys)
                    batch = (
                        lengths.numel() // len(keys)
                        if keys_ready and keys and torch.is_tensor(lengths)
                        else "absent cache"
                    )
                    per_feature = (
                        dict(zip(keys, lengths_per_key))
                        if keys_ready and isinstance(lengths_per_key, list)
                        and len(keys) == len(lengths_per_key)
                        and all(isinstance(n, numbers.Integral) for n in lengths_per_key)
                        else "absent cache"
                    )
                    print(
                        f"[module-probe] step={self._gs} input={name} "
                        f"type={type(value).__name__} values_shape={tuple(values.shape)} "
                        f"dtype={values.dtype} device={values.device} "
                        f"lengths_shape={tuple(lengths.shape) if torch.is_tensor(lengths) else None} "
                        f"B={batch} cached_feature_value_counts={per_feature}",
                        flush=True,
                    )

    def _mk_pre(self, i):
        def hook(mod, args, kwargs):
            if self._disabled or self.reported or not self._active:
                return
            try:
                if i == 0:
                    self._shape_log(args, kwargs)
                tensors, complete = self._tensors((args, kwargs))
                if not complete:
                    self._warn_coverage(i, "input")
                ib = self._bad_tensors(tensors)
                weights = [p for _, p in self._weights[i]] if self.check_params else []
                pb = self._bad_tensors(weights)
                self._executed.add(i)
                self._checked_state.update((i, j) for j in range(len(weights)))
                self._pending[i].append((ib, pb, len(tensors), complete, len(weights)))
            except Exception as error:
                self._disable(f"{self.names[i]} pre-hook", error)

        return hook

    def _mk_post(self, i):
        def hook(mod, args, kwargs, output):
            if self._disabled or self.reported or not self._active:
                return
            try:
                ib, pb, input_n, input_complete, weight_n = self._pending[i].pop()
                tensors, output_complete = self._tensors(output)
                if not output_complete:
                    self._warn_coverage(i, "output")
                ob = self._bad_tensors(tensors)
                row = self.flags[i]
                first = (row[self.STEP] == 0) & ((ob > 0) | (pb > 0))
                coverage = (
                    (self.INPUT_COMPLETE if input_complete else 0)
                    | (self.OUTPUT_COMPLETE if output_complete else 0)
                    | (self.WEIGHTS_CHECKED if self.check_params else 0)
                )
                values = (self.step, self._rank, ib, pb, input_n, len(tensors), coverage, weight_n, ob)
                # The predicate is materialized before any record is updated.
                for field, value in enumerate(values):
                    row[field] = torch.where(first, value, row[field])
                self._rank += 1
            except Exception as error:
                self._disable(f"{self.names[i]} post-hook", error)

        return hook

    def set_step(self, gs):
        if self._disabled or self.reported:
            return
        try:
            self._gs = int(gs)
            self._rank = 0
            self.step.fill_(self._gs)
            self._active = True
        except Exception as error:
            self._disable("set_step", error)

    def pause(self):
        """Exclude backward recomputation/eval until the next explicit train step."""
        self._active = False
        if not self._disabled and any(self._pending):
            self._disable("pause", "unmatched forward pre-hook state")

    def pump(self):
        """Keep one snapshot in flight; never overwrite unread pinned memory."""
        if self._disabled or self.reported or self._copy_pending:
            return
        try:
            self.host.copy_(self.flags, non_blocking=True)
            self.copy_done.record(torch.cuda.current_stream(self.dev))
            self._coverage_snapshot = (
                self._gs, frozenset(self._executed), frozenset(self._checked_state)
            )
            self._copy_pending = True
        except Exception as error:
            self._disable("pump", error)

    def report_if_bad(self):
        if self._disabled or self.reported or not self._copy_pending:
            return
        try:
            if not self.copy_done.query():
                return
            self._copy_pending = False
            self._report_coverage()
            # query() confirmed D2H completion. Every value below is CPU-only.
            rows = [
                (int(row[self.STEP]), int(row[self.ORDER]), i, row.tolist())
                for i, row in enumerate(self.host)
                if int(row[self.STEP]) > 0
            ]
            if not rows:
                return
            self.reported = True
            rows.sort(key=lambda row: (row[0], row[1]))
            print("\n[module-probe] !! NON-FINITE MODULE REPORT", flush=True)
            print(
                "[module-probe] step order input_bad weight_bad output_bad input_n output_n leaf module",
                flush=True,
            )
            origins = []
            for step, order, i, values in rows:
                coverage = values[self.COVERAGE]
                finite_inputs = bool(coverage & self.INPUT_COMPLETE) and values[self.INPUT_BAD] == 0
                checked_weights = bool(coverage & self.WEIGHTS_CHECKED)
                candidate = (
                    values[self.OUTPUT_BAD] > 0
                    and finite_inputs
                    and (not checked_weights or values[self.WEIGHT_BAD] == 0)
                )
                if candidate and self.leaf[i]:
                    origins.append((step, order, i, checked_weights))
                mark = ""
                if candidate:
                    mark = " <== CANDIDATE (checked weights finite)" if checked_weights else " <== CANDIDATE (weights unchecked)"
                elif not coverage & self.INPUT_COMPLETE:
                    mark = " (input coverage incomplete)"
                if values[self.WEIGHT_BAD] > 0:
                    mark += " (weight already non-finite before forward)"
                print(
                    f"[module-probe] {step} {order} {values[self.INPUT_BAD]} "
                    f"{values[self.WEIGHT_BAD] if checked_weights else 'unchecked'} "
                    f"{values[self.OUTPUT_BAD]} "
                    f"{values[self.INPUT_N]} {values[self.OUTPUT_N]} "
                    f"{'Y' if self.leaf[i] else 'n'} {self.names[i]}{mark}",
                    flush=True,
                )
            if origins:
                step, order, i, checked = origins[0]
                print(
                    f"[module-probe] EARLIEST LEAF CANDIDATE: {self.names[i]} "
                    f"step={step} order={order} weights_checked={checked}; "
                    "this identifies an observed module boundary, not the responsible kernel",
                    flush=True,
                )
            else:
                print(
                    "[module-probe] no leaf with fully inspected finite inputs and "
                    "finite checked weights; inspect prior backward updates and coverage gaps",
                    flush=True,
                )
        except Exception as error:
            self._disable("report", error)

    def _report_coverage(self):
        """Only report inspected state after its snapshot's completion event."""
        step, executed, checked = self._coverage_snapshot
        signature = (executed, checked)
        if signature == self._last_coverage_report:
            return
        self._last_coverage_report = signature
        checked_owners = {i for i, _ in checked}
        checked_tbe_buffers = sum(self._tbe_buffer_counts[i] for i in checked_owners)
        print(
            f"[module-probe] completed coverage step={step} "
            f"executed_modules={len(executed)}/{self.n} "
            f"inspected_weight_tensors={len(checked)} "
            f"inspected_weight_numel={sum(self._weights[i][j][1].numel() for i, j in checked)} "
            f"tbe_buffers_inspected={checked_tbe_buffers}/{sum(self._tbe_buffer_counts)}",
            flush=True,
        )
        for i, weights in enumerate(self._weights):
            if weights and i not in checked_owners:
                print(
                    f"[module-probe] weight owner not inspected: {self.names[i]} "
                    f"tensors={len(weights)} forward_executed={i in executed}; "
                    "may alias state inspected through another owner",
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
