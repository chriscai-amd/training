"""Raw, local state for the rolling NaN replay recorder.

``collect_state`` never clones, transfers, or flushes tensors. Its first return
value contains live references; the recorder owns storage deduplication and
copying. The second value contains only serializable, non-tensor controls.

Restore in this order: ``restore_controls``, ``collect_state``, copy the saved
tensor bytes into the newly collected references, then ``restore_rng``. Controls
can recreate missing optimizer/gradient slots but do not restore tensor values.
Call these helpers at a completed-step boundary, after ``optimizer.zero_grad``.
This implementation deliberately rejects TBE caches, prefetching, and streaming.
"""

from __future__ import annotations

import enum
import importlib
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


_VERSION = 1
_HIDDEN_MODULE_LISTS = ("_lookups", "_emb_modules", "_embedding_lookups_per_rank")
_TBE_CONTROL_NAMES = (
    "optimizer_args", "step", "timestep", "iter_cpu", "learning_rate_tensor",
    "timesteps_prefetched", "lxu_cache_locations_list", "prefetched_info_list",
    "lxu_cache_locations", "lxu_cache_locations_empty", "_indices", "_offsets",
    "_vbe_B_offsets", "_vbe_max_B", "_writeback_precomputed_index",
    "last_uvm_cache_print_state", "last_reported_step", "debug_step",
    "last_reported_uvm_stats", "last_reported_uvm_stats_step",
    "detailed_mem_breakdown_report_count", "stochastic_rounding",
    "gwd_start_iter", "gwd_lower_bound", "_max_counter_update_freq",
    "_ensemble_mode", "_emainplace_mode",
)
_TBE_TOPOLOGY_NAMES = (
    "embedding_specs", "feature_table_map", "table_names", "optimizer",
    "weights_precision", "output_dtype", "pooling_mode", "weight_decay_mode",
    "bounds_check_mode", "bounds_check_mode_int", "bounds_check_version", "embedding_table_index_type",
    "embedding_table_offset_type", "use_rowwise_bias_correction",
    "info_B_num_bits", "info_B_mask", "is_experimental", "is_nobag",
    "use_homogeneous_placements", "use_uniq_cache_locations_bwd",
    "total_D", "max_D", "total_hash_size", "total_hash_size_bits",
    "weights_physical_placements", "weights_physical_offsets",
    "momentum1_physical_placements", "momentum1_physical_offsets",
    "momentum2_physical_placements", "momentum2_physical_offsets",
    "prev_iter_physical_placements", "prev_iter_physical_offsets",
    "row_counter_physical_placements", "row_counter_physical_offsets",
    "_used_rowwise_adagrad_with_counter",
    "_used_rowwise_adagrad_with_global_weight_decay",
)


def _class_name(value: Any) -> str:
    return type(value).__module__ + "." + type(value).__qualname__


def _walk_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    found: dict[str, torch.nn.Module] = {}
    seen: set[int] = set()

    def visit(path: str, module: torch.nn.Module) -> None:
        if id(module) in seen:
            return
        seen.add(id(module))
        found[path] = module
        for name, child in sorted(module._modules.items()):
            if child is not None:
                visit(path + "/" + name, child)
        # TorchRec training lookups are ordinary lists, outside _modules.
        for name in _HIDDEN_MODULE_LISTS:
            children = getattr(module, name, None)
            if isinstance(children, (list, tuple)):
                for index, child in enumerate(children):
                    if isinstance(child, torch.nn.Module):
                        visit(f"{path}/{name}/{index}", child)

    visit("model", model)
    return found


def _is_tbe(module: Any) -> bool:
    return (
        hasattr(module, "split_embedding_weights")
        and hasattr(module, "optimizer_args")
        and hasattr(module, "weights_dev")
    )


def _check_tbe(path: str, module: Any) -> None:
    for name in ("prefetch_pipeline", "enable_raw_embedding_streaming",
                 "use_writeback_bwd_prehook"):
        if getattr(module, name, False):
            raise ValueError(f"{path}: replay does not support {name}=True")
    if getattr(module, "prefetch_stream", None) is not None:
        raise ValueError(f"{path}: replay requires no prefetch stream")
    cache = getattr(module, "lxu_cache_weights", None)
    if isinstance(cache, torch.Tensor) and cache.numel():
        raise ValueError(f"{path}: replay does not support active embedding caches")
    for name in ("timesteps_prefetched", "lxu_cache_locations_list",
                 "prefetched_info_list"):
        if getattr(module, name, []):
            raise ValueError(f"{path}: replay requires an empty {name}")
    for spec in getattr(module, "embedding_specs", []):
        placement = getattr(spec[2], "value", spec[2])
        if placement != 0:  # FBGEMM EmbeddingLocation.DEVICE
            raise ValueError(f"{path}: replay currently requires DEVICE embeddings")


def _tensor_spec(tensor: torch.Tensor) -> dict[str, Any]:
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise ValueError(f"Unsupported replay tensor: {tensor.layout}, {tensor.device}")
    return {
        "shape": list(tensor.shape), "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype), "device": str(tensor.device),
    }


def _encode(value: Any, path: str, refs: dict[str, torch.Tensor],
            specs: dict[str, Any]) -> Any:
    """Replace tensor leaves with references and explicitly tag containers."""
    if isinstance(value, torch.Tensor):
        if path in refs and refs[path] is not value:
            raise ValueError(f"Duplicate replay tensor name: {path}")
        refs[path] = value
        specs[path] = _tensor_spec(value)
        return {"kind": "tensor", "name": path}
    if isinstance(value, enum.Enum):
        return {"kind": "enum", "module": type(value).__module__,
                "class": type(value).__qualname__, "name": value.name}
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return {"kind": "value", "value": value}
    if isinstance(value, (torch.dtype, torch.device)):
        return {"kind": "dtype" if isinstance(value, torch.dtype) else "device",
                "value": str(value)}
    if isinstance(value, Mapping):
        items = []
        for index, (key, item) in enumerate(sorted(
                value.items(), key=lambda pair: (_class_name(pair[0]), repr(pair[0])))):
            if isinstance(key, torch.Tensor):
                raise TypeError(f"Tensor dictionary key in replay controls: {path}")
            items.append((_encode(key, f"{path}/key/{index}", refs, specs),
                          _encode(item, f"{path}/value/{index}", refs, specs)))
        return {"kind": "dict", "items": items}
    if isinstance(value, (tuple, list)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [_encode(item, f"{path}/{index}", refs, specs)
                          for index, item in enumerate(value)]}
    raise TypeError(f"Unsupported replay control at {path}: {_class_name(value)}")


def _decode(value: Any, refs: dict[str, torch.Tensor], specs: dict[str, Any],
            *, allocate: bool = True) -> Any:
    kind = value["kind"]
    if kind == "tensor":
        name = value["name"]
        tensor = refs.get(name)
        spec = specs[name]
        if tensor is None or _tensor_spec(tensor) != spec:
            if not allocate:
                raise ValueError(f"Replay tensor topology changed: {name}")
            tensor = torch.empty_strided(
                tuple(spec["shape"]), tuple(spec["stride"]),
                dtype=getattr(torch, spec["dtype"].removeprefix("torch.")),
                device=spec["device"],
            )
            refs[name] = tensor
        return tensor
    if kind == "value":
        return value["value"]
    if kind == "dtype":
        return getattr(torch, value["value"].removeprefix("torch."))
    if kind == "device":
        return torch.device(value["value"])
    if kind == "enum":
        cls = importlib.import_module(value["module"])
        for component in value["class"].split("."):
            cls = getattr(cls, component)
        return cls[value["name"]]
    if kind == "dict":
        return {_decode(key, refs, specs): _decode(item, refs, specs)
                for key, item in value["items"]}
    if kind in ("tuple", "list"):
        result = [_decode(item, refs, specs) for item in value["items"]]
        return tuple(result) if kind == "tuple" else result
    raise ValueError(f"Unknown replay control kind: {kind}")


def _optimizer_tree(model: Any, optimizer: Any) -> dict[str, Any]:
    found: dict[str, Any] = {}
    seen: set[int] = set()

    def visit(path: str, opt: Any) -> None:
        if opt is None or id(opt) in seen:
            return
        seen.add(id(opt))
        found[path] = opt
        for index, child in enumerate(getattr(opt, "_optims", [])):
            label, inner = child if isinstance(child, tuple) else ("", child)
            visit(f"{path}/combined/{index}/{label}", inner)
        child = getattr(opt, "_optimizer", None)
        if child is not None:
            visit(path + "/wrapped", child)

    visit("optimizer", optimizer)
    visit("fused_optimizer", getattr(model, "fused_optimizer", None))
    return found


def _parameter_names(modules: dict[str, Any]) -> dict[int, str]:
    result = {}
    for path, module in modules.items():
        for name, param in sorted(module._parameters.items()):
            if param is not None:
                result.setdefault(id(param), f"{path}/parameter/{name}")
    return result


def _optimizer_signature(opt: Any, parameter_names: dict[int, str]) -> dict[str, Any]:
    keyed_names = {id(param): "keyed:" + name
                   for name, param in getattr(opt, "params", {}).items()}
    groups = []
    for group in opt.param_groups:
        names = []
        for param in group.get("params", []):
            name = parameter_names.get(id(param), keyed_names.get(id(param)))
            if name is None:
                raise ValueError("Optimizer parameter is outside the raw model tree")
            names.append(name)
        groups.append(names)
    return {"class": _class_name(opt), "groups": groups,
            "keyed_names": sorted(getattr(opt, "params", {}).keys())}


def _capture_policy(value: Any, path: str) -> Any:
    """Small scalar tensor hyperparameters need values for sequential replay."""
    refs: dict[str, torch.Tensor] = {}
    specs: dict[str, Any] = {}
    encoded = _encode(value, path, refs, specs)
    scalars = {}
    for name, tensor in refs.items():
        if tensor.numel() != 1:
            raise ValueError(f"Non-scalar tensor optimizer hyperparameter: {name}")
        scalars[name] = {"spec": specs[name], "value": tensor.item()}
    return {"encoded": encoded, "scalars": scalars}


def _restore_policy(value: Any, live: Any, path: str) -> Any:
    refs: dict[str, torch.Tensor] = {}
    ignored: dict[str, Any] = {}
    _encode(live, path, refs, ignored)
    specs = {name: scalar["spec"] for name, scalar in value["scalars"].items()}
    result = _decode(value["encoded"], refs, specs)
    with torch.no_grad():
        for name, scalar in value["scalars"].items():
            refs[name].fill_(scalar["value"])
    return result


def collect_state(model: torch.nn.Module, optimizer: Any
                  ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return live tensor references and independent, non-tensor controls."""
    refs: dict[str, torch.Tensor] = {}
    controls: dict[str, Any] = {
        "version": _VERSION, "tensor_specs": {}, "modules": {}, "optimizers": {},
    }
    specs = controls["tensor_specs"]
    modules = _walk_modules(model)
    for path, module in modules.items():
        saved: dict[str, Any] = {
            "class": _class_name(module), "training": module.training,
            "parameters": {}, "buffers": {}, "tbe": None,
        }
        for name, param in sorted(module._parameters.items()):
            base = f"{path}/parameter/{name}"
            saved["parameters"][name] = None if param is None else {
                "tensor": _encode(param, base, refs, specs),
                "requires_grad": param.requires_grad,
                "grad": _encode(param.grad, base + "/grad", refs, specs),
            }
        for name, buffer in sorted(module._buffers.items()):
            saved["buffers"][name] = _encode(
                buffer, f"{path}/buffer/{name}", refs, specs)
        if _is_tbe(module):
            _check_tbe(path, module)
            topology = {
                name: _encode(getattr(module, name), f"{path}/topology/{name}",
                              {}, {})
                for name in _TBE_TOPOLOGY_NAMES if hasattr(module, name)
            }
            # Future plain tensor attributes are included automatically. Unlike
            # state_dict, this also sees iter_cpu and learning_rate_tensor.
            names = set(_TBE_CONTROL_NAMES)
            names.update(name for name, item in vars(module).items()
                         if isinstance(item, torch.Tensor))
            attrs = {}
            for name in sorted(names):
                if hasattr(module, name):
                    value = getattr(module, name)
                    if name == "optimizer_args":
                        value = value._asdict()
                    attrs[name] = _encode(value, f"{path}/attribute/{name}", refs, specs)
            saved["tbe"] = {
                "topology": topology, "attributes": attrs,
                "learning_rate": module.learning_rate_tensor.item()
                if hasattr(module, "learning_rate_tensor") else None,
            }
        controls["modules"][path] = saved

    parameter_names = _parameter_names(modules)
    for path, opt in _optimizer_tree(model, optimizer).items():
        saved = {
            "topology": _optimizer_signature(opt, parameter_names),
            "groups": [_encode({key: value for key, value in group.items()
                                if key != "params"}, f"{path}/group/{index}", refs, specs)
                       for index, group in enumerate(opt.param_groups)],
            "step_groups": [_capture_policy(
                {key: value for key, value in group.items() if key != "params"},
                f"{path}/group/{index}")
                for index, group in enumerate(opt.param_groups)],
            "defaults": _encode(opt.defaults, path + "/defaults", refs, specs),
            "state": None,
        }
        # Wrapper/combined properties alias or synthesize their children's
        # state. Fused optimizer tensor states already live in the TBE buffers.
        leaf = not getattr(opt, "_optims", []) and getattr(opt, "_optimizer", None) is None
        if leaf and not _is_tbe(getattr(opt, "_emb_module", None)):
            state = {}
            for param, value in opt.state.items():
                if id(param) not in parameter_names:
                    raise ValueError(f"{path}: unknown optimizer state parameter")
                name = parameter_names[id(param)]
                state[name] = _encode(value, f"{path}/state/{name}", refs, specs)
            saved["state"] = state
        controls["optimizers"][path] = saved
    return refs, controls


def restore_controls(model: torch.nn.Module, optimizer: Any,
                     controls: dict[str, Any]) -> None:
    """Prepare saved slots and restore controls; the caller restores bytes next.

    No optimizer ``step``, ``zero_grad``, state-dict hook, or cache flush is run.
    Fused group LRs and the exact TBE LR tensor are separate saved state: neither
    is derived from the other during restoration.
    """
    if controls.get("version") != _VERSION:
        raise ValueError("Unsupported replay-state version")
    refs, current = collect_state(model, optimizer)
    modules = _walk_modules(model)
    optimizers = _optimizer_tree(model, optimizer)
    specs = controls["tensor_specs"]
    if current["modules"].keys() != controls["modules"].keys():
        raise ValueError("Replay module paths differ")
    if current["optimizers"].keys() != controls["optimizers"].keys():
        raise ValueError("Replay optimizer paths differ")
    # Validate all fixed model slots before changing controls. Dynamic plain
    # attributes, optimizer moments, and gradient slots may be reconstructed.
    for path, saved in controls["modules"].items():
        now = current["modules"][path]
        if saved["class"] != now["class"]:
            raise ValueError(f"Replay module class differs: {path}")
        for section in ("parameters", "buffers"):
            if saved[section].keys() != now[section].keys():
                raise ValueError(f"Replay {section} differ: {path}")
            for name, value in saved[section].items():
                if section == "parameters":
                    if (value is None) != (now[section][name] is None):
                        raise ValueError(f"Replay parameter presence differs: {path}/{name}")
                    if value is None:
                        continue
                    value = value["tensor"]
                if value["kind"] == "tensor":
                    _decode(value, refs, specs, allocate=False)
                elif value != now[section][name]:
                    raise ValueError(f"Replay buffer presence differs: {path}/{name}")
        if (saved["tbe"] is None) != (now["tbe"] is None):
            raise ValueError(f"Replay TBE differs: {path}")
        if saved["tbe"] is not None and saved["tbe"]["topology"] != now["tbe"]["topology"]:
            raise ValueError(f"Replay TBE configuration differs: {path}")
    for path, saved in controls["optimizers"].items():
        if saved["topology"] != current["optimizers"][path]["topology"]:
            raise ValueError(f"Replay optimizer topology differs: {path}")

    parameter_by_name = {}
    for path, saved in controls["modules"].items():
        module = modules[path]
        module.training = saved["training"]
        for name, value in saved["parameters"].items():
            if value is not None:
                param = module._parameters[name]
                param.requires_grad_(value["requires_grad"])
                param.grad = _decode(value["grad"], refs, specs)
                parameter_by_name[f"{path}/parameter/{name}"] = param
        if saved["tbe"] is not None:
            for name, value in saved["tbe"]["attributes"].items():
                decoded = _decode(value, refs, specs)
                if name == "optimizer_args":
                    module.optimizer_args = module.optimizer_args._replace(**decoded)
                else:
                    setattr(module, name, decoded)

    for path, saved in controls["optimizers"].items():
        opt = optimizers[path]
        for group, saved_group in zip(opt.param_groups, saved["groups"]):
            for key in list(group):
                if key != "params":
                    del group[key]
            group.update(_decode(saved_group, refs, specs))
        opt.defaults.clear()
        opt.defaults.update(_decode(saved["defaults"], refs, specs))
        if saved["state"] is not None:
            # Mutate the mapping in place so KeyedOptimizerWrapper aliases stay
            # attached. Clearing also removes moments absent in an earlier step.
            restored = {parameter_by_name[name]: _decode(value, refs, specs)
                        for name, value in saved["state"].items()}
            opt.state.clear()
            opt.state.update(restored)


def restore_step_policy(model: torch.nn.Module, optimizer: Any,
                        controls: dict[str, Any]) -> None:
    """Apply the next frame's training/LR policy after replaying its predecessor.

    This leaves weights, moments, gradients, iteration counters, and RNG state
    produced by the preceding replayed step intact. Restore RNG separately only
    when the experiment specifically calls for each frame's recorded RNG state.
    """
    if controls.get("version") != _VERSION:
        raise ValueError("Unsupported replay-state version")
    modules = _walk_modules(model)
    optimizers = _optimizer_tree(model, optimizer)
    parameter_names = _parameter_names(modules)
    if modules.keys() != controls["modules"].keys():
        raise ValueError("Replay module paths differ")
    if optimizers.keys() != controls["optimizers"].keys():
        raise ValueError("Replay optimizer paths differ")
    for path, saved in controls["optimizers"].items():
        opt = optimizers[path]
        if saved["topology"] != _optimizer_signature(opt, parameter_names):
            raise ValueError(f"Replay optimizer topology differs: {path}")
    for path, saved in controls["modules"].items():
        module = modules[path]
        if saved["class"] != _class_name(module):
            raise ValueError(f"Replay module class differs: {path}")
        module.training = saved["training"]
        if saved["tbe"] is not None:
            _check_tbe(path, module)
            attrs = saved["tbe"]["attributes"]
            for name in ("optimizer_args", "stochastic_rounding",
                         "gwd_start_iter", "gwd_lower_bound"):
                if name in attrs:
                    decoded = _decode(attrs[name], {}, {})
                    if name == "optimizer_args":
                        module.optimizer_args = module.optimizer_args._replace(**decoded)
                    else:
                        setattr(module, name, decoded)
            lr = saved["tbe"]["learning_rate"]
            if lr is not None:
                with torch.no_grad():
                    module.learning_rate_tensor.fill_(lr)
    for path, saved in controls["optimizers"].items():
        for index, (group, policy) in enumerate(zip(
                optimizers[path].param_groups, saved["step_groups"])):
            live = {key: value for key, value in group.items() if key != "params"}
            decoded = _restore_policy(policy, live, f"{path}/group/{index}")
            for key in list(group):
                if key != "params":
                    del group[key]
            group.update(decoded)


def capture_rng() -> dict[str, Any]:
    """Capture process RNGs without initializing CUDA for a CPU-only caller."""
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Replay CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def pack_kjt(kjt: Any) -> dict[str, Any] | None:
    """Own the input tensor bytes and retain KJT stride/inverse-index metadata."""
    if kjt is None:
        return None

    def own(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return None if tensor is None else tensor.detach().to("cpu", copy=True)

    inverse = kjt.inverse_indices_or_none()
    variable = kjt.variable_stride_per_key()
    return {
        "keys": list(kjt.keys()), "values": own(kjt.values()),
        "weights": own(kjt.weights_or_none()),
        "lengths": own(kjt.lengths_or_none()), "offsets": own(kjt.offsets_or_none()),
        "stride": None if variable else kjt.stride(),
        "stride_per_key_per_rank": [list(row) for row in kjt.stride_per_key_per_rank()]
        if variable else None,
        "inverse_indices": None if inverse is None else (list(inverse[0]), own(inverse[1])),
    }


def unpack_kjt(data: dict[str, Any] | None, device: Any) -> Any:
    if data is None:
        return None
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    kwargs = dict(data)
    for name in ("values", "weights", "lengths", "offsets"):
        if kwargs[name] is not None:
            kwargs[name] = kwargs[name].to(device=device, copy=True)
    if kwargs["inverse_indices"] is not None:
        names, tensor = kwargs["inverse_indices"]
        kwargs["inverse_indices"] = (list(names), tensor.to(device=device, copy=True))
    return KeyedJaggedTensor(**kwargs)
