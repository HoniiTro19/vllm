# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3 trace operations retained through torch.compile and CUDA Graph replay.

The custom op genuinely mutates a dedicated diagnostic token, never model data.
Its schema therefore makes the observation observable to functionalization/DCE.
The fake implementation performs no recording. Runtime observations belong to
an explicit eager or capture scope; Graph replay drains capture-owned buffers.
"""

from __future__ import annotations

import atexit
import dataclasses
import os
import socket
import threading
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import torch

from .tensor_trace import TensorTrace, enabled

_local = threading.local()
_lock = threading.RLock()
_owners = {}
_graphs = {}


def _stack():
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


def _scope_metadata():
    return getattr(_local, "scope_metadata", [])


def tensor_tree(name, value):
    if isinstance(value, torch.Tensor):
        yield name, value
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from tensor_tree(f"{name}.{index}", child)
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensor_tree(f"{name}.{key}", child)
    elif hasattr(value, "tensors"):
        # IntermediateTensors contains PP tensors, not the worker's KV cache.
        yield from tensor_tree(name, value.tensors)


def scalar_tree(name, value):
    if value is None or isinstance(value, (bool, int, float, str)):
        yield name, value
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from scalar_tree(f"{name}.{index}", child)
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from scalar_tree(f"{name}.{key}", child)


@torch.library.custom_op("k3_trace::snapshot", mutates_args=("token",))
def snapshot(name: str, value: torch.Tensor, token: torch.Tensor) -> None:
    stack = _stack()
    if not stack and not getattr(_local, "warmup_depth", 0):
        raise RuntimeError(f"compiled K3 trace outside model/capture scope: {name}")
    if stack:
        stack[-1].record(name, value, model_scopes=_scope_metadata())
    token.add_(1)


@snapshot.register_fake
def _snapshot_fake(name, value, token):
    return None


def record_module(module, name, value):
    # Hooks are only installed when enabled, and these attributes are static
    # module state while Dynamo traces the inner model.
    token = getattr(module, "_k3_trace_token", None)
    if token is None:
        return
    for path, tensor in tensor_tree(f"{module._k3_trace_path}.{name}", value):
        snapshot(path, tensor, token)


def _new_trace(name):
    identity = {
        "engine": "vllm",
        "run_id": os.environ["K3_TRACE_RUN_ID"],
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "name": name,
    }
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        identity["rank"] = torch.distributed.get_rank()
        identity["world_size"] = torch.distributed.get_world_size()
    return TensorTrace(
        Path(os.environ["K3_TRACE_ROOT"])
        / (f"compiled-{identity['host']}-{identity['pid']}-{uuid.uuid4().hex}"),
        identity=identity,
        max_pending_bytes=int(
            os.environ.get("K3_TRACE_MAX_PENDING_BYTES", str(4 * 1024**3))
        ),
    )


def _record_boundary(name, value):
    if not _stack() and getattr(_local, "warmup_depth", 0):
        return
    for path, tensor in tensor_tree(name, value):
        _stack()[-1].record(path, tensor, model_scopes=_scope_metadata())


@contextmanager
def model_scope(name, inputs=None, metadata=None):
    if not enabled():
        yield
        return
    with _lock:
        stack = _stack()
        if not stack and getattr(_local, "warmup_depth", 0):
            yield
            return
        owns_frame = not stack
        if owns_frame:
            key = (os.getpid(), threading.get_ident(), name)
            if key not in _owners:
                _owners[key] = _new_trace(name)
            trace = _owners[key]
            trace.handoff()
            trace.begin({"model_scope": name, "execution": "eager", **(metadata or {})})
            stack.append(trace)
        if not hasattr(_local, "scope_metadata"):
            _local.scope_metadata = []
        _local.scope_metadata.append({"name": name, **(metadata or {})})
        try:
            _record_boundary(f"{name}.inputs", inputs)
            yield
            if owns_frame:
                trace.end()
        except BaseException:
            if owns_frame:
                try:
                    trace._fail("vLLM K3 model execution aborted")
                except RuntimeError:
                    pass
            raise
        finally:
            _local.scope_metadata.pop()
            if owns_frame:
                stack.pop()


@contextmanager
def warmup_scope():
    _local.warmup_depth = getattr(_local, "warmup_depth", 0) + 1
    try:
        yield
    finally:
        _local.warmup_depth -= 1


def traced_warmup(function):
    @wraps(function)
    def run(*args, **kwargs):
        if not enabled():
            return function(*args, **kwargs)
        with warmup_scope():
            return function(*args, **kwargs)

    return run


@contextmanager
def graph_capture(metadata):
    if not enabled():
        yield None
        return
    with _lock:
        key = uuid.uuid4().hex
        trace = _new_trace(f"graph:{key}")
        _graphs[key] = trace
        trace.begin_capture(key)
        trace._write_json("capture.json", metadata)
        _stack().append(trace)
        try:
            yield key
            trace.end_capture()
        except BaseException:
            try:
                trace._fail("vLLM K3 CUDA Graph capture aborted")
            except RuntimeError:
                pass
            raise
        finally:
            _stack().pop()


def graph_replay(key, inputs, metadata):
    if key is None or (getattr(_local, "warmup_depth", 0) and not _stack()):
        return
    with _lock:
        trace = _graphs[key]
        trace.handoff()
        trace.replay(
            key,
            {
                **metadata,
                "live_tensor_timing": "after_replay",
                "argument_scalars": dict(scalar_tree("live_graph_inputs", inputs)),
            },
            dict(tensor_tree("live_graph_inputs", inputs)),
        )


@contextmanager
def graph_replay_scope(key, inputs, metadata):
    """Snapshot mutable graph buffers before execution and after execution."""
    if key is None or (getattr(_local, "warmup_depth", 0) and not _stack()):
        yield
        return
    with model_scope("graph.live_io", inputs, {"capture_key": key, **metadata}):
        yield
        _record_boundary("graph.live_io.after_replay", inputs)
        graph_replay(key, None, metadata)


def draft_graph_observations(speculator, input_batch, descriptor, stage, step=None):
    """Expose buffer handles; graph_replay_scope clones them at both boundaries.

    The target batch supplies request identity, while the draft buffers supply
    the actual draft positions/hidden states (which can change during replay).
    """
    if not enabled():
        return {}
    num_reqs = input_batch.num_reqs
    num_tokens = descriptor.num_tokens
    num_reqs_padded = descriptor.num_reqs or num_reqs
    inputs = {}
    for field in ("input_ids", "positions", "is_padding"):
        inputs[f"draft.{field}"] = getattr(speculator.input_buffers, field)[:num_tokens]
    inputs["draft.query_start_loc"] = speculator.input_buffers.query_start_loc[
        : num_reqs_padded + 1
    ]
    inputs["draft.seq_lens"] = speculator.input_buffers.seq_lens[:num_reqs_padded]
    for field, size in (
        ("hidden_states", num_tokens),
        ("inputs_embeds", num_tokens),
        ("idx_mapping", num_reqs_padded),
        ("last_token_indices", num_reqs_padded),
        ("sample_src_positions", num_reqs_padded),
        ("draft_tokens", num_reqs_padded),
        ("draft_logits", num_reqs_padded),
        ("draft_input_id_overrides", num_reqs_padded),
        ("cached_draft_input_ids", num_reqs_padded),
        ("cached_draft_input_embeds", num_reqs_padded),
        ("cached_target_hidden_states", num_reqs_padded),
    ):
        value = getattr(speculator, field, None)
        if isinstance(value, torch.Tensor):
            inputs[f"draft.{field}"] = value[:size]
    for field in ("current_draft_step", "temperature", "seeds"):
        value = getattr(speculator, field, None)
        if isinstance(value, torch.Tensor):
            inputs[f"draft.{field}"] = value
    result = fullgraph_observations(inputs, input_batch)
    result["trace_metadata"].update(
        draft_stage=stage,
        draft_step=step,
        num_speculative_steps=speculator.num_speculative_steps,
        draft_num_tokens_padded=num_tokens,
        draft_num_reqs_padded=num_reqs_padded,
    )
    return result


def context_observations(context):
    """Read known host scalars/tensor handles without synchronizing the GPU."""
    if context is None:
        return {"forward_context_available": False}, {}
    descriptor = context.batch_descriptor
    metadata = {
        "forward_context_available": True,
        "runtime_mode": context.cudagraph_runtime_mode.name,
        "batch_descriptor": (
            dataclasses.asdict(descriptor) if descriptor is not None else None
        ),
        "has_attention_metadata": context.attn_metadata is not None,
    }
    tensors = {"slot_mapping": context.slot_mapping, "is_padding": context.is_padding}
    batches = context.attn_metadata
    if not isinstance(batches, list):
        batches = [batches]
    for batch_index, batch in enumerate(batches):
        if not isinstance(batch, dict):
            continue
        for layer, attention in batch.items():
            key = f"attention.{batch_index}.{layer}"
            scalars = {}
            for field in (
                "num_actual_tokens",
                "num_prefills",
                "num_prefill_tokens",
                "num_decode_tokens",
                "num_decodes",
            ):
                value = getattr(attention, field, None)
                if isinstance(value, (int, bool)):
                    scalars[field] = value
            metadata[key] = scalars
            for field in (
                "query_start_loc",
                "seq_lens",
                "slot_mapping",
                "block_table",
                "state_indices_tensor",
            ):
                value = getattr(attention, field, None)
                if isinstance(value, torch.Tensor):
                    tensors[f"{key}.{field}"] = value
    return metadata, tensors


def fullgraph_observations(model_inputs, input_batch, slot_mapping=None):
    if not enabled():
        return {}
    tensors = {"model_inputs": model_inputs, "slot_mapping": slot_mapping}
    for field in (
        "input_ids",
        "positions",
        "idx_mapping",
        "expanded_idx_mapping",
        "expanded_local_pos",
        "query_start_loc",
        "seq_lens",
        "is_padding",
    ):
        value = getattr(input_batch, field, None)
        if isinstance(value, torch.Tensor):
            tensors[f"batch.{field}"] = value
    return {
        "trace_inputs": tensors,
        "trace_metadata": {
            "request_ids": input_batch.req_ids,
            "num_tokens": input_batch.num_tokens,
            "num_reqs": input_batch.num_reqs,
        },
    }


def install_model_trace(model, name, context_provider=None):
    if not enabled():
        return
    if hasattr(model, "_k3_trace_installed"):
        raise RuntimeError("vLLM K3 trace already installed")
    model._k3_trace_installed = True
    device = next(model.parameters()).device
    # Each module owns a real mutable scalar. Modules can run on independent
    # CUDA streams, so sharing a scalar across modules would introduce a race.
    installed = {}
    inventory_trace = _new_trace(f"{name}.inventory")
    _owners[(os.getpid(), threading.get_ident(), f"inventory:{uuid.uuid4().hex}")] = (
        inventory_trace
    )

    def refresh():
        inventory = []
        changed = False
        for path, module in model.named_modules():
            if isinstance(getattr(module, "_orig_mod", None), torch.nn.Module):
                # OptimizedModule forwards attributes to its wrapped module;
                # observe the original module, not the compiler wrapper.
                continue
            info = {"path": path, "type": type(module).__qualname__}
            shard = getattr(module, "shard_indices", None)
            if dataclasses.is_dataclass(shard):
                info["vocab_shard_indices"] = dataclasses.asdict(shard)
            inventory.append(info)
            if module in installed:
                continue
            changed = True
            installed[module] = path
            module.register_buffer(
                "_k3_trace_token",
                torch.zeros((), device=device, dtype=torch.int64),
                persistent=False,
            )
            module._k3_trace_path = f"{name}.{path}" if path else name
            if path:

                def hook(module, args, output):
                    record_module(module, "output", output)

                module.register_forward_hook(hook)
        if changed:
            inventory_trace._write_json(
                "module_inventory.json",
                {"modules": inventory, "coverage_verified": False},
            )

    def wrap(method_name):
        original = getattr(model, method_name)

        @wraps(original)
        def run(*args, **kwargs):
            stack = _stack()
            if not stack or stack[-1]._frame.capture_key is None:
                refresh()
            metadata, context_tensors = context_observations(
                context_provider() if context_provider else None
            )
            metadata["argument_scalars"] = dict(
                scalar_tree("inputs", {"args": args, "kwargs": kwargs})
            )
            with model_scope(
                f"{name}.{method_name}",
                {"args": args, "kwargs": kwargs, "context": context_tensors},
                metadata,
            ):
                result = original(*args, **kwargs)
                _record_boundary(f"{name}.{method_name}.output", result)
                return result

        setattr(model, method_name, run)

    refresh()
    wrap("forward")
    if hasattr(model, "compute_logits"):
        wrap("compute_logits")
    if hasattr(model, "embed_input_ids"):
        wrap("embed_input_ids")
    if hasattr(model, "combine_hidden_states"):
        wrap("combine_hidden_states")


def close_process():
    errors = []
    with _lock:
        for trace in list(_owners.values()) + list(_graphs.values()):
            trace._owner = threading.get_ident()
            try:
                trace.close()
            except RuntimeError as exc:
                errors.append(str(exc))
    if errors:
        raise RuntimeError("; ".join(errors))


atexit.register(close_process)
