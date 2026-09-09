# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone recorder contract tests, without importing GPU-only vLLM modules.

The failure under test is a missing/reordered observation after compilation or
graph replay, so compare saved tensor values rather than Python hook counters.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

COMMON = Path(__file__).resolve().parents[3] / "vllm/models/kimi_k3/common"


def load_local_module(filename):
    # Reuse the canonical module identity if another test imported vLLM first;
    # custom operators must not be registered twice in the same process.
    name = f"vllm.models.kimi_k3.common.{filename}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, COMMON / f"{filename}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_local_module("tensor_trace")
tracing = load_local_module("compiled_trace")


class CompiledTraceTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        env = patch.dict(
            os.environ,
            {"K3_TRACE_ROOT": str(self.root), "K3_TRACE_RUN_ID": "compile-test"},
        )
        env.start()
        self.addCleanup(env.stop)
        for name in ("_owners", "_graphs"):
            context: Any = patch.object(tracing, name, {})
            context.start()
            self.addCleanup(context.stop)
        self.addCleanup(tracing.close_process)

    def frames(self):
        frames = [
            torch.load(path, weights_only=True)
            for path in self.root.glob("*/frame-*.pt")
        ]
        return sorted(frames, key=lambda frame: frame["metadata"]["observation_id"])

    @torch.inference_mode()
    def test_fullgraph_retains_observations_before_input_mutation_on_every_call(self):
        def operation(value, token):
            tracing.snapshot("before", value, token)
            value.add_(10)
            tracing.snapshot("after", value, token)
            return value * 2

        for backend in ("aot_eager", "inductor"):
            with self.subTest(backend=backend):
                compiled = torch.compile(operation, backend=backend, fullgraph=True)
                token = torch.zeros((), dtype=torch.int64)
                for step in range(2):
                    value = torch.full((2, 3), float(step), dtype=torch.bfloat16)
                    with tracing.model_scope(
                        f"test.{backend}", metadata={"step": step}
                    ):
                        result = compiled(value, token)
                    torch.testing.assert_close(
                        result, torch.full_like(result, (step + 10) * 2)
                    )
                self.assertEqual(token.item(), 4)
        tracing.close_process()
        frames = self.frames()
        self.assertEqual(len(frames), 4)
        for frame in frames:
            step = frame["metadata"]["step"]
            tensors = {item["name"]: item["value"] for item in frame["tensors"]}
            torch.testing.assert_close(
                tensors["before"], torch.full((2, 3), float(step), dtype=torch.bfloat16)
            )
            torch.testing.assert_close(
                tensors["after"],
                torch.full((2, 3), float(step + 10), dtype=torch.bfloat16),
            )

    @torch.inference_mode()
    def test_draft_aux_projection_is_recorded_outside_forward(self):
        class Draft(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(6, 2, bias=False)
                self.fc.weight.copy_(torch.arange(12).view(2, 6))

            def combine_hidden_states(self, value):
                return self.fc(value)

            def forward(self, value):
                return value

        model = Draft()
        tracing.install_model_trace(model, "eagle3")
        value = torch.arange(6, dtype=torch.float32).view(1, 6)
        result = model.combine_hidden_states(value)
        value.fill_(-1)
        tracing.close_process()
        frames = self.frames()
        self.assertEqual(len(frames), 1)
        tensors = {item["name"]: item["value"] for item in frames[0]["tensors"]}
        torch.testing.assert_close(result, torch.tensor([[55.0, 145.0]]))
        torch.testing.assert_close(tensors["eagle3.fc.output"], result)
        torch.testing.assert_close(
            tensors["eagle3.combine_hidden_states.output"], result
        )

    @torch.inference_mode()
    def test_nn_module_hook_survives_compilation_of_inner_model(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = torch.nn.Sequential(
                    torch.nn.Linear(2, 2, bias=False), torch.nn.ReLU()
                )
                self.inner[0].weight.copy_(torch.eye(2))

            def forward(self, value):
                return self.inner(value)

        model = Model()
        tracing.install_model_trace(model, "main")
        model.inner = torch.compile(model.inner, backend="aot_eager", fullgraph=True)
        result = model(torch.tensor([[-1.0, 2.0]]))
        tracing.close_process()
        torch.testing.assert_close(result, torch.tensor([[0.0, 2.0]]))
        tensors = {item["name"]: item["value"] for item in self.frames()[0]["tensors"]}
        torch.testing.assert_close(
            tensors["main.inner.0.output"], torch.tensor([[-1.0, 2.0]])
        )
        torch.testing.assert_close(tensors["main.inner.1.output"], result)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA Graph")
    @torch.inference_mode()
    def test_graph_replay_saves_each_step_with_live_inputs(self):
        def operation(value, token):
            output = value * 3
            tracing.snapshot("layer.output", output, token)
            return output

        compiled = torch.compile(operation, fullgraph=True)
        value = torch.ones(4, device="cuda")
        token = torch.zeros((), device="cuda", dtype=torch.int64)
        with tracing.warmup_scope(), tracing.model_scope("warmup"):
            compiled(value, token)
        graph = torch.cuda.CUDAGraph()
        with (
            tracing.warmup_scope(),
            tracing.graph_capture({"bucket": 4}) as key,
            torch.cuda.graph(graph),
        ):
            compiled(value, token)
        for step in range(3):
            value.fill_(step)
            graph.replay()
            tracing.graph_replay(key, value, {"step": step})
        tracing.close_process()
        frames = [
            frame
            for frame in self.frames()
            if frame["metadata"].get("execution") == "graph_replay"
        ]
        self.assertEqual(len(frames), 3)
        for step, frame in enumerate(frames):
            tensors = {item["name"]: item["value"] for item in frame["tensors"]}
            torch.testing.assert_close(
                tensors["layer.output"], torch.full((4,), float(3 * step))
            )
            torch.testing.assert_close(
                tensors["live_graph_inputs"], torch.full((4,), float(step))
            )

    @torch.inference_mode()
    def test_compiled_warmup_is_excluded_without_losing_serving_observations(self):
        def operation(value, token):
            tracing.snapshot("value", value, token)
            return value + 1

        compiled = torch.compile(operation, backend="aot_eager", fullgraph=True)
        token = torch.zeros((), dtype=torch.int64)
        with tracing.warmup_scope(), tracing.model_scope("model.forward"):
            compiled(torch.zeros(2), token)
        with tracing.model_scope("model.forward"):
            result = compiled(torch.ones(2), token)
        tracing.close_process()
        self.assertEqual(len(self.frames()), 1)
        torch.testing.assert_close(
            self.frames()[0]["tensors"][0]["value"], torch.ones(2)
        )
        torch.testing.assert_close(result, torch.full((2,), 2.0))

    @torch.inference_mode()
    def test_replay_boundary_preserves_input_before_in_place_draft_update(self):
        with tracing.graph_capture({"test": "live_boundary"}) as key:
            pass
        value = torch.tensor([1, 2])
        for step in range(2):
            with tracing.graph_replay_scope(
                key, {"positions": value}, {"step": step, "request_ids": ["case-1"]}
            ):
                value.add_(10)
        tracing.close_process()
        live_frames = [
            frame
            for frame in self.frames()
            if frame["metadata"].get("model_scope") == "graph.live_io"
        ]
        self.assertEqual(len(live_frames), 2)
        for step, frame in enumerate(live_frames):
            tensors = {item["name"]: item["value"] for item in frame["tensors"]}
            self.assertEqual(frame["metadata"]["request_ids"], ["case-1"])
            torch.testing.assert_close(
                tensors["graph.live_io.inputs.positions"],
                torch.tensor([1, 2]) + 10 * step,
            )
            torch.testing.assert_close(
                tensors["graph.live_io.after_replay.positions"],
                torch.tensor([1, 2]) + 10 * (step + 1),
            )


if __name__ == "__main__":
    unittest.main()
