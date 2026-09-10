# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot ownership, durable failure reporting, and Graph replay contracts."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

TRACE_PATH = Path(__file__).resolve().parents[3] / "vllm/utils/k3_tensor_trace.py"
spec = importlib.util.spec_from_file_location("k3_trace_under_test", TRACE_PATH)
assert spec is not None and spec.loader is not None
trace_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trace_module
spec.loader.exec_module(trace_module)
TensorTrace = trace_module.TensorTrace


class TensorTraceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "trace"

    def trace(self, **kwargs):
        return TensorTrace(self.root, identity={"rank": 0, "role": "decode"}, **kwargs)

    def read(self, frame=0):
        return torch.load(self.root / f"frame-{frame:08d}.pt", weights_only=True)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_eager_d2h_preserves_source_without_retaining_device_snapshots(self):
        trace = self.trace()
        value = torch.arange(1024 * 1024, dtype=torch.float32, device="cuda")
        expected = value.cpu()
        torch.accelerator.synchronize()
        allocated = torch.accelerator.memory_allocated()
        trace.begin({"case": "direct-d2h"})
        trace.record("before", value)
        self.assertEqual(torch.accelerator.memory_allocated(), allocated)
        value.fill_(-1)
        trace.record("after", value)
        self.assertEqual(torch.accelerator.memory_allocated(), allocated)
        trace.end()
        trace.close()
        tensors = self.read()["tensors"]
        torch.testing.assert_close(tensors[0]["value"], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            tensors[1]["value"], torch.full_like(expected, -1), rtol=0, atol=0
        )

    def test_snapshot_failure_preserves_original_cause_and_releases_budget(self):
        trace = self.trace()
        tensor = torch.zeros(4)
        trace.begin({"case": "allocation-failure"})
        with (
            patch.object(
                torch.Tensor,
                "clone",
                side_effect=torch.OutOfMemoryError("test allocation"),
            ),
            self.assertRaisesRegex(RuntimeError, "OutOfMemoryError: test allocation"),
        ):
            trace.record("layer.72.activation", tensor)
        self.assertEqual(trace._pending, 0)
        error = json.loads((self.root / "incomplete.json").read_text())["error"]
        self.assertIn("layer.72.activation: OutOfMemoryError: test allocation", error)
        with self.assertRaisesRegex(RuntimeError, "OutOfMemoryError: test allocation"):
            trace.close()
        self.assertFalse((self.root / "recorder_closed.json").exists())

    def test_file_fragment_limit_is_independent_of_total_snapshot_budget(self):
        trace = self.trace(max_pending_bytes=1024, max_fragment_bytes=16)
        trace.begin({"case": "separate-file-budget"})
        value = torch.zeros(4)
        for layer in range(4):
            value.fill_(layer)
            trace.record(f"layer.{layer}", value)
        value.fill_(-1)
        trace.end()
        trace.close()
        for layer in range(4):
            frame = self.read(layer)
            self.assertEqual(len(frame["tensors"]), 1)
            torch.testing.assert_close(
                frame["tensors"][0]["value"], torch.full((4,), float(layer))
            )
            self.assertEqual(
                frame["metadata"]["trace_fragment"],
                {"index": layer, "final": layer == 3},
            )

    def test_single_large_tensor_stays_whole_in_dedicated_fragment(self):
        trace = self.trace(max_pending_bytes=128, max_fragment_bytes=16)
        trace.begin({"case": "whole-tensor"})
        trace.record("before", torch.zeros(2))
        trace.record("large", torch.arange(12))
        trace.record("after", torch.ones(2))
        trace.end()
        trace.close()
        self.assertEqual(
            [self.read(i)["tensors"][0]["name"] for i in range(3)],
            ["before", "large", "after"],
        )
        torch.testing.assert_close(
            self.read(1)["tensors"][0]["value"], torch.arange(12)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA Graph")
    def test_capture_larger_than_file_budget_replays_in_lossless_fragments(self):
        trace = self.trace(max_pending_bytes=1024, max_fragment_bytes=64)
        value = torch.ones(16, device="cuda")
        graph = torch.cuda.CUDAGraph()
        trace.begin_capture("bs1")
        with torch.cuda.graph(graph):
            for layer in range(3):
                trace.record(f"layer.{layer}", value * (layer + 1))
        trace.end_capture()
        for step in range(2):
            value.fill_(step + 2)
            graph.replay()
            trace.replay("bs1", {"step": step})
        trace.close()
        for step in range(2):
            for layer in range(3):
                frame = self.read(step * 3 + layer)
                self.assertEqual(frame["metadata"]["step"], step)
                self.assertEqual(frame["tensors"][0]["name"], f"layer.{layer}")
                torch.testing.assert_close(
                    frame["tensors"][0]["value"],
                    torch.full((16,), float((step + 2) * (layer + 1))),
                )


if __name__ == "__main__":
    unittest.main()
