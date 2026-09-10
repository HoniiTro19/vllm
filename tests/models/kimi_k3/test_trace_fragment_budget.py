# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot ownership, durable failure reporting, and Graph replay contracts."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

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
