# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

from vllm.transformers_utils.configs.deepseek_v3_swa import DeepseekV3SWAConfig


class DeepseekSWAConfigTest(unittest.TestCase):
    def test_checkpoint_roundtrip_preserves_rope_window_and_dense_topology(self):
        config = DeepseekV3SWAConfig(
            architectures=["Eagle3DeepseekV2SWAForCausalLM"],
            hidden_size=7168,
            intermediate_size=28672,
            num_attention_heads=96,
            num_key_value_heads=96,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            max_position_embeddings=82001,
            eagle_aux_hidden_state_layer_ids=[1, 45, 89],
            target_hidden_size=7168,
            num_aux_hidden_states=3,
            rope_scaling=None,
            torch_dtype="bfloat16",
        )
        restored = DeepseekV3SWAConfig.from_dict(config.to_dict())
        self.assertEqual(restored.model_type, "deepseek_v3_swa")
        self.assertEqual(restored.architectures, config.architectures)
        self.assertEqual(
            restored.rope_parameters, {"rope_type": "default", "rope_theta": 10000000.0}
        )
        self.assertEqual(restored.sliding_window, 2048)
        self.assertEqual(restored.num_hidden_layers, 1)
        self.assertEqual(restored.n_routed_experts, 0)
        self.assertEqual(restored.n_shared_experts, 0)
        self.assertEqual(restored.eagle_aux_hidden_state_layer_ids, [1, 45, 89])
        self.assertTrue(restored.mla_use_output_gate)
        self.assertFalse(restored.mla_use_nope)

    def test_rejects_configs_that_would_silently_change_draft_semantics(self):
        for overrides in (
            {"sliding_window": 0},
            {"use_sliding_window": False},
            {"mla_use_nope": True},
            {"mla_use_output_gate": False},
            {"num_hidden_layers": 2},
            {"q_lora_rank": None},
            {
                "rope_parameters": {
                    "rope_type": "linear",
                    "factor": 2.0,
                    "rope_theta": 10000000.0,
                }
            },
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                DeepseekV3SWAConfig(**overrides)


if __name__ == "__main__":
    unittest.main()
