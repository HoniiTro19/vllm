# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import DeepseekV3Config


class DeepseekV3SWAConfig(DeepseekV3Config):
    """Configuration for K3's dense, gated sliding-window Eagle3 head."""

    model_type = "deepseek_v3_swa"
    has_no_defaults_at_init = True

    def __init__(
        self,
        sliding_window: int = 2048,
        use_sliding_window: bool = True,
        mla_use_nope: bool = False,
        mla_use_output_gate: bool = True,
        rope_theta: float = 10000000.0,
        **kwargs,
    ) -> None:
        kwargs.setdefault("n_routed_experts", 0)
        kwargs.setdefault("n_shared_experts", 0)
        kwargs.setdefault("num_experts_per_tok", 0)
        kwargs.setdefault("num_hidden_layers", 1)
        rope_parameters = dict(
            kwargs.get("rope_parameters") or kwargs.get("rope_scaling") or {}
        )
        rope_parameters.setdefault("rope_type", "default")
        rope_parameters.setdefault("rope_theta", rope_theta)
        kwargs["rope_parameters"] = rope_parameters
        super().__init__(**kwargs)
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.mla_use_nope = mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate
        if (
            not use_sliding_window
            or not isinstance(sliding_window, int)
            or sliding_window < 1
            or mla_use_nope
            or not mla_use_output_gate
            or self.q_lora_rank is None
            or self.num_hidden_layers != 1
            or self.rope_parameters["rope_type"] != "default"
        ):
            raise ValueError(
                "K3 SWA Eagle3 requires one layer, Q-LoRA, default RoPE, "
                "a positive causal sliding window and an output gate."
            )
