# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3 Eagle3 with expanded MLA, NeoX RoPE and causal sliding-window attention.

The generic latent-cache MLA backends reject sliding windows. This head uses
vLLM's standard paged Attention with expanded K/V, as DeepseekV2Attention does,
while preserving the checkpoint's low-rank projections and output gate.
"""

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.deepseek_eagle3 import (
    DeepseekV2Eagle3DecoderLayer,
    DeepseekV2Eagle3Model,
    Eagle3DeepseekV2ForCausalLM,
)
from vllm.models.kimi_k3.common.compiled_trace import (
    install_model_trace,
    record_module,
)


class DeepseekV2SWAEagle3Attention(nn.Module):
    def __init__(
        self,
        *,
        vllm_config,
        config,
        hidden_size,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        q_lora_rank,
        kv_lora_rank,
        max_position_embeddings,
        cache_config,
        quant_config,
        prefix,
        input_size,
    ):
        super().__init__()
        if quant_config is not None:
            raise ValueError("K3 SWA Eagle3 requires its unquantized draft weights")
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size:
            raise ValueError("Eagle3 attention heads must divide tensor parallel size")
        self.num_local_heads = num_heads // tp_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        if v_head_dim > self.qk_head_dim:
            raise ValueError("Expanded MLA requires v_head_dim <= qk_head_dim")

        self.fused_qkv_a_proj = MergedColumnParallelLinear(
            input_size,
            [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
            bias=False,
            disable_tp=True,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * self.qk_head_dim,
            bias=False,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.g_proj = ColumnParallelLinear(
            input_size,
            num_heads * v_head_dim,
            bias=False,
            prefix=f"{prefix}.g_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_local_heads,
            self.qk_head_dim,
            self.qk_head_dim**-0.5,
            num_kv_heads=self.num_local_heads,
            cache_config=cache_config,
            per_layer_sliding_window=config.sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        record_module(self, "attention_input", hidden_states)
        qkv = self.fused_qkv_a_proj(hidden_states)[0]
        q_a, kv_a, k_pe = qkv.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        q = self.q_b_proj(self.q_a_layernorm(q_a))[0].view(
            -1, self.num_local_heads, self.qk_head_dim
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_a))[0].view(
            -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        record_module(self, "q_before_rope", q)
        record_module(self, "k_rope_input", k_pe)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        q = torch.cat((q_nope, q_pe), dim=-1)
        k = torch.cat((k_nope, k_pe.expand(-1, self.num_local_heads, -1)), dim=-1)
        if llama_4_scaling is not None:
            q = q * llama_4_scaling
        record_module(self, "q_after_rope", q)
        record_module(self, "k_after_rope", k)
        record_module(self, "v", v)
        v = torch.nn.functional.pad(v, (0, self.qk_head_dim - self.v_head_dim))
        output = (
            self.attn(q, k, v)
            .view(-1, self.num_local_heads, self.qk_head_dim)[..., : self.v_head_dim]
            .reshape(-1, self.num_local_heads * self.v_head_dim)
        )
        gate = self.g_proj(hidden_states)[0]
        record_module(self, "attention_before_gate", output)
        record_module(self, "raw_output_gate", gate)
        output = output * gate.sigmoid()
        record_module(self, "attention_after_gate", output)
        return self.o_proj(output)[0]


class DeepseekV2SWAEagle3DecoderLayer(DeepseekV2Eagle3DecoderLayer):
    attention_cls = DeepseekV2SWAEagle3Attention


class DeepseekV2SWAEagle3Model(DeepseekV2Eagle3Model):
    decoder_layer_cls = DeepseekV2SWAEagle3DecoderLayer


class Eagle3DeepseekV2SWAForCausalLM(Eagle3DeepseekV2ForCausalLM):
    model_cls = DeepseekV2SWAEagle3Model

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        install_model_trace(self, "eagle3")
