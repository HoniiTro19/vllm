# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real paged SWA against a dense reference before using the K3 draft adapter.

The adapter returns a gated, projected attention output. These tests guard its
NeoX RoPE, cached continuation, causal multi-token verification and window edge.
They use the native Attention backend with small weights; checkpoint loading,
TP8 and full-engine speculation require separate integration validation.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.config import CacheConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.deepseek_eagle3_swa import (
    DeepseekV2SWAEagle3Attention,
)
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture
def swa(dist_init, default_vllm_config):
    cfg = default_vllm_config
    cfg.attention_config.backend = AttentionBackendEnum.FLASH_ATTN
    cfg.cache_config = CacheConfig(block_size=32, cache_dtype="bfloat16")
    config = SimpleNamespace(
        rms_norm_eps=1e-6,
        sliding_window=2048,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000_000.0},
    )
    torch.manual_seed(318)
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        layer = DeepseekV2SWAEagle3Attention(
            vllm_config=cfg,
            config=config,
            hidden_size=64,
            input_size=128,
            num_heads=8,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=32,
            kv_lora_rank=16,
            max_position_embeddings=4096,
            cache_config=cfg.cache_config,
            quant_config=None,
            prefix="model.layers.0.self_attn",
        )
    with torch.no_grad():
        for parameter in layer.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1)
            else:
                parameter.normal_(std=parameter.shape[-1] ** -0.5)
    # Native per-layer cache is logically [block, head, token, K+V].
    layer.attn.kv_cache = torch.zeros(
        (66, 32, 8, 384), device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    block_table = torch.randperm(66, device="cuda", dtype=torch.int32)[None]
    return layer, cfg, block_table


def reference(layer, value):
    """Independent dense math, preserving BF16 projection/norm boundaries."""
    qkv = F.linear(value, layer.fused_qkv_a_proj.weight)
    q_a, kv_a, k_pe = qkv.split([32, 16, 64], dim=-1)

    def norm(x, weight):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * weight

    q = F.linear(norm(q_a, layer.q_a_layernorm.weight), layer.q_b_proj.weight)
    kv = F.linear(norm(kv_a, layer.kv_a_layernorm.weight), layer.kv_b_proj.weight)
    q = q.view(-1, 8, 192)
    k_nope, v = kv.view(-1, 8, 256).split([128, 128], -1)
    q_nope, q_pe = q.split([128, 64], -1)
    positions = torch.arange(value.shape[0], device=value.device)
    inv_freq = 10_000_000.0 ** (
        -torch.arange(0, 64, 2, device=value.device, dtype=torch.float32) / 64
    )
    angles = positions.float()[:, None] * inv_freq[None]
    cos = angles.cos().to(value.dtype).float()[:, None]
    sin = angles.sin().to(value.dtype).float()[:, None]

    def rotate(x):
        left, right = x.float().chunk(2, dim=-1)
        return torch.cat((left * cos - right * sin, right * cos + left * sin), -1).to(
            x.dtype
        )

    q = torch.cat((q_nope, rotate(q_pe)), -1)
    k = torch.cat((k_nope, rotate(k_pe[:, None]).expand(-1, 8, -1)), -1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * 192**-0.5
    distance = positions[:, None] - positions[None, :]
    scores.masked_fill_(((distance < 0) | (distance >= 2048))[None], -torch.inf)
    output = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
    output = output.to(value.dtype).reshape(-1, 8 * 128)
    output = output * F.linear(value, layer.g_proj.weight).sigmoid()
    return F.linear(output, layer.o_proj.weight)


def run_chunk(swa, value, start, end):
    layer, config, block_table = swa
    positions = torch.arange(start, end, device="cuda", dtype=torch.long)
    slots = block_table[0, positions // 32].long() * 32 + positions % 32
    metadata = FlashAttentionMetadata(
        num_actual_tokens=end - start,
        max_query_len=end - start,
        query_start_loc=torch.tensor(
            [0, end - start], device="cuda", dtype=torch.int32
        ),
        max_seq_len=end,
        seq_lens=torch.tensor([end], device="cuda", dtype=torch.int32),
        block_table=block_table,
        slot_mapping=slots,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    name = "model.layers.0.self_attn.attn"
    with set_forward_context(
        {name: metadata}, config, num_tokens=end - start, slot_mapping={name: slots}
    ):
        return layer(positions, value[start:end])


@torch.inference_mode()
def test_rope_gate_prefill_and_causal_cached_verify(swa):
    value = torch.randn((35, 128), device="cuda", dtype=torch.bfloat16)
    expected = reference(swa[0], value)
    for start, end in ((0, 31), (31, 32), (32, 35)):
        actual = run_chunk(swa, value, start, end)
        torch.testing.assert_close(actual, expected[start:end], atol=4e-3, rtol=3e-2)


@torch.inference_mode()
def test_decode_evicts_oldest_token_at_exact_window_boundary(swa):
    layer = swa[0]
    for parameter in layer.parameters():
        parameter.zero_()
    layer.q_a_layernorm.weight.fill_(1)
    layer.kv_a_layernorm.weight.fill_(1)
    # Only token zero contributes a value; uniform attention makes the window
    # boundary observable instead of hiding its effect inside BF16 tolerances.
    layer.fused_qkv_a_proj.weight[32, 0] = 1
    for head in range(8):
        layer.kv_b_proj.weight[head * 256 + 128, 0] = 20
    layer.o_proj.weight[0, 0] = 1
    value = torch.zeros((2051, 128), device="cuda", dtype=torch.bfloat16)
    value[0, 0] = 1
    prefill = run_chunk(swa, value, 0, 2048)
    assert prefill[-1, 0].item() > 0.01
    for start, end in ((2048, 2049), (2049, 2051)):
        actual = run_chunk(swa, value, start, end)
        torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
