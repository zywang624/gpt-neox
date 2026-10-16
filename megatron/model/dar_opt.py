# Copyright (c) 2026 The DAR authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DAR-opt self-attention: DAR in an MLA-style latent formulation.

Training-time implementation of Sec. 2.4 / Appendix B of "Dual Dimensionality
for Local and Global Attention". Each token j produces two content latents and
one rotary key:

    c^L_j = RMSNorm(h_j W^{LD})     (d -> d^L_c)   used while i - j <  w
    c^D_j = RMSNorm(h_j W^{DD})     (d -> d^D_c)   used once  i - j >= w
    k^R_j = RoPE(h_j W^{KR})        (d -> d_r)     shared across heads

and per head a:

    q^C_{i,a} = h_i W^{CQ}_a        q^R_{i,a} = RoPE(h_i W^{QR}_a)
    k^C_j     = c^L_j W^{LK}  or  c^D_j W^{DK}
    v_j       = c^L_j W^{LV}  or  c^D_j W^{DV}
    s_ij      = alpha * (q^C_i . k^C_j + q^R_i . k^R_j),  alpha = 1/sqrt(d_h + d_r)

The content part carries no position (NoPE); position enters only through the
rotary part, which is identical for both paths. Local and distant scores share
one softmax. As in MLA's kv_a_layernorm, each content latent is RMS-normalized
before its up-projection; the cached latent is the normalized one, so the
weight absorption of Appendix B.2/B.4 (an inference-time rewrite) still
applies. Keys/values are materialized here for training.

As in MLA, every projection (query, rotary, latent down/up and output) is
linear without bias.

Special case: `dar_window_size = 0` drops the local latent entirely, so every
token uses c^D. This is the uniform-latent (plain MLA) baseline with
d_c = `dar_distant_latent_dim`.
"""

import math

import torch
import torch.nn as nn

from megatron import mpu
from megatron.model.dar import build_local_window_mask
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.norms import RMSNorm
from megatron.model.positional_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_torch,
)
from megatron.model.utils import get_fusion_type


class LatentRMSNorm(RMSNorm):
    """RMSNorm over a content latent, computed in fp32.

    Subclasses the stock RMSNorm so its `scale` is excluded from weight decay,
    but normalizes as x * rsqrt(mean(x^2) + eps) in fp32 (as in MLA) instead of
    taking the L2 norm in the input dtype, which can overflow in fp16.
    """

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.scale.float() * x).to(dtype)


class ParallelDAROptSelfAttention(nn.Module):
    """MLA-style attention with a high-dimensional latent inside the local
    window and a low-dimensional latent beyond it."""

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
        parallel_output=False,
    ):
        super().__init__()

        if mpu.get_model_parallel_world_size() != 1:
            raise NotImplementedError("DAR-opt does not support model parallelism.")
        if neox_args.attention_config[layer_number] != "global":
            raise NotImplementedError(
                "DAR-opt only supports the 'global' attention type, got "
                f"'{neox_args.attention_config[layer_number]}'."
            )
        if neox_args.pos_emb != "rotary" or rpe is not None:
            raise ValueError("DAR-opt requires pos_emb='rotary' (decoupled RoPE).")
        if use_cache:
            raise NotImplementedError(
                "DAR-opt has no KV cache in NeoX; convert to HF for generation."
            )

        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"
        self.layer_number = layer_number
        self.use_cache = use_cache

        hidden_size = neox_args.hidden_size
        self.num_heads = neox_args.num_attention_heads
        self.head_dim = mpu.divide(hidden_size, self.num_heads)

        self.window_size = neox_args.dar_window_size
        if self.window_size is None or self.window_size < 0:
            raise ValueError(
                f"dar_window_size must be a non-negative integer, got {self.window_size}"
            )
        local_dim = neox_args.dar_local_latent_dim
        distant_dim = neox_args.dar_distant_latent_dim
        rotary_dim = neox_args.dar_rotary_dim
        if distant_dim is None or distant_dim <= 0:
            raise ValueError(
                f"dar_distant_latent_dim must be a positive integer, got {distant_dim}"
            )
        if self.window_size > 0 and (local_dim is None or local_dim <= 0):
            raise ValueError(
                f"dar_local_latent_dim must be a positive integer, got {local_dim}"
            )
        if rotary_dim is None or rotary_dim <= 0 or rotary_dim % 2 != 0:
            raise ValueError(
                f"dar_rotary_dim must be a positive even integer, got {rotary_dim}"
            )
        self.rotary_dim = rotary_dim

        def linear(in_features, out_features):
            layer = nn.Linear(in_features, out_features, bias=False)
            init_method(layer.weight)
            return layer

        # Content query W^{CQ}.
        self.query = linear(hidden_size, self.num_heads * self.head_dim)

        # Distant path: W^{DD}, latent RMSNorm and [W^{DK}; W^{DV}].
        self.distant_latent_proj = linear(hidden_size, distant_dim)
        self.distant_latent_norm = LatentRMSNorm(
            distant_dim, eps=neox_args.rms_norm_epsilon
        )
        self.distant_kv_proj = linear(distant_dim, 2 * self.num_heads * self.head_dim)

        # Local path: W^{LD}, latent RMSNorm and [W^{LK}; W^{LV}]. Absent when w = 0.
        if self.window_size > 0:
            self.local_latent_proj = linear(hidden_size, local_dim)
            self.local_latent_norm = LatentRMSNorm(
                local_dim, eps=neox_args.rms_norm_epsilon
            )
            self.local_kv_proj = linear(local_dim, 2 * self.num_heads * self.head_dim)

        # Decoupled RoPE: shared rotary key W^{KR}, per-head rotary query W^{QR}.
        self.rotary_key = linear(hidden_size, rotary_dim)
        self.rotary_query = linear(hidden_size, self.num_heads * rotary_dim)
        self.rotary_emb = RotaryEmbedding(
            rotary_dim, base=neox_args.rotary_emb_base, precision=neox_args.params_dtype
        )

        # alpha = 1 / sqrt(d_h + d_r), same query-key layer scaling as stock NeoX.
        coeff = None
        self.norm_factor = math.sqrt(self.head_dim + rotary_dim)
        if neox_args.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.fp16,
            input_in_bf16=self.bf16,
            fusion_type=get_fusion_type(neox_args),
            mask_func=attention_mask_func,
            softmax_in_fp32=neox_args.attention_softmax_in_fp32
            or neox_args.apply_query_key_layer_scaling,
            scale=coeff,
        )
        self.attention_dropout = nn.Dropout(neox_args.attention_dropout)

        # Output W^O, without bias as in MLA.
        self.dense = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=hidden_size,
            output_size=hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            bias=False,
            skip_bias_add=True,
            parallel_output=parallel_output,
        )
        # ParallelTransformerLayer adds the returned attention bias in its fused
        # bias-dropout-residual, so it needs a tensor: hand it a constant zero.
        # Not a parameter and not saved in checkpoints.
        self.register_buffer(
            "zero_output_bias", torch.zeros(hidden_size), persistent=False
        )

    def _content_kv(self, latent, kv_proj, sq, b):
        """Up-project a content latent [sq, b, d_c] to keys/values
        [b * np, sq, hn] each."""
        key, value = kv_proj(latent).chunk(2, dim=-1)
        key, value = (
            t.reshape(sq, b * self.num_heads, self.head_dim).transpose(0, 1)
            for t in (key, value)
        )
        return key, value

    def forward(self, hidden_states, attention_mask, layer_past=None):

        # hidden_states: [sq, b, h]
        sq, b, _ = hidden_states.shape
        np_, hn, dr = self.num_heads, self.head_dim, self.rotary_dim

        # =====================
        # Queries and rotary key
        # =====================

        # [sq, b, h] -> [b * np, sq, hn]
        query_content = self.query(hidden_states).view(sq, b * np_, hn).transpose(0, 1)

        # q^R: [sq, b, np, dr], k^R: [sq, b, 1, dr]
        query_rot = self.rotary_query(hidden_states).view(sq, b, np_, dr)
        key_rot = self.rotary_key(hidden_states).view(sq, b, 1, dr)
        apply_rotary_fn = (
            apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb
        )
        cos, sin = self.rotary_emb(key_rot, seq_len=sq)
        query_rot, key_rot = apply_rotary_fn(query_rot, key_rot, cos, sin)
        # cos/sin follow the RotaryEmbedding buffer dtype, which need not match
        # the activations; keep the rotary term in the score dtype.
        query_rot = query_rot.to(query_content.dtype)
        key_rot = key_rot.to(query_content.dtype)

        # ===========================
        # Raw attention scores [b, np, sq, sk]
        # ===========================

        output_size = (b, np_, sq, sq)

        # Every product is scaled by alpha inside baddbmm, as in stock NeoX, so
        # unscaled scores are never materialized in fp16.
        alpha = 1.0 / self.norm_factor

        # Content term alpha * q^C . k^C on each path. [b * np, sq, sk]
        distant_key, distant_value = self._content_kv(
            self.distant_latent_norm(self.distant_latent_proj(hidden_states)),
            self.distant_kv_proj,
            sq,
            b,
        )
        attention_scores = torch.baddbmm(
            torch.empty(
                b * np_,
                sq,
                sq,
                dtype=query_content.dtype,
                device=query_content.device,
            ),
            query_content,
            distant_key.transpose(1, 2),
            beta=0.0,
            alpha=alpha,
        )

        if self.window_size > 0:
            local_key, local_value = self._content_kv(
                self.local_latent_norm(self.local_latent_proj(hidden_states)),
                self.local_kv_proj,
                sq,
                b,
            )
            local_scores = torch.baddbmm(
                torch.empty_like(attention_scores),
                query_content,
                local_key.transpose(1, 2),
                beta=0.0,
                alpha=alpha,
            )
            local_mask = build_local_window_mask(
                sq, sq, self.window_size, hidden_states.device
            )
            attention_scores = torch.where(
                local_mask,
                local_scores.view(*output_size),
                attention_scores.view(*output_size),
            ).view(b * np_, sq, sq)
            # Release the per-path scores right away to cap peak memory.
            del local_scores

        # Rotary term alpha * q^R . k^R (k^R broadcast across heads), added in
        # place: neither where() nor baddbmm() needs its output for backward.
        # [b * np, sq, sk]
        query_rot = query_rot.permute(1, 2, 0, 3).reshape(b * np_, sq, dr)
        key_rot = key_rot.permute(1, 2, 0, 3).expand(b, np_, sq, dr)
        key_rot = key_rot.reshape(b * np_, sq, dr)
        attention_scores.baddbmm_(query_rot, key_rot.transpose(1, 2), alpha=alpha)
        attention_scores = attention_scores.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, h]
        # =========================

        attention_probs = attention_probs.view(b * np_, sq, sq)
        if self.window_size > 0:
            local_mask = local_mask.view(1, sq, sq).to(attention_probs.dtype)
            context_layer = torch.bmm(
                attention_probs * local_mask, local_value
            ) + torch.bmm(attention_probs * (1.0 - local_mask), distant_value)
        else:
            context_layer = torch.bmm(attention_probs, distant_value)

        # [b * np, sq, hn] -> [sq, b, np * hn]
        context_layer = context_layer.view(b, np_, sq, hn).permute(2, 0, 1, 3)
        context_layer = context_layer.reshape(sq, b, np_ * hn)

        # =================
        # Output. [sq, b, h]
        # =================

        output, _ = self.dense(context_layer)
        return output, self.zero_output_bias
