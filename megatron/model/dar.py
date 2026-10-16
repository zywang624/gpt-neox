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

"""Distance-Adaptive Representation (DAR) self-attention.

Reference implementation of Sec. 2.1-2.3 of "Dual Dimensionality for Local and
Global Attention". For a query at position i and a key at position j:

    local  path (i - j <  w):  k_j = RoPE(h_j W_K),        v_j = h_j W_V
    global path (i - j >= w):  k_j = RoPE(h'_j W_K),       v_j = h'_j W_V
                               h'_j = h_j W_down W_up      (d -> d_down -> d)

Both paths share W_Q / W_K / W_V (the stock fused `query_key_value`) and are
normalized by a single softmax over all preceding positions. The bottleneck
W_down / W_up is purely linear (no bias, no activation).

Special case: `dar_window_size = 0` sends every token, including the current
one, through the bottleneck. This is the "Uniform reduction" baseline.
"""

import torch
import torch.nn as nn

from megatron import mpu
from megatron.model.positional_embeddings import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_torch,
)
from megatron.model.transformer import ParallelSelfAttention
from megatron.model.utils import exists


def build_local_window_mask(sq, sk, window_size, device):
    """Bool mask [1, 1, sq, sk], True where key j is in the local window of query i.

    Queries are the last `sq` of `sk` positions (sq < sk when decoding with a
    KV cache). Key j is local to query i iff 0 <= i - j < window_size.
    """
    q_pos = torch.arange(sk - sq, sk, device=device)
    k_pos = torch.arange(sk, device=device)
    distance = q_pos[:, None] - k_pos[None, :]
    local = (distance >= 0) & (distance < window_size)
    return local.view(1, 1, sq, sk)


class ParallelDARSelfAttention(ParallelSelfAttention):
    """Self-attention with full-dimensional local keys/values and
    bottlenecked global keys/values."""

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
        super().__init__(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            rpe=rpe,
            rotary=rotary,
            use_cache=use_cache,
            parallel_output=parallel_output,
        )
        if self.use_flash_attention or self.sparse:
            raise NotImplementedError(
                "DAR only supports the 'global' attention type, got "
                f"'{self.attention_type}'."
            )

        self.window_size = neox_args.dar_window_size
        if self.window_size is None or self.window_size < 0:
            raise ValueError(
                f"dar_window_size must be a non-negative integer, got {self.window_size}"
            )
        down_dim = neox_args.dar_down_dim
        if down_dim is None or down_dim <= 0:
            raise ValueError(f"dar_down_dim must be a positive integer, got {down_dim}")

        # [W_down, W_up] of Eq. (1)-(2).
        self.down_up_proj = nn.Sequential(
            nn.Linear(neox_args.hidden_size, down_dim, bias=False),
            nn.Linear(down_dim, neox_args.hidden_size, bias=False),
        )

    def attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        layer_past,
        attention_mask,
        global_key_layer,
        global_value_layer,
    ):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        global_key_layer = global_key_layer.view(
            output_size[3], output_size[0] * output_size[1], -1
        )

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Local and global raw attention scores. [b * np, sq, sk]
        local_scores = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )
        global_scores = torch.baddbmm(
            torch.empty_like(matmul_result),
            query_layer.transpose(0, 1),
            global_key_layer.transpose(0, 1).transpose(1, 2),
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # Pick the local score inside the window and the global score outside
        # it (Eq. (3)). [b, np, sq, sk]
        local_mask = build_local_window_mask(
            output_size[2], output_size[3], self.window_size, query_layer.device
        )
        attention_scores = torch.where(
            local_mask,
            local_scores.view(*output_size),
            global_scores.view(*output_size),
        )

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if self.use_cache:
            with torch.no_grad():
                attention_mask = attention_mask[
                    ..., : attention_scores.size(3), : attention_scores.size(3)
                ]

        # ===========================
        # Attention probs and dropout
        # ===========================

        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
            attention_scores += rpe  # [1, np, sq, sk]

        if self.pos_emb == "alibi":
            attention_scores = self.alibi_embed(attention_scores)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )
        global_value_layer = global_value_layer.view(
            global_value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # Local positions read local values, global positions read global
        # values. matmul: [b * np, sq, hn]
        local_mask = local_mask.view(1, output_size[2], -1).to(attention_probs.dtype)
        context_layer = torch.bmm(
            attention_probs * local_mask, value_layer.transpose(0, 1)
        ) + torch.bmm(
            attention_probs * (1.0 - local_mask), global_value_layer.transpose(0, 1)
        )

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def forward(self, hidden_states, attention_mask, layer_past=None):

        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Local path uses h, global path uses h' = h W_down W_up.
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)
        global_mixed_x_layer, _ = self.query_key_value(self.down_up_proj(hidden_states))

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        global_mixed_x_layer = global_mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(
            mixed_x_layer, 3
        )
        # The global path has no query of its own: queries always come from h.
        (_, global_key_layer, global_value_layer) = mpu.split_tensor_along_last_dim(
            global_mixed_x_layer, 3
        )

        if exists(self.rotary_emb):
            if exists(self.rotary_ndims):
                # partial rotary
                query_rot, query_pass = (
                    query_layer[..., : self.rotary_ndims],
                    query_layer[..., self.rotary_ndims :],
                )
                key_rot, key_pass = (
                    key_layer[..., : self.rotary_ndims],
                    key_layer[..., self.rotary_ndims :],
                )
                global_key_rot, global_key_pass = (
                    global_key_layer[..., : self.rotary_ndims],
                    global_key_layer[..., self.rotary_ndims :],
                )
            else:
                # full rotary
                query_rot, key_rot = query_layer, key_layer
                global_key_rot = global_key_layer
            apply_rotary_fn = (
                apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb
            )

            seq_len = key_layer.shape[0]
            offset = 0
            if exists(layer_past) and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                seq_len += offset
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            # Global keys get the same rotary embedding as local keys.
            _, global_key_layer = apply_rotary_fn(
                query_rot, global_key_rot, cos, sin, offset=offset
            )
            query_layer, key_layer = apply_rotary_fn(
                query_rot, key_rot, cos, sin, offset=offset
            )

            if exists(self.rotary_ndims):
                query_layer = torch.cat((query_layer, query_pass), dim=-1)
                key_layer = torch.cat((key_layer, key_pass), dim=-1)
                global_key_layer = torch.cat(
                    (global_key_layer, global_key_pass), dim=-1
                )

        # ==================================
        # Cache key and value for inference
        # ==================================

        if exists(layer_past) and layer_past.numel() > 0:
            past_key, past_value, past_global_key, past_global_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat(
                (past_value.type_as(value_layer), value_layer), dim=0
            )
            global_key_layer = torch.cat(
                (past_global_key.type_as(global_key_layer), global_key_layer), dim=0
            )
            global_value_layer = torch.cat(
                (past_global_value.type_as(global_value_layer), global_value_layer),
                dim=0,
            )

        if self.use_cache:
            present = torch.stack(
                (key_layer, value_layer, global_key_layer, global_value_layer)
            )

        context_layer = self.attention(
            query_layer,
            key_layer,
            value_layer,
            layer_past,
            attention_mask,
            global_key_layer,
            global_value_layer,
        )

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if self.use_cache:
            output = [output, present]

        return output, bias
