"""Transformer with Layer Adaptive Representation (LAR).

Each LT layer learns a per-layer scalar alpha that selects between a wide
(d_high) and a narrow (d_low) bottleneck projection.  During training both
branches are computed and soft-mixed via Gumbel-Softmax; at inference the
hard sign of alpha picks one branch.

The token-level AlphaRouterPipe used by transformer_ddim is NOT used here —
alpha is internal to each attention layer.  The pipeline tuple format is
identical to the stock LSKV transformer: (hidden_states, attention_mask).
"""

import math
import torch
import torch.nn as nn

from .norms import get_norm
from megatron import mpu
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.activations import get_activation
from megatron.model.utils import exists, get_fusion_type
from megatron.model.positional_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_torch,
    AliBi,
    apply_rotary_pos_emb_torch_k,
    apply_rotary_pos_emb_k,
)
from megatron.model.fused_bias_dropout import (
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from megatron.model.utils import configure_sparse_attention

# Reuse unchanged pieces from the stock LSKV transformer.
from megatron.model.transformer import build_st_mask, ParallelMLP

# ---------------------------------------------------------------------------
# Global Gumbel temperature — updated by training.py after each iteration.
# ---------------------------------------------------------------------------
_LAR_TAU: float = 1.0


def update_lar_tau(tau: float) -> None:
    global _LAR_TAU
    _LAR_TAU = float(tau)


def compute_lar_tau(
    step: int,
    total_steps: int,
    tau_start: float = 1.0,
    tau_end: float = 0.1,
) -> float:
    """Exponential annealing: tau_start → tau_end over total_steps."""
    if total_steps <= 0 or step >= total_steps:
        return float(tau_end)
    frac = step / total_steps
    return float(tau_start) * (float(tau_end) / float(tau_start)) ** frac


def _layer_label(name: str) -> str:
    """Extract the first integer from a module name and return 'layerN'."""
    import re
    m = re.search(r"\d+", name)
    return f"layer{m.group()}" if m else name


def print_lar_selections(model) -> None:
    """Print the hard branch chosen by each LAR layer after training."""
    found = False
    for name, module in model.named_modules():
        if isinstance(module, ParallelSelfAttention):
            val = module.alpha.item()
            chosen = module.d_high if val > 0 else module.d_low
            print(f"[LAR] {_layer_label(name)}.alpha={val:+.4f} → d_down={chosen}", flush=True)
            found = True
    if not found:
        print("[LAR] no ParallelSelfAttention (LAR) modules found.", flush=True)


def collect_lar_entropy_reg(model) -> torch.Tensor:
    """Return sum_layers p*(1-p) where p=sigmoid(alpha/_LAR_TAU).

    Adding lambda * this term to the loss penalises p near 0.5 and pushes
    each layer toward committing to one branch.  Gradient is zero at p=0.5
    but the task-loss gradient provides the seed direction; regularisation
    then amplifies it as alpha grows away from zero.
    """
    total = None
    for module in model.modules():
        if isinstance(module, ParallelSelfAttention):
            a = module.alpha.float()
            p = torch.sigmoid(a / _LAR_TAU)
            reg = p * (1.0 - p)
            total = reg if total is None else total + reg
    if total is None:
        return torch.tensor(0.0)
    return total


def log_lar_alphas(model, iteration: int) -> None:
    """Log per-layer alpha and p=sigmoid(alpha/tau); tau shown to separate model vs temperature."""
    import math
    entries = []
    for name, module in model.named_modules():
        if isinstance(module, ParallelSelfAttention):
            a = module.alpha.item()
            p = 1.0 / (1.0 + math.exp(-a / _LAR_TAU))
            entries.append((_layer_label(name), a, p))
    if not entries:
        return
    vals = "  ".join(
        f"{label}(a={a:+.4f},p={p:.3f},{'H' if a > 0 else 'L'})"
        for label, a, p in entries
    )
    print(f"[LAR @ step {iteration} tau={_LAR_TAU:.3f}]  {vals}", flush=True)


# ---------------------------------------------------------------------------
# LAR attention
# ---------------------------------------------------------------------------

class ParallelSelfAttention(nn.Module):
    """LSKV self-attention with per-layer learnable capacity selection.

    Replaces the single down_up_proj with two independent branches
    (d_high and d_low) mixed via Gumbel-Softmax during training.
    """

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

        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = neox_args.apply_query_key_layer_scaling
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = neox_args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number

        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(neox_args.hidden_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            neox_args.hidden_size, neox_args.num_attention_heads
        )
        self.num_attention_heads_per_partition = mpu.divide(
            neox_args.num_attention_heads, world_size
        )
        self.pos_emb = neox_args.pos_emb

        # ---- LSKV window (ST path) ----
        self.lskv_st_window_size = neox_args.lskv_window_size
        if self.lskv_st_window_size is None or self.lskv_st_window_size < 0:
            raise ValueError(
                f"lskv_window_size must be >= 0, got {self.lskv_st_window_size}"
            )

        # ---- LAR: two projection branches + per-layer scalar ----
        self.d_high = int(getattr(neox_args, "lar_d_high", 128))
        self.d_low  = int(getattr(neox_args, "lar_d_low",  64))
        if not (0 < self.d_low < self.d_high):
            raise ValueError(
                f"lar_d_high ({self.d_high}) must be > lar_d_low ({self.d_low}) > 0"
            )

        # Initialised to 0 → 50/50 soft mix at the start of training.
        self.alpha = nn.Parameter(torch.zeros(1))

        self.down_up_proj_high = nn.Sequential(
            nn.Linear(neox_args.hidden_size, self.d_high, bias=False),
            nn.Linear(self.d_high, neox_args.hidden_size, bias=False),
        )
        self.down_up_proj_low = nn.Sequential(
            nn.Linear(neox_args.hidden_size, self.d_low, bias=False),
            nn.Linear(self.d_low, neox_args.hidden_size, bias=False),
        )
        print(
            f"[LAR] layer={layer_number} d_high={self.d_high} d_low={self.d_low} "
            f"window={self.lskv_st_window_size}",
            flush=True,
        )

        # ---- Standard QKV projection ----
        self.query_key_value = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=3 * neox_args.hidden_size,
            gather_output=False,
            init_method=init_method,
        )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        self.rpe = rpe

        if self.pos_emb == "alibi":
            self.alibi_embed = AliBi(
                neox_args.num_attention_heads,
                neox_args.model_parallel_size,
                mpu.get_model_parallel_rank(),
            )

        if rotary:
            if neox_args.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert neox_args.rotary_pct < 1
                self.rotary_ndims = int(
                    self.hidden_size_per_attention_head * neox_args.rotary_pct
                )
            dim = (
                self.rotary_ndims
                if self.rotary_ndims is not None
                else self.hidden_size_per_attention_head
            )
            self.rotary_emb = RotaryEmbedding(
                dim, base=neox_args.rotary_emb_base, precision=neox_args.params_dtype
            )
        else:
            self.rotary_emb = None

        self.attention_type = neox_args.attention_config[layer_number]
        self.use_flash_attention = self.attention_type == "flash"
        self.sparse = self.attention_type != "global" and not self.use_flash_attention
        if self.sparse:
            self.sparse_attn = configure_sparse_attention(
                neox_args,
                self.attention_type,
                self.num_attention_heads_per_partition,
                mpu=mpu,
            )
        else:
            if self.use_flash_attention:
                from megatron.model.flash_attention import (
                    flash_attn_unpadded_qkvpacked_func,
                )
                self.flash_attention_function = flash_attn_unpadded_qkvpacked_func
            else:
                self.scale_mask_softmax = FusedScaleMaskSoftmax(
                    input_in_fp16=self.fp16,
                    input_in_bf16=self.bf16,
                    fusion_type=get_fusion_type(neox_args),
                    mask_func=self.attention_mask_func,
                    softmax_in_fp32=self.attention_softmax_in_fp32,
                    scale=coeff,
                )
            self.dropout_p = neox_args.attention_dropout
            self.attention_dropout = nn.Dropout(self.dropout_p)

        self.dense = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
        )

    # ------------------------------------------------------------------
    # LAR mixing logic
    # ------------------------------------------------------------------

    def _lt_hidden(self, hidden_states):
        h_high = self.down_up_proj_high(hidden_states)
        h_low  = self.down_up_proj_low(hidden_states)
        if self.training:
            p = torch.sigmoid(self.alpha.to(hidden_states.dtype) / _LAR_TAU)
            return p * h_high + (1.0 - p) * h_low
        else:
            return h_high if self.alpha.item() > 0 else h_low

    # ------------------------------------------------------------------
    # Attention (identical to transformer.py)
    # ------------------------------------------------------------------

    def attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        layer_past,
        attention_mask,
        lt_key_layer,
        lt_value_layer,
    ):
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        query_layer  = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer    = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        lt_key_layer = lt_key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),
            key_layer.transpose(0, 1).transpose(1, 2),
            beta=0.0, alpha=(1.0 / self.norm_factor),
        )

        lt_matmul_result = torch.empty_like(matmul_result)
        lt_matmul_result = torch.baddbmm(
            lt_matmul_result,
            query_layer.transpose(0, 1),
            lt_key_layer.transpose(0, 1).transpose(1, 2),
            beta=0.0, alpha=(1.0 / self.norm_factor),
        )

        attention_scores = matmul_result.view(*output_size)

        SQ, SK = output_size[2], output_size[3]
        cache_position = torch.arange(SK - SQ, SK, device=attention_scores.device)
        st_mask_bool = build_st_mask(
            output_size,
            self.lskv_st_window_size,
            cache_position=cache_position,
            device=attention_scores.device,
            dtype=attention_scores.dtype,
        )
        attention_scores = torch.where(
            st_mask_bool, attention_scores, lt_matmul_result.view(*output_size)
        )

        if self.use_cache:
            with torch.no_grad():
                attention_mask = attention_mask[
                    ..., : attention_scores.size(3), : attention_scores.size(3)
                ]

        if exists(self.rpe):
            attention_scores += self.rpe(query_layer.size(0), key_layer.size(0))

        if self.pos_emb == "alibi":
            attention_scores = self.alibi_embed(attention_scores)

        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )
        value_layer    = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
        lt_value_layer = lt_value_layer.view(lt_value_layer.size(0), output_size[0] * output_size[1], -1)
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        st_mask_f = st_mask_bool.to(attention_probs.dtype).view(1, SQ, SK)
        context_layer = (
            torch.bmm(attention_probs * st_mask_f,          value_layer.transpose(0, 1))
          + torch.bmm(attention_probs * (1.0 - st_mask_f),  lt_value_layer.transpose(0, 1))
        )
        return context_layer.view(*output_size)

    def forward(self, hidden_states, attention_mask, layer_past=None):
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        lt_hidden_states = self._lt_hidden(hidden_states)   # ← LAR mixing
        lt_mixed_x_layer, _ = self.query_key_value(lt_hidden_states)

        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        lt_mixed_x_layer = lt_mixed_x_layer.view(*new_tensor_shape)

        (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)
        _, lt_key_layer, lt_value_layer       = mpu.split_tensor_along_last_dim(lt_mixed_x_layer, 3)

        if exists(self.rotary_emb):
            if exists(self.rotary_ndims):
                query_rot,  query_pass  = query_layer[..., :self.rotary_ndims],  query_layer[..., self.rotary_ndims:]
                key_rot,    key_pass    = key_layer[..., :self.rotary_ndims],    key_layer[..., self.rotary_ndims:]
                lt_key_rot, lt_key_pass = lt_key_layer[..., :self.rotary_ndims], lt_key_layer[..., self.rotary_ndims:]
            else:
                query_rot, key_rot, lt_key_rot = query_layer, key_layer, lt_key_layer

            apply_rotary_fn    = apply_rotary_pos_emb_torch    if self.bf16 else apply_rotary_pos_emb
            lt_apply_rotary_fn = apply_rotary_pos_emb_torch_k  if self.bf16 else apply_rotary_pos_emb_k

            seq_len = key_layer.shape[0]
            offset  = 0
            if exists(layer_past) and layer_past.numel() > 0:
                offset   = layer_past[0].shape[0]
                seq_len += offset
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            query_layer, key_layer = apply_rotary_fn(query_rot, key_rot, cos, sin, offset=offset)

            lt_cos, lt_sin = self.rotary_emb(lt_value_layer, seq_len=seq_len)
            lt_key_layer   = lt_apply_rotary_fn(query_rot, lt_key_rot, lt_cos, lt_sin, offset=offset)

            if exists(self.rotary_ndims):
                query_layer  = torch.cat((query_layer,  query_pass),  dim=-1)
                key_layer    = torch.cat((key_layer,    key_pass),    dim=-1)
                lt_key_layer = torch.cat((lt_key_layer, lt_key_pass), dim=-1)

        if exists(layer_past) and layer_past.numel() > 0:
            past_key, past_value, past_lt_key, past_lt_value = layer_past
            key_layer      = torch.cat((past_key.type_as(key_layer),          key_layer),      dim=0)
            value_layer    = torch.cat((past_value.type_as(value_layer),       value_layer),    dim=0)
            lt_key_layer   = torch.cat((past_lt_key.type_as(lt_key_layer),     lt_key_layer),   dim=0)
            lt_value_layer = torch.cat((past_lt_value.type_as(lt_value_layer), lt_value_layer), dim=0)

        if self.use_cache:
            present = torch.stack((key_layer, value_layer, lt_key_layer, lt_value_layer))

        if self.use_flash_attention:
            raise NotImplementedError("LAR attention does not support flash attention.")
        elif not self.sparse:
            context_layer = self.attention(
                query_layer, key_layer, value_layer,
                layer_past, attention_mask, lt_key_layer, lt_value_layer,
            )
        else:
            raise NotImplementedError("LAR attention does not support sparse attention.")

        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        context_layer = context_layer.view(
            context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        )

        output, bias = self.dense(context_layer)
        if self.use_cache:
            output = [output, present]
        return output, bias


# ---------------------------------------------------------------------------
# Transformer layer and pipeline wrapper (same structure as transformer.py)
# ---------------------------------------------------------------------------

class ParallelTransformerLayer(nn.Module):

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
    ):
        super().__init__()
        self.layer_number = layer_number

        norm, eps = get_norm(neox_args)
        self.input_layernorm        = norm(neox_args.hidden_size, eps=eps)
        self.post_attention_layernorm = norm(neox_args.hidden_size, eps=eps)
        self.use_cache              = use_cache
        self.hidden_dropout         = neox_args.hidden_dropout
        self.bias_dropout_fusion    = neox_args.bias_dropout_fusion
        self.gpt_j_residual         = neox_args.gpt_j_residual
        self.gpt_j_tied             = neox_args.gpt_j_tied

        if self.gpt_j_residual:
            self.reduce = mpu.mappings.reduce_from_model_parallel_region

        self.attention = ParallelSelfAttention(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            rpe=rpe,
            use_cache=self.use_cache,
            rotary=rotary,
            parallel_output=self.gpt_j_residual,
        )

        self.mlp = ParallelMLP(
            neox_args=neox_args,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            parallel_output=self.gpt_j_residual,
        )

        self.layer_past = None

    def _get_bias_dropout(self):
        if self.bias_dropout_fusion:
            fn = bias_dropout_add_fused_train if self.training else bias_dropout_add_fused_inference
        else:
            fn = get_bias_dropout_add(self.training)
        return fn

    def forward(self, x, attention_mask, layer_past=None):
        layer_past      = layer_past if layer_past is not None else self.layer_past
        bias_dropout_fn = self._get_bias_dropout()

        if self.gpt_j_residual:
            residual = x
            if self.gpt_j_tied:
                x       = self.input_layernorm(x)
                x1, x2 = x, x
            else:
                x1, x2 = self.input_layernorm(x), self.post_attention_layernorm(x)

            attention_output, attention_bias = self.attention(x1, attention_mask, layer_past=layer_past)
            if self.use_cache:
                attention_output, presents = attention_output
                self.layer_past = presents

            with torch.enable_grad():
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(attention_output),
                    residual=None,
                    prob=self.hidden_dropout,
                )

            mlp_output, mlp_bias = self.mlp(x2)
            with torch.enable_grad():
                output = bias_dropout_fn(
                    mlp_output,
                    bias=mlp_bias.expand_as(mlp_output),
                    residual=attention_output,
                    prob=self.hidden_dropout,
                )
            output = residual + self.reduce(output)
        else:
            residual = x
            attention_output, attention_bias = self.attention(
                self.input_layernorm(x), attention_mask, layer_past=layer_past
            )
            if self.use_cache:
                attention_output, presents = attention_output
                self.layer_past = presents
            with torch.enable_grad():
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(residual),
                    residual=residual,
                    prob=self.hidden_dropout,
                )
            mlp_output, mlp_bias = self.mlp(self.post_attention_layernorm(attention_output))
            with torch.enable_grad():
                output = bias_dropout_fn(
                    mlp_output,
                    bias=mlp_bias.expand_as(attention_output),
                    residual=attention_output,
                    prob=self.hidden_dropout,
                )
        return output


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Pipeline-compatible wrapper — passes (hidden_states, attention_mask) through."""

    def forward(self, args):
        assert len(args) == 2, (
            "ParallelTransformerLayerPipe (LAR) expects 2 args: (hidden_states, attention_mask)"
        )
        hidden_states, attention_mask = args
        return super().forward(hidden_states, attention_mask), attention_mask
