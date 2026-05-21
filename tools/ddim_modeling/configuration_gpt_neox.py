# coding=utf-8
# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
"""GPTNeoX (DDIM / alpha-routed LSKV) model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class GPTNeoXConfig(PretrainedConfig):
    """GPTNeoX config extended for DDIM / alpha-routed LSKV.

    Adds knobs on top of the vanilla LSKV config (all mirror the NeoX yml
    fields with the same name; HF inference reproduces training-time forward):
      * lskv_bottleneck_dim: bottleneck dim of the *full* down-up projection
      * lskv_st_window_size: short-term attention window
      * use_alpha_lookup: if True, alpha comes from a frozen per-token-id
        lookup table (registered buffer `alpha_lookup`); if False, alpha is
        produced by a learned `alpha_router = sigmoid(Linear(embed))`. Mirrors
        NeoX yml `alpha_lookup_path` being set.
      * alpha_hard_routing: if True, mix is always thresholded ((alpha > 0.5)),
        both training and inference. Mirrors yml `alpha_hard_routing`.
      * alpha_ste: if True, training used the STE (forward = hard); at HF eval
        we therefore also use hard mix. Mirrors yml `alpha_ste`.
      * alpha_hard_inference: if True, mix uses (alpha > 0.5) at eval time
        when neither alpha_hard_routing nor alpha_ste is set.
      * alpha_hard_threshold: threshold used by all hard-forward paths (STE,
        alpha_hard_routing, alpha_hard_inference) to binarize alpha:
        `mix = (alpha > alpha_hard_threshold)`. Mirrors yml
        `alpha_hard_threshold`. Default 0.5 reproduces the historical
        hardcoded behavior.
    """

    model_type = "gpt_neox"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.attention.query_key_value": "colwise",
        "layers.*.attention.dense": "rowwise",
        "layers.*.mlp.dense_h_to_4h": "colwise",
        "layers.*.mlp.dense_4h_to_h": "rowwise",
    }
    base_model_pp_plan = {
        "embed_in": (["input_ids"], ["inputs_embeds"]),
        "emb_dropout": (["inputs_embeds"], ["hidden_states"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "final_layer_norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=50432,
        hidden_size=6144,
        num_hidden_layers=44,
        num_attention_heads=64,
        intermediate_size=24576,
        hidden_act="gelu",
        rotary_pct=0.25,
        rotary_emb_base=10000,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        classifier_dropout=0.1,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_parallel_residual=True,
        rope_scaling=None,
        attention_bias=True,
        lskv_bottleneck_dim=None,
        lskv_st_window_size=None,
        use_alpha_lookup=False,
        alpha_hard_routing=False,
        alpha_ste=False,
        alpha_hard_inference=False,
        alpha_hard_threshold=0.5,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rotary_pct = rotary_pct
        self.partial_rotary_factor = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.rope_theta = rotary_emb_base
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_parallel_residual = use_parallel_residual
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.lskv_bottleneck_dim = lskv_bottleneck_dim
        self.lskv_st_window_size = lskv_st_window_size
        self.use_alpha_lookup = bool(use_alpha_lookup)
        self.alpha_hard_routing = bool(alpha_hard_routing)
        self.alpha_ste = bool(alpha_ste)
        self.alpha_hard_inference = bool(alpha_hard_inference)
        self.alpha_hard_threshold = float(alpha_hard_threshold)
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisible by the number of attention heads! Make sure to update them!"
            )


__all__ = ["GPTNeoXConfig"]
