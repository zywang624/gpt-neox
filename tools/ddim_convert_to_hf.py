# ddim_convert_to_hf.py
#
# Convert a DDIM / alpha-routed LSKV NeoX checkpoint to a HuggingFace
# GPTNeoXForCausalLM model.
#
# Two pipeline layouts are supported, picked by inspecting the yml:
#   (A) Learned router (no `alpha_lookup_path` in yml):
#          layer_00          → word_embeddings
#          layer_01          → router (the alpha router)
#          layer_02          → dummy _pre_transformer_block_alpha (no file)
#          layer_03 ..       → transformer layers (i + 3 for the i-th layer)
#          layer_(N+4)       → final layernorm (`norm`)
#          layer_(N+5)       → output linear (`final_linear`)
#   (B) Frozen alpha lookup (yml has `alpha_lookup_path` OR
#       `alpha_lookup_random_init: true`):
#          layer_00          → embedding + alpha_lookup buffer
#                              (EmbeddingPipeWithFrozenAlpha replaces the
#                               embed + router pair, so everything below
#                               shifts back by 1)
#          layer_01          → dummy _pre_transformer_block_alpha (no file)
#          layer_02 ..       → transformer layers (i + 2)
#          layer_(N+3)       → final layernorm
#          layer_(N+4)       → output linear
#
# Each attention layer has two parallel projections, bias=False and no
# activation: `down_up_proj` (full) and `down_up_proj_half` (half). Neither
# is MP-sharded.
#
# All alpha-* HF config fields are read verbatim from the yml — no CLI
# overrides, so HF inference reproduces training-time forward exactly.

import os, sys, yaml, argparse
from tqdm import tqdm
import torch
from ddim_modeling import GPTNeoXConfig, GPTNeoXForCausalLM

from typing import List

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from megatron.tokenizer import build_tokenizer


def load_partitions(
    input_checkpoint_path, mp_partitions, layer_idx
) -> List[torch.Tensor]:
    loaded_tp_ranks = [
        torch.load(
            os.path.join(
                input_checkpoint_path,
                f"layer_{layer_idx:02}-model_{i:02}-model_states.pt",
            )
        )
        for i in range(mp_partitions)
    ]
    return loaded_tp_ranks


def get_key(loaded_config, key, default=None):
    key = key.replace("_", "-")
    try:
        return loaded_config[key]
    except KeyError:
        key = key.replace("-", "_")
        try:
            return loaded_config[key]
        except KeyError:
            return default


def create_config(neox_config):
    class TokenizerArgs:
        def __init__(self, neox_config):
            self.make_vocab_size_divisible_by = get_key(
                neox_config, "make-vocab-size-divisible-by", default=128
            )
            self.model_parallel_size = get_key(neox_config, "model-parallel-size")
            self.vocab_file = get_key(neox_config, "vocab-file")
            self.merge_file = get_key(neox_config, "merge-file")
            self.tokenizer_type = get_key(neox_config, "tokenizer-type")

            self.rank = 0

    args = TokenizerArgs(neox_config)
    tokenizer = build_tokenizer(args)
    try:
        pad_token = tokenizer.pad
    except:
        pad_token = 1

    use_tied_lns = get_key(neox_config, "gpt-j-tied", False)
    if use_tied_lns:
        raise NotImplementedError(
            "Tied layernorm (gpt-j-tied) is not supported by HF GPT-NeoX."
        )

    hf_config = GPTNeoXConfig(
        vocab_size=args.padded_vocab_size,
        hidden_size=get_key(neox_config, "hidden-size"),
        num_hidden_layers=get_key(neox_config, "num-layers"),
        num_attention_heads=get_key(neox_config, "num-attention-heads"),
        intermediate_size=(get_key(neox_config, "hidden-size") * 4),
        hidden_act=get_key(neox_config, "activation", default="gelu"),
        rotary_pct=get_key(neox_config, "rotary-pct", default=1.0),
        rotary_emb_base=get_key(neox_config, "rotary-emb-base", default=10000),
        max_position_embeddings=get_key(neox_config, "max-position-embeddings"),
        initializer_range=get_key(neox_config, "init-method-std", 0.02),
        layer_norm_eps=get_key(neox_config, "layernorm-epsilon", 1e-5),
        use_cache=True,
        bos_token_id=tokenizer.eod,
        eos_token_id=tokenizer.eod,
        tie_word_embeddings=(not get_key(neox_config, "no-weight-tying", False)),
        use_parallel_residual=get_key(neox_config, "gpt-j-residual", False),
        lskv_st_window_size=get_key(neox_config, "lskv_window_size", None),
        lskv_bottleneck_dim=get_key(neox_config, "lskv_bottleneck_dim", None),
        # All alpha-* flags mirror yml verbatim. `use_alpha_lookup` is derived
        # from either yml field that triggers the frozen-lookup pipeline
        # layout (loaded from .npy OR random-init Bernoulli(0.5)); both end
        # up storing the table as a buffer at layer_00, so HF only needs to
        # know "is this layout the frozen one" — the path itself no longer
        # matters once the buffer is in the checkpoint.
        use_alpha_lookup=(
            bool(get_key(neox_config, "alpha_lookup_path", None))
            or bool(get_key(neox_config, "alpha_lookup_random_init", False))
        ),
        alpha_hard_routing=bool(get_key(neox_config, "alpha_hard_routing", False)),
        alpha_ste=bool(get_key(neox_config, "alpha_ste", False)),
        alpha_hard_inference=bool(get_key(neox_config, "alpha_hard_inference", False)),
        alpha_hard_threshold=float(get_key(neox_config, "alpha_hard_threshold", 0.5)),
    )
    return hf_config


def convert(input_checkpoint_path, loaded_config, output_checkpoint_path):
    hf_config = create_config(loaded_config)

    hf_model = GPTNeoXForCausalLM(hf_config).half()

    mp_partitions = get_key(loaded_config, "model-parallel-size")
    num_layers = get_key(loaded_config, "num-layers")

    # Frozen-lookup mode collapses (embed + router) into layer_00 and shifts
    # everything below by -1.
    is_frozen = hf_config.use_alpha_lookup
    tx_offset = 2 if is_frozen else 3
    norm_idx = num_layers + (3 if is_frozen else 4)
    out_idx = num_layers + (4 if is_frozen else 5)

    ### Embedding layer (layer_00) — and the alpha_lookup buffer if frozen ###
    loaded_tp_ranks = load_partitions(input_checkpoint_path, mp_partitions, 0)
    hf_model.gpt_neox.embed_in.load_state_dict(
        {
            "weight": torch.cat(
                [t["word_embeddings.weight"] for t in loaded_tp_ranks], dim=0
            )
        }
    )

    assert (
        hf_config.vocab_size == hf_model.gpt_neox.embed_in.weight.shape[0]
    ), f"ERROR: calculated vocab size {hf_config.vocab_size} != embed param size {hf_model.gpt_neox.embed_in.weight.shape[0]}"

    if is_frozen:
        # alpha_lookup is a buffer, not MP-sharded; take rank 0.
        alpha_lookup = loaded_tp_ranks[0]["alpha_lookup"]
        assert alpha_lookup.shape == (hf_config.vocab_size,), (
            f"alpha_lookup shape {tuple(alpha_lookup.shape)} != "
            f"(vocab_size={hf_config.vocab_size},)"
        )
        # Mirror the saved dtype exactly so the (alpha > 0.5) threshold reads
        # the same numbers training did. The HF model was .half()-ed above,
        # but we overwrite the buffer here with whatever dtype training saved
        # (typically fp16 with NeoX fp16:true, but fp32 is also possible if
        # the deepspeed path skipped buffer casting). Use-time cast in
        # GPTNeoXModel.forward then matches the embedding dtype.
        hf_model.gpt_neox.alpha_lookup = alpha_lookup.clone()
    else:
        ### Alpha router (layer_01) — not MP-sharded, take rank 0 ###
        loaded_tp_ranks = load_partitions(input_checkpoint_path, mp_partitions, 1)
        hf_model.gpt_neox.alpha_router.load_state_dict(
            {
                "weight": loaded_tp_ranks[0]["router.weight"],
                "bias":   loaded_tp_ranks[0]["router.bias"],
            }
        )
    del loaded_tp_ranks

    ### Transformer layers (layer_(i + tx_offset)) ###
    for layer_i in tqdm(range(num_layers)):
        hf_layer = hf_model.gpt_neox.layers[layer_i]

        # Offset depends on whether AlphaRouterPipe occupied layer_01.
        loaded_tp_ranks = load_partitions(
            input_checkpoint_path, mp_partitions, layer_i + tx_offset
        )

        state_dict = {}
        for key in [
            "attention.dense.weight",
            "mlp.dense_4h_to_h.weight",
        ]:
            state_dict[key] = torch.cat([t[key] for t in loaded_tp_ranks], dim=1)

        # average layernorm stats over mp ranks
        for key in [
            "input_layernorm.weight",
            "input_layernorm.bias",
            "post_attention_layernorm.weight",
            "post_attention_layernorm.bias",
        ]:
            state_dict[key] = (sum([t[key] for t in loaded_tp_ranks])) / len(
                loaded_tp_ranks
            )

        # LinearWithTPMerge (column-parallel) — concat on output dim
        for key in [
            "mlp.dense_h_to_4h.weight",
            "mlp.dense_h_to_4h.bias",
            "attention.query_key_value.weight",
            "attention.query_key_value.bias",
        ]:
            state_dict[key] = torch.cat([t[key] for t in loaded_tp_ranks], dim=0)

        # LinearWithTPSplitBias — sum bias across ranks
        for key in [
            "mlp.dense_4h_to_h.bias",
            "attention.dense.bias",
        ]:
            state_dict[key] = sum([t[key] for t in loaded_tp_ranks])

        # DDIM-specific: both down_up_proj and down_up_proj_half are NOT
        # MP-sharded and have no biases / no activation (indices 0 and 1).
        for key in [
            "attention.down_up_proj.0.weight",
            "attention.down_up_proj.1.weight",
            "attention.down_up_proj_half.0.weight",
            "attention.down_up_proj_half.1.weight",
        ]:
            state_dict[key] = loaded_tp_ranks[0][key]

        hf_layer.load_state_dict(state_dict)

    # Final layer norm — layer_(num_layers + 3) for frozen, +4 for learned.
    loaded_tp_ranks = load_partitions(
        input_checkpoint_path, mp_partitions, norm_idx
    )
    hf_model.gpt_neox.final_layer_norm.load_state_dict(
        {
            "weight": (sum([t["norm.weight"] for t in loaded_tp_ranks]))
            / len(loaded_tp_ranks),
            "bias": (sum([t["norm.bias"] for t in loaded_tp_ranks]))
            / len(loaded_tp_ranks),
        }
    )
    del loaded_tp_ranks

    # Output embedding — layer_(num_layers + 4) for frozen, +5 for learned.
    loaded_tp_ranks = load_partitions(
        input_checkpoint_path, mp_partitions, out_idx
    )
    hf_model.embed_out.load_state_dict(
        {
            "weight": torch.cat(
                [t["final_linear.weight"] for t in loaded_tp_ranks], dim=0
            ),
        }
    )
    del loaded_tp_ranks

    return hf_model


if __name__ == "__main__":
    from huggingface_hub import create_repo, HfApi

    parser = argparse.ArgumentParser(
        description="Merge MP partitions and convert DDIM (alpha-routed) NeoX checkpoint to HF."
    )
    parser.add_argument("--input_dir", type=str,
                        help="Path to NeoX checkpoint, e.g. /path/to/model/global_step143000")
    parser.add_argument("--config_file", type=str,
                        help="Path to NeoX yml config for the checkpoint.")
    parser.add_argument("--output_dir", type=str,
                        help="Output dir for the HF model + tokenizer.")
    parser.add_argument("--upload", action="store_true",
                        help="Set to true in order to upload to the HF Hub directly.")
    args = parser.parse_args()

    with open(args.config_file) as f:
        loaded_config = yaml.full_load(f)

    hf_model = convert(
        args.input_dir,
        loaded_config,
        args.output_dir,
    )

    hf_model.save_pretrained(args.output_dir)

    tokenizer_type = get_key(loaded_config, "tokenizer-type")
    if tokenizer_type == "HFTokenizer":
        print(f"saving tokenizer from file {get_key(loaded_config, 'vocab-file')}")
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=get_key(loaded_config, "vocab-file")
        )
        print("loaded tokenizer: ", tokenizer)
        tokenizer.save_pretrained(args.output_dir)
        print("tokenizer saved!")

    if args.upload:
        repo_name = input("Provide a repository name for the HF Hub: ")
        create_repo(repo_name, repo_type="model", private=False, use_auth_token=True)
        api = HfApi()
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=repo_name,
            repo_type="model",
        )
