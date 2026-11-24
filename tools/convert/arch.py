"""What the converter knows about each architecture: config fields, tensor roles, tokenizer and chat template."""

import ast
import re

QWEN2_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# The Llama 3 split is the Qwen one with numbers in pieces of up to three digits.
LLAMA3_PATTERN = QWEN2_PATTERN.replace(r"|\p{N}|", r"|\p{N}{1,3}|")


class Llama:
    """Llama family decoders, SmolLM2 included."""

    name = "llama"

    def config(self, hf):
        heads = hf["num_attention_heads"]
        return {
            "hidden": hf["hidden_size"],
            "layers": hf["num_hidden_layers"],
            "heads": heads,
            "kvHeads": hf.get("num_key_value_heads", heads),
            "headDim": hf.get("head_dim") or hf["hidden_size"] // heads,
            "intermediate": hf["intermediate_size"],
            "vocab": hf["vocab_size"],
            "ropeTheta": hf.get("rope_theta", 10000.0),
            "rmsEps": hf["rms_norm_eps"],
            "maxPositions": hf["max_position_embeddings"],
            "tieEmbeddings": bool(hf.get("tie_word_embeddings", False)),
            "activation": hf["hidden_act"],
        }

    LAYER = {
        "input_layernorm.weight": ("attn_norm", "vector"),
        "post_attention_layernorm.weight": ("mlp_norm", "vector"),
        "self_attn.q_proj.weight": ("q", "matrix"),
        "self_attn.k_proj.weight": ("k", "matrix"),
        "self_attn.v_proj.weight": ("v", "matrix"),
        "self_attn.o_proj.weight": ("o", "matrix"),
        "mlp.gate_proj.weight": ("gate", "matrix"),
        "mlp.up_proj.weight": ("up", "matrix"),
        "mlp.down_proj.weight": ("down", "matrix"),
    }

    def role(self, key, config):
        """(canonical name, kind) of a checkpoint tensor, None for tensors the runtime does not read."""
        if key == "model.embed_tokens.weight":
            return "embed", "matrix"
        if key == "model.norm.weight":
            return "final_norm", "vector"
        if key == "lm_head.weight":
            return (None if config["tieEmbeddings"] else ("head", "matrix"))
        m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", key)
        if m and m.group(2) in self.LAYER:
            canonical, kind = self.LAYER[m.group(2)]
            return f"layers.{m.group(1)}.{canonical}", kind
        return None

    def required(self, config):
        names = {"embed", "final_norm"}
        if not config["tieEmbeddings"]:
            names.add("head")
        for i in range(config["layers"]):
            for canonical, _ in self.LAYER.values():
                names.add(f"layers.{i}.{canonical}")
        return names

    def prepare(self, canonical, array, config):
        """The tensor as the runtime reads it, after any change of layout the family needs."""
        return array

    def tokenizer(self, tok, tok_config, hf, specials):
        pre = tok["pre_tokenizer"]
        steps = pre["pretokenizers"] if pre["type"] == "Sequence" else [pre]
        digits = any(s["type"] == "Digits" and s.get("individual_digits") for s in steps)
        byte_level = [s for s in steps if s["type"] == "ByteLevel"]
        splits = [s for s in steps if s["type"] == "Split"]
        kind = "gpt2"
        pattern = splits[0]["pattern"].get("Regex") if splits else None
        if pattern in (QWEN2_PATTERN, LLAMA3_PATTERN) and splits[0]["behavior"] == "Isolated":
            kind, digits = ("qwen2" if pattern == QWEN2_PATTERN else "llama3"), True
        elif not byte_level or not byte_level[0].get("use_regex", True):
            raise SystemExit("only byte level BPE with the GPT-2, Qwen or Llama 3 split is supported")
        eos = hf.get("eos_token_id", [])
        eos = eos if isinstance(eos, list) else [eos]
        return {
            "kind": "bytebpe",
            "pretokenizer": kind,
            "splitDigits": digits,
            "addPrefixSpace": bool(byte_level[0].get("add_prefix_space", False)),
            "ignoreMerges": bool(tok["model"].get("ignore_merges")),
            "specials": specials,
            "eos": eos,
            "bos": self.leading_bos(tok),
            "addBos": self.leading_bos(tok) is not None,
        }

    @staticmethod
    def leading_bos(tok):
        """The id a post processor puts in front of every text, None when it adds none."""
        post = tok.get("post_processor") or {}
        for step in post.get("processors", [post]):
            single = step.get("single") or []
            if step.get("type") == "TemplateProcessing" and single and "SpecialToken" in single[0]:
                return step["special_tokens"][single[0]["SpecialToken"]["id"]]["ids"][0]
        return None

    def chat(self, template, tok_config=None):
        """The declarative ChatML form, checked against the Jinja template by tools/convert/check_chat.py."""
        if "<|start_header_id|>" in template:
            return self.llama3_chat(template, tok_config)
        if "<|im_start|>" not in template:
            raise SystemExit("only the ChatML and Llama 3 templates are supported")
        default = re.search(r"messages\[0\]\['role'\] != 'system' %}\{\{ '<\|im_start\|>system\n(.*?)<\|im_end\|>", template, re.S)
        if default is None:
            default = re.search(r"\{\{-?\s*'<\|im_start\|>system\\n([^'+]*?)<\|im_end\|>\\n'\s*\}\}", template)
        spec = {"kind": "chatml", "defaultSystem": default.group(1) if default else None}
        if re.match(r"\s*\{\{-?\s*bos_token", template):
            spec["opening"] = tok_config["bos_token"]
        if "enable_thinking is defined and enable_thinking is false" in template:
            spec["generationPrefix"] = "<think>\n\n</think>\n\n"
        return spec


    @staticmethod
    def llama3_chat(template, tok_config):
        """Llama 3 header markers: the beginning of sequence token, a system turn with the template's default message
        and `<|eot_id|>` after every turn."""
        default = re.search(r'set system_message = ("(?:[^"\\]|\\.)*")', template)
        if default is None or not re.match(r"\s*\{\{-?\s*bos_token", template):
            raise SystemExit("the Llama 3 template has no default system message or does not open with bos_token")
        return {
            "kind": "llama3",
            "defaultSystem": ast.literal_eval(default.group(1)),
            "opening": tok_config["bos_token"],
        }


class Qwen2(Llama):
    """Qwen2 and Qwen2.5: biases on the q, k and v projections."""

    name = "qwen2"
    LAYER = dict(
        Llama.LAYER,
        **{
            "self_attn.q_proj.bias": ("q_bias", "vector"),
            "self_attn.k_proj.bias": ("k_bias", "vector"),
            "self_attn.v_proj.bias": ("v_bias", "vector"),
        },
    )


class Qwen3(Llama):
    """Qwen3: a norm on each head's q and k, no biases."""

    name = "qwen3"
    LAYER = dict(
        Llama.LAYER,
        **{
            "self_attn.q_norm.weight": ("q_norm", "vector"),
            "self_attn.k_norm.weight": ("k_norm", "vector"),
        },
    )


class Gemma3(Llama):
    """Gemma 3 text models: four norms per layer, a norm on each head's q and k, local and global layers, and a
    SentencePiece style BPE tokenizer."""

    name = "gemma3"
    LAYER = dict(
        Llama.LAYER,
        **{
            "post_attention_layernorm.weight": ("post_attn_norm", "vector"),
            "pre_feedforward_layernorm.weight": ("mlp_norm", "vector"),
            "post_feedforward_layernorm.weight": ("post_mlp_norm", "vector"),
            "self_attn.q_norm.weight": ("q_norm", "vector"),
            "self_attn.k_norm.weight": ("k_norm", "vector"),
        },
    )

    def config(self, hf):
        if hf.get("rope_scaling"):
            raise SystemExit("rope_scaling is not supported")
        if hf.get("attn_logit_softcapping") or hf.get("final_logit_softcapping"):
            raise SystemExit("logit soft capping is not supported")
        # Older checkpoints have rope_theta and rope_local_base_freq, newer ones rope_parameters per layer kind.
        params = hf.get("rope_parameters") or {}
        theta = (params.get("full_attention") or {}).get("rope_theta", hf.get("rope_theta", 1000000.0))
        local = (params.get("sliding_attention") or {}).get("rope_theta", hf.get("rope_local_base_freq", 10000.0))
        config = super().config(
            dict(
                hf,
                rope_theta=theta,
                tie_word_embeddings=hf.get("tie_word_embeddings", True),
                hidden_act=hf.get("hidden_activation", hf.get("hidden_act")),
            )
        )
        pattern = hf.get("sliding_window_pattern", 6)
        kinds = hf.get("layer_types")
        if kinds:
            pattern = kinds.index("full_attention") + 1
            want = ["full_attention" if (i + 1) % pattern == 0 else "sliding_attention" for i in range(len(kinds))]
            if kinds != want:
                raise SystemExit("the layer types are not a repeating pattern of local layers and one global layer")
        config.update(
            {
                "slidingWindow": hf["sliding_window"],
                "slidingPattern": pattern,
                "ropeLocalTheta": local,
                "attnScalar": hf.get("query_pre_attn_scalar", config["headDim"]),
            }
        )
        return config

    def tokenizer(self, tok, tok_config, hf, specials):
        model = tok["model"]
        norm = tok.get("normalizer") or {}
        if model["type"] != "BPE" or not model.get("byte_fallback") or model.get("ignore_merges"):
            raise SystemExit("only a BPE tokenizer with byte fallback is supported")
        if norm.get("type") != "Replace" or norm["pattern"].get("String") != " " or norm["content"] != "\u2581":
            raise SystemExit("only the space to U+2581 normalizer is supported")
        post = (tok.get("post_processor") or {}).get("single") or []
        add_bos = bool(post) and "SpecialToken" in post[0]
        eos = hf.get("eos_token_id", [])
        return {
            "kind": "spm",
            "specials": specials,
            "eos": eos if isinstance(eos, list) else [eos],
            "bos": hf.get("bos_token_id"),
            "addBos": add_bos,
        }

    def chat(self, template, tok_config=None):
        if "<start_of_turn>" not in template:
            raise SystemExit("only the Gemma turn template is supported")
        return {"kind": "gemma", "opening": (tok_config or {}).get("bos_token", "<bos>")}


class Lfm2(Llama):
    """LFM2 and LFM2.5: gated short convolution layers between a few attention layers, a norm on each head's q and k."""

    name = "lfm2"
    LAYER = {
        "operator_norm.weight": ("attn_norm", "vector"),
        "ffn_norm.weight": ("mlp_norm", "vector"),
        "feed_forward.w1.weight": ("gate", "matrix"),
        "feed_forward.w3.weight": ("up", "matrix"),
        "feed_forward.w2.weight": ("down", "matrix"),
    }
    ATTENTION = {
        "self_attn.q_proj.weight": ("q", "matrix"),
        "self_attn.k_proj.weight": ("k", "matrix"),
        "self_attn.v_proj.weight": ("v", "matrix"),
        "self_attn.out_proj.weight": ("o", "matrix"),
        "self_attn.q_layernorm.weight": ("q_norm", "vector"),
        "self_attn.k_layernorm.weight": ("k_norm", "vector"),
    }
    CONV = {
        "conv.in_proj.weight": ("conv_in", "matrix"),
        "conv.conv.weight": ("conv_taps", "vector"),
        "conv.out_proj.weight": ("conv_out", "matrix"),
    }

    def config(self, hf):
        if hf.get("conv_bias"):
            raise SystemExit("a bias on the short convolution layers is not supported")
        params = hf.get("rope_parameters") or {"rope_theta": hf.get("rope_theta", 1000000.0)}
        if params.get("rope_type", "default") != "default":
            raise SystemExit("rope_scaling is not supported")
        heads = hf["num_attention_heads"]
        inter = hf.get("block_ff_dim", hf["intermediate_size"])
        if hf.get("block_auto_adjust_ff_dim", True):
            inter = int(2 * inter / 3)
            multiplier = hf.get("block_ffn_dim_multiplier", 1.0)
            if multiplier is not None:
                multiple = hf.get("block_multiple_of", 256)
                inter = multiple * ((int(multiplier * inter) + multiple - 1) // multiple)
        kinds = hf.get("layer_types") or [
            "full_attention" if i in hf["full_attn_idxs"] else "conv" for i in range(hf["num_hidden_layers"])
        ]
        if set(kinds) - {"full_attention", "conv"}:
            raise SystemExit(f"unsupported layer types {sorted(set(kinds) - {'full_attention', 'conv'})}")
        return {
            "hidden": hf["hidden_size"],
            "layers": hf["num_hidden_layers"],
            "heads": heads,
            "kvHeads": hf["num_key_value_heads"],
            "headDim": hf.get("head_dim") or hf["hidden_size"] // heads,
            "intermediate": inter,
            "vocab": hf["vocab_size"],
            "ropeTheta": params["rope_theta"],
            "rmsEps": hf["norm_eps"],
            "maxPositions": hf["max_position_embeddings"],
            "tieEmbeddings": bool(hf.get("tie_word_embeddings", True)),
            "activation": "silu",
            "layerTypes": ["conv" if k == "conv" else "attention" for k in kinds],
            "convKernel": hf["conv_L_cache"],
        }

    def layer_table(self, config, layer):
        return dict(self.LAYER, **(self.CONV if config["layerTypes"][layer] == "conv" else self.ATTENTION))

    def role(self, key, config):
        if key == "model.embed_tokens.weight":
            return "embed", "matrix"
        if key == "model.embedding_norm.weight":
            return "final_norm", "vector"
        if key == "lm_head.weight":
            return (None if config["tieEmbeddings"] else ("head", "matrix"))
        m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", key)
        if m:
            table = self.layer_table(config, int(m.group(1)))
            if m.group(2) in table:
                canonical, kind = table[m.group(2)]
                return f"layers.{m.group(1)}.{canonical}", kind
        return None

    def required(self, config):
        names = {"embed", "final_norm"} | ({"head"} if not config["tieEmbeddings"] else set())
        for i in range(config["layers"]):
            for canonical, _ in self.layer_table(config, i).values():
                names.add(f"layers.{i}.{canonical}")
        return names

    def prepare(self, canonical, array, config):
        # The depthwise filters are [channels, 1, taps], stored as one run of taps per channel.
        return array.reshape(-1) if canonical.endswith(".conv_taps") else array


class Llama4Text(Llama):
    """Llama 4 text with dense layers (MobileLLM-R1.5): the Llama decoder with a parameter free norm on each head's q and
    k. The q and k rows are reordered so that the rotation of adjacent pairs becomes the half split one the runtime
    applies."""

    name = "llama4"
    LAYER = {
        "input_layernorm.weight": ("attn_norm", "vector"),
        "post_attention_layernorm.weight": ("mlp_norm", "vector"),
        "self_attn.q_proj.weight": ("q", "matrix"),
        "self_attn.k_proj.weight": ("k", "matrix"),
        "self_attn.v_proj.weight": ("v", "matrix"),
        "self_attn.o_proj.weight": ("o", "matrix"),
        "feed_forward.gate_proj.weight": ("gate", "matrix"),
        "feed_forward.up_proj.weight": ("up", "matrix"),
        "feed_forward.down_proj.weight": ("down", "matrix"),
    }

    def config(self, hf):
        layers = hf["num_hidden_layers"]
        if hf.get("moe_layers") or hf.get("num_local_experts") or hf.get("interleave_moe_layer_step"):
            raise SystemExit("mixture of experts layers are not supported")
        if not all(hf.get("no_rope_layers", [1] * layers)[:layers]):
            raise SystemExit("layers without rotary embeddings are not supported")
        if not hf.get("use_qk_norm", True):
            raise SystemExit("a model without the q and k norm is not supported")
        if hf.get("attn_temperature_tuning"):
            raise SystemExit("attention temperature tuning is not supported")
        if hf.get("rope_scaling") or hf.get("attention_bias"):
            raise SystemExit("rope_scaling and attention biases are not supported")
        chunked = "chunked_attention" in (hf.get("layer_types") or [])[:layers]
        if chunked and hf["attention_chunk_size"] < hf["max_position_embeddings"]:
            raise SystemExit("chunked attention shorter than the context is not supported")
        return super().config(dict(hf, intermediate_size=hf["intermediate_size_mlp"]))

    def prepare(self, canonical, array, config):
        for role, heads in (("q", config["heads"]), ("k", config["kvHeads"])):
            if canonical.endswith(f".{role}"):
                # Row 2i of a head pairs with row 2i + 1. The half split form pairs row i with row i + head / 2.
                rows = array.reshape(heads, config["headDim"] // 2, 2, -1).transpose(0, 2, 1, 3)
                return rows.reshape(array.shape)
        return array


def for_config(hf):
    kind = hf.get("model_type")
    if kind == "llama":
        return Llama()
    if kind == "qwen2":
        return Qwen2()
    if kind == "qwen3":
        return Qwen3()
    if kind == "gemma3_text":
        return Gemma3()
    if kind == "lfm2":
        return Lfm2()
    if kind == "llama4_text":
        return Llama4Text()
    raise SystemExit(f"unsupported model_type {kind}")
