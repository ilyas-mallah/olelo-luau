#!/usr/bin/env python3
"""Writes the test fixtures under tests/fixtures.

    make_fixtures.py <SmolLM2 checkpoint dir> [--qwen3 <Qwen3 dir>] [--gemma3 <Gemma 3 dir>] [--lfm2 <LFM2.5 dir>]
                     [--mobilellm <MobileLLM-R1.5 dir>] [--out tests/fixtures]

tiny-llama/   a random 2 layer model with padded shapes and a small trained tokenizer, as an f32, a BF16, a Q8 and a Q4
              container with its reference dumps, for parity tests that run in seconds
tiny-gemma3/  the same for a random Gemma 3 model with a SentencePiece style tokenizer, local and global layers
tokenizer/    the SmolLM2 and Qwen3 tokenizers as containers without weights, and the Hugging Face token ids of a text
              corpus, also for Gemma 3 (the test builds its container from the checkpoint)
tiny-lfm2/    the same for a random LFM2 model, three short convolution layers and two attention layers
tiny-llama4/  and for a random Llama 4 text model, with the Llama 3 split and a tokenizer that skips merges for whole tokens
Needs torch, transformers and tokenizers (see tools/convert/README.md).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import torch
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    Lfm2Config,
    Lfm2ForCausalLM,
    Llama4ForCausalLM,
    Llama4TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arch as archs  # noqa: E402
CHAT_TEMPLATE = (
    "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a tiny test model<|im_end|>\n' }}{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

THINKING_TEMPLATE = CHAT_TEMPLATE.replace(
    "{{ '<|im_start|>assistant\n' }}",
    "{{ '<|im_start|>assistant\n' }}{% if enable_thinking is defined and enable_thinking is false %}"
    "{{ '<think>\n\n</think>\n\n' }}{% endif %}",
)

# LFM2's template: the beginning of sequence token, no default system message, ChatML turns.
LFM2_TEMPLATE = (
    "{{- bos_token -}}{%- if messages[0]['role'] == 'system' -%}"
    "{{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' -}}{%- set messages = messages[1:] -%}{%- endif -%}"
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' -}}{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
)

# Llama 3's template as MobileLLM uses it: a system turn with a default message, headers, trimmed contents.
LLAMA3_TEMPLATE = (
    "{{- bos_token }}{%- if messages[0]['role'] == 'system' %}"
    "{%- set system_message = messages[0]['content']|trim %}{%- set messages = messages[1:] %}"
    '{%- else %}{%- set system_message = "Please reason step by step, and put your final answer within \\\\boxed{}." %}'
    "{%- endif %}"
    "{{- '<|start_header_id|>system<|end_header_id|>\n\n' + system_message + '<|eot_id|>' }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content']|trim + '<|eot_id|>' }}"
    "{%- endfor %}{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{%- endif %}"
)

TRAIN_TEXT = [
    "the quick brown fox jumps over the lazy dog and the dog does not mind",
    "hello world, this is a small model for tests. it has two layers and a tiny vocabulary.",
    "what is a hash table? a hash table maps keys to values. i think that is right.",
    "numbers like 12345 and 6789 are split into digits, and spaces matter in text.",
    "user: hi there assistant: hello, how can i help you today? thanks, that is all.",
] * 20

CORPUS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure we've seen it, they'll say it's fine, you'd know.",
    "DON'T SHOUT, WE'LL HEAR YOU",
    "1234567890",
    "Order 66 costs 4.99 USD, or 1,299.00 in total.",
    "x = 10; y = x * 2 + 3",
    "  leading and trailing spaces  ",
    "tabs\tand\nnewlines\r\nand\n\n\nblank lines",
    "   ",
    "\n",
    " ",
    "a",
    "",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "for (int i = 0; i < n; ++i) { sum += a[i]; }",
    "https://example.com/path?query=1&other=two#fragment",
    "snake_case camelCase PascalCase kebab-case",
    "café naïve résumé façade Zoë",
    "你好，世界。今天天气很好。",
    "こんにちは世界、元気ですか",
    "안녕하세요 세계",
    "Привет, мир! Как дела?",
    "مرحبا بالعالم",
    "שלום עולם",
    "😀 🚀 🎉 emoji and 👨‍👩‍👧‍👦 family",
    "∑ x² ≤ π ≈ 3.14159 → ∞",
    "non breaking em space and　ideographic space",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "word " * 40,
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    "text <|endoftext|> more text",
    "punctuation!!! ??? ... --- ___ ///",
    "'s 't 're 've 'm 'll 'd",
    "it's, can't; won't: 'quoted' \"double\"",
    "A1B2C3 version 2.0.1-beta",
    "٣٤٥ १२३ ๑๒๓",
    "line one\nline two\n  indented\n\ttabbed",
    "Mixed 中文 and English and 123 numbers",
    "end with space ",
    " ",
    "ǅ ǆ ǈ titlecase letters",
    "Hello!\n\nWorld",
    "a \n b",
    "x  \n  y\n",
    "wait...\r\n\r\nok?!\n",
    " ,\n",
    "'Hello 'tis 'TWAS 'Ve 'RE",
    "\t\tindented\t",
    "tab\tseparated\tvalues  nbsp",
    "year2024 v1.2.3 3.14159",
]

CHATS = [
    [{"role": "user", "content": "Hi!"}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2 + 2?"}],
    [
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Nice to meet you, Ada."},
        {"role": "user", "content": "What is my name?"},
    ],
]


def run(*args):
    subprocess.run([sys.executable, *args], check=True)


def gemma_tokenizer(work):
    """A small tokenizer in Gemma's form: spaces become U+2581, characters outside the vocabulary fall back to byte
    tokens, the newline runs are added tokens, and the text is one piece."""
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.Replace(" ", "\u2581")
    specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
    alphabet = [chr(c) for c in range(33, 127)] + ["\u2581"]
    trainer = trainers.BpeTrainer(vocab_size=420, special_tokens=specials, initial_alphabet=alphabet)
    tok.train_from_iterator([t.replace(" ", "\u2581") for t in TRAIN_TEXT], trainer)
    data = json.loads(tok.to_str())
    vocab = data["model"]["vocab"]
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    model = models.BPE(vocab, [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in data["model"]["merges"]], byte_fallback=True, unk_token="<unk>")
    tok = Tokenizer(model)
    tok.normalizer = normalizers.Replace(" ", "\u2581")
    tok.decoder = decoders.Sequence([decoders.Replace("\u2581", " "), decoders.ByteFallback(), decoders.Fuse()])
    tok.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", vocab["<bos>"])])
    tok.add_special_tokens(specials)
    tok.add_tokens([AddedToken("\n" * k, normalized=False) for k in (1, 2, 3)])
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        clean_up_tokenization_spaces=False,
    )
    with open(os.path.join(HERE, "gemma_chat_template.jinja"), encoding="utf-8") as f:
        fast.chat_template = f.read()
    fast.save_pretrained(work)
    return fast, len(tok.get_vocab())


def llama3_style_tokenizer(work, kind):
    """A small byte level tokenizer with the Llama 3 split (numbers in pieces of up to three digits), a beginning of
    sequence token that the post processor adds, and for llama4 merges that a whole token skips (ignore_merges)."""
    llama4 = kind == "llama4"
    specials = (
        ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
        if llama4
        else ["<|pad|>", "<|startoftext|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    )
    bos, eos = (specials[0], specials[4]) if llama4 else (specials[1], specials[4])
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(archs.LLAMA3_PATTERN), "isolated", False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(TRAIN_TEXT, trainer)
    if llama4:
        data = json.loads(tok.to_str())
        data["model"]["ignore_merges"] = True
        tok = Tokenizer.from_str(json.dumps(data))
    tok.post_processor = processors.TemplateProcessing(
        single=f"{bos} $A", special_tokens=[(bos, tok.token_to_id(bos))]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=bos,
        eos_token=eos,
        pad_token=eos,
        clean_up_tokenization_spaces=False,
    )
    fast.chat_template = LLAMA3_TEMPLATE if llama4 else LFM2_TEMPLATE
    fast.save_pretrained(work)
    return fast, len(tok.get_vocab())


def tiny(out, kind="llama", wide=False):
    """A random model whose shapes need padding: hidden 60, heads 6 of 10, intermediate 100. With `wide` the shapes are
    whole groups of 32 instead (hidden 128, 4 heads of 32, intermediate 192), which a quantized model needs to be divided
    among workers, and only the containers are written. `kind` is the family:
    llama, qwen2 (biases on q, k and v), qwen3 (a norm on each head's q and k, thinking off in the template) or
    gemma3 (four norms, local and global layers, a SentencePiece style tokenizer)."""
    torch.manual_seed(7)
    name = f"{'wide' if wide else 'tiny'}-{kind}"
    work = os.path.join(out, name, "hf")
    os.makedirs(work, exist_ok=True)
    if kind == "gemma3":
        fast, vocab = gemma_tokenizer(work)
    elif kind in ("lfm2", "llama4"):
        fast, vocab = llama3_style_tokenizer(work, kind)
    else:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.ByteLevel(add_prefix_space=False)]
        )
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=400,
            special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator(TRAIN_TEXT, trainer)
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            bos_token="<|im_start|>",
            eos_token="<|im_end|>",
            unk_token="<|endoftext|>",
            pad_token="<|im_end|>",
            clean_up_tokenization_spaces=False,
        )
        fast.chat_template = THINKING_TEMPLATE if kind == "qwen3" else CHAT_TEMPLATE
        fast.save_pretrained(work)
        vocab = len(tok.get_vocab())
    config_class, model_class = {
        "lfm2": (Lfm2Config, Lfm2ForCausalLM),
        "llama4": (Llama4TextConfig, Llama4ForCausalLM),
        "llama": (LlamaConfig, LlamaForCausalLM),
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        "gemma3": (Gemma3TextConfig, Gemma3ForCausalLM),
    }[kind]
    args = dict(
        vocab_size=vocab,
        hidden_size=128 if wide else 60,
        intermediate_size=192 if wide else 100,
        num_hidden_layers=2,
        num_attention_heads=4 if wide else 6,
        num_key_value_heads=2,
        head_dim=32 if wide else 10,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
        initializer_range=0.3,
        hidden_act="silu",
    )
    if kind == "gemma3":
        # Two local layers and a global one, a window shorter than the prompts, a score scale that is not head_dim.
        args.update(
            num_hidden_layers=3,
            sliding_window=6,
            sliding_window_pattern=3,
            query_pre_attn_scalar=12,
            rope_local_base_freq=10000.0,
            rope_theta=100000.0,
            hidden_activation="gelu_pytorch_tanh",
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
        )
    if kind == "lfm2":
        # Three short convolution layers and two attention layers, a feed forward width the auto adjustment rounds
        # (150 to 104), no head_dim of its own.
        for key in ("head_dim", "rope_theta", "rms_norm_eps", "hidden_act", "intermediate_size"):
            args.pop(key)
        args.update(
            num_hidden_layers=5,
            intermediate_size=150,
            block_multiple_of=8,
            norm_eps=1e-5,
            conv_L_cache=3,
            layer_types=["conv", "conv", "full_attention", "conv", "full_attention"],
            rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
            bos_token_id=1,
            eos_token_id=4,
            pad_token_id=0,
        )
    if kind == "llama4":
        args.update(
            num_hidden_layers=3,
            intermediate_size_mlp=100,
            num_local_experts=0,
            num_experts_per_tok=0,
            interleave_moe_layer_step=0,
            moe_layers=[],
            no_rope_layers=[1, 1, 1],
            use_qk_norm=True,
            attn_temperature_tuning=False,
            attention_chunk_size=256,
            layer_types=["full_attention"] * 3,
            bos_token_id=0,
            eos_token_id=4,
            pad_token_id=4,
        )
    config = config_class(**args)
    model = model_class(config)
    if kind == "gemma3":
        model.generation_config.eos_token_id = [1, 5]
    for p in model.parameters():
        if p.ndim == 1:
            p.data.add_(torch.randn_like(p) * 0.1)
    model.save_pretrained(work, safe_serialization=True)
    dest = os.path.join(out, name)
    run(os.path.join(HERE, "convert.py"), work, "--out", os.path.join(dest, "model.olelo"))
    run(os.path.join(HERE, "convert.py"), work, "--out", os.path.join(dest, "model-q8.olelo"), "--recipe", "q8")
    if kind == "llama" or wide:
        run(os.path.join(HERE, "convert.py"), work, "--out", os.path.join(dest, "model-q8v.olelo"), "--recipe", "q8", "--q8-layout", "v12")
        for recipe in ("bf16", "q4"):
            run(os.path.join(HERE, "convert.py"), work, "--out", os.path.join(dest, f"model-{recipe}.olelo"), "--recipe", recipe)
        if not wide:
            for recipe in ("bf16", "q8", "q4"):
                rounded(model, work, dest, recipe)
    if wide:
        shutil.rmtree(work)
        return
    run(os.path.join(HERE, "reference.py"), work, os.path.join(HERE, "tiny_prompts.json"), os.path.join(dest, "ref"), "--new-tokens", "24")
    if kind in ("llama", "gemma3", "lfm2", "llama4"):
        corpus(work, os.path.join(dest, "tokens.json"))
    shutil.rmtree(work)


def rounded(model, work, dest, recipe):
    """References from the model with its weights rounded as `recipe` stores them (the embedding as Q8 for q4), so the
    parity tests of the formats compare against the same weights and not against the unrounded ones."""
    sys.path.insert(0, HERE)
    import container

    copy = type(model)(model.config)
    copy.load_state_dict(model.state_dict())
    with torch.no_grad():
        for name, p in copy.named_parameters():
            if p.ndim == 2:
                kind = recipe if (recipe != "q4" or ".layers." in name) else "q8"
                p.copy_(torch.from_numpy(container.dequantize(p.float().numpy(), kind)))
    folder = f"{work}-{recipe}"
    shutil.copytree(work, folder, ignore=shutil.ignore_patterns("*.safetensors"))
    copy.save_pretrained(folder, safe_serialization=True)
    run(os.path.join(HERE, "reference.py"), folder, os.path.join(HERE, "tiny_prompts.json"), os.path.join(dest, f"ref-{recipe}"), "--new-tokens", "24")
    shutil.rmtree(folder)


def corpus(checkpoint, path):
    tok = AutoTokenizer.from_pretrained(checkpoint)
    texts = [{"text": t, "ids": tok(t)["input_ids"]} for t in CORPUS]
    chats = []
    for messages in CHATS:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        chats.append({"messages": messages, "text": text, "ids": ids})
    decoded = [{"ids": e["ids"], "text": tok.decode(e["ids"])} for e in texts[:20]]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "chats": chats, "decoded": decoded}, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("smollm2")
    ap.add_argument("--qwen3", help="a Qwen3 checkpoint dir, for the Qwen tokenizer fixture")
    ap.add_argument("--gemma3", help="a Gemma 3 checkpoint dir, for the token ids of the corpus (its container is too big to keep)")
    ap.add_argument("--out", default="tests/fixtures")
    ap.add_argument("--lfm2", help="an LFM2.5 checkpoint dir, for the token ids of the corpus (the test builds its container)")
    ap.add_argument("--mobilellm", help="a MobileLLM-R1.5 checkpoint dir, for the token ids of the corpus")
    ap.add_argument("--kinds", default="llama,qwen2,qwen3,gemma3", help="the tiny models to write")
    ap.add_argument("--wide", action="store_true", help="write only the wide models, for the tests of tensor parallel")
    ap.add_argument("--no-tokenizers", action="store_true", help="skip the real tokenizer fixtures")
    args = ap.parse_args()
    for kind in filter(None, args.kinds.split(",")):
        tiny(args.out, kind, args.wide)
    if args.no_tokenizers or args.wide:
        return
    dest = os.path.join(args.out, "tokenizer")
    os.makedirs(dest, exist_ok=True)
    run(os.path.join(HERE, "convert.py"), args.smollm2, "--out", os.path.join(dest, "smollm2.olelo"), "--tokenizer-only")
    corpus(args.smollm2, os.path.join(dest, "smollm2.json"))
    if args.gemma3:
        corpus(args.gemma3, os.path.join(dest, "gemma3.json"))
    if args.lfm2:
        corpus(args.lfm2, os.path.join(dest, "lfm2.json"))
    if args.mobilellm:
        corpus(args.mobilellm, os.path.join(dest, "mobilellm.json"))
    if args.qwen3:
        run(os.path.join(HERE, "convert.py"), args.qwen3, "--out", os.path.join(dest, "qwen3.olelo"), "--tokenizer-only")
        corpus(args.qwen3, os.path.join(dest, "qwen3.json"))


if __name__ == "__main__":
    main()
