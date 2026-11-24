#!/usr/bin/env python3
"""Converts a Hugging Face checkpoint directory to an Olelo container.

    convert.py <checkpoint dir> --out models/smollm2-135m.olelo [--recipe f32|bf16|q8|q4]

`--recipe bf16` stores every matrix as BF16 and `--recipe q8` as Q8_0, the embedding and head included. `--recipe q4`
stores the layer matrices as Q4_0 and the embedding and head as Q8_0. `--q8-layout v12` writes the Q8 matrices in the vector layout, which runs only where the vector read is fast. The norm weights and biases stay f32. The
runtime reads only the container. Needs torch, safetensors, numpy and jinja2 (see tools/convert/README.md).
"""

import argparse
import json
import os
import struct
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch as archs  # noqa: E402
import container  # noqa: E402

CONVERTER = "olelo-convert 1"
# Kernel steps every f32 kernel divides: R8 for decode and R4 for prefill, 24 floats for the vector loop.
F32_ROW_TILE = 8
F32_KSTEP = 24

WHITE_SPACE = [
    (0x09, 0x0D), (0x20, 0x20), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680), (0x2000, 0x200A),
    (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000),
]  # fmt: skip


def category_ranges(prefix):
    """Code point ranges whose general category starts with `prefix`."""
    ranges = []
    start = None
    for cp in range(0x110000):
        hit = unicodedata.category(chr(cp)).startswith(prefix)
        if hit and start is None:
            start = cp
        elif not hit and start is not None:
            ranges.append((start, cp - 1))
            start = None
    if start is not None:
        ranges.append((start, 0x10FFFF))
    return ranges


def pack_ranges(ranges):
    out = struct.pack("<I", len(ranges))
    for lo, hi in ranges:
        out += struct.pack("<II", lo, hi)
    return out


def unicode_blob():
    return pack_ranges(category_ranges("L")) + pack_ranges(category_ranges("N")) + pack_ranges(WHITE_SPACE)


def tokenizer_blobs(tok):
    """The vocabulary and merges of a byte level BPE tokenizer.json as two binary blobs."""
    vocab = tok["model"]["vocab"]
    size = max(vocab.values()) + 1
    by_id = [None] * size
    for text, i in vocab.items():
        by_id[i] = text
    for t in tok["added_tokens"]:
        if t["id"] >= size:
            by_id.extend([None] * (t["id"] + 1 - len(by_id)))
            size = len(by_id)
        by_id[t["id"]] = t["content"]
    vocab_blob = struct.pack("<I", size)
    for text in by_id:
        raw = (text or "").encode("utf-8")
        vocab_blob += struct.pack("<H", len(raw)) + raw
    merges = tok["model"]["merges"]
    merges_blob = struct.pack("<I", len(merges))
    for merge in merges:
        a, b = merge.split(" ", 1) if isinstance(merge, str) else merge
        merges_blob += struct.pack("<III", vocab[a], vocab[b], vocab[a + b])
    return vocab_blob, merges_blob, size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", default="f32", choices=["f32", "bf16", "q8", "q4"])
    ap.add_argument("--q8-layout", default="p8", choices=["p8", "v12"], help="v12 needs the vector read at run time")
    ap.add_argument("--tokenizer-only", action="store_true", help="write the header and tokenizer blobs, no weights")
    args = ap.parse_args()

    with open(os.path.join(args.checkpoint, "config.json")) as f:
        hf_config = json.load(f)
    with open(os.path.join(args.checkpoint, "tokenizer.json")) as f:
        tok = json.load(f)
    with open(os.path.join(args.checkpoint, "tokenizer_config.json")) as f:
        tok_config = json.load(f)

    generation = os.path.join(args.checkpoint, "generation_config.json")
    if os.path.exists(generation):
        with open(generation) as f:
            extra = json.load(f).get("eos_token_id", [])
        ids = hf_config.get("eos_token_id", [])
        ids = ids if isinstance(ids, list) else [ids]
        hf_config["eos_token_id"] = ids + [i for i in (extra if isinstance(extra, list) else [extra]) if i not in ids]

    arch = archs.for_config(hf_config)
    config = arch.config(hf_config)
    vocab_blob, merges_blob, _ = tokenizer_blobs(tok)

    template = tok_config.get("chat_template")
    jinja = os.path.join(args.checkpoint, "chat_template.jinja")
    if template is None and os.path.exists(jinja):
        with open(jinja, encoding="utf-8") as f:
            template = f.read()
    if isinstance(template, list):
        template = next(t["template"] for t in template if t["name"] == "default")
    chat = arch.chat(template, tok_config)
    specials = [{"id": t["id"], "content": t["content"]} for t in tok["added_tokens"]]
    header = {
        "converter": CONVERTER,
        "arch": arch.name,
        "config": config,
        "tokenizer": arch.tokenizer(tok, tok_config, hf_config, specials),
        "chat": chat,
    }
    writer = container.Writer(header)
    writer.add_blob("tokenizer.vocab", vocab_blob)
    writer.add_blob("tokenizer.merges", merges_blob)
    writer.add_blob("unicode", unicode_blob())

    if args.tokenizer_only:
        size = writer.write(args.out)
        print(f"wrote {args.out} ({size} bytes, tokenizer only)")
        return

    from safetensors import safe_open

    files = sorted(f for f in os.listdir(args.checkpoint) if f.endswith(".safetensors"))
    seen = set()
    for name in files:
        with safe_open(os.path.join(args.checkpoint, name), framework="pt") as st:
            for key in st.keys():
                role = arch.role(key, config)
                if role is None:
                    continue
                canonical, kind = role
                array = arch.prepare(canonical, st.get_tensor(key).float().numpy(), config)
                if canonical.endswith(".gate") and array.shape[0] != config["intermediate"]:
                    raise SystemExit(f"{canonical} has {array.shape[0]} rows, the config says {config['intermediate']}")
                layer = canonical.startswith("layers.")
                if kind == "matrix" and (args.recipe == "q8" or (args.recipe == "q4" and not layer)):
                    writer.add_matrix_q8(canonical, array, layout=args.q8_layout)
                elif kind == "matrix" and args.recipe == "q4":
                    writer.add_matrix_q4(canonical, array)
                elif kind == "matrix" and args.recipe == "bf16":
                    writer.add_matrix(canonical, array, "bf16", F32_ROW_TILE, F32_KSTEP)
                elif kind == "matrix":
                    writer.add_matrix(canonical, array, "f32", F32_ROW_TILE, F32_KSTEP)
                else:
                    writer.add_vector(canonical, array)
                seen.add(canonical)
    missing = arch.required(config) - seen
    if missing:
        raise SystemExit(f"the checkpoint lacks tensors: {sorted(missing)[:5]}")
    size = writer.write(args.out)
    print(f"wrote {args.out} ({size} bytes, {len(writer.tensors)} tensors)")


if __name__ == "__main__":
    main()
