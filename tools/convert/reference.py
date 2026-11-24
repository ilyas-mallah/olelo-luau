#!/usr/bin/env python3
"""Writes the reference dumps a parity test compares against.

    reference.py <checkpoint dir> <prompts.json> <out dir> [--new-tokens 64] [--positions 16]

Runs the Hugging Face model in float32 on the CPU, from the same weight values the container holds. For each prompt it
writes the token ids, the greedy continuation with the margin between the two best logits at every step, the hidden
state after every decoder layer for the first positions, and the logits of the last prompt position.
Needs torch and transformers (see tools/convert/README.md).
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--positions", type=int, default=16)
    args = ap.parse_args()

    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.float32).eval()
    with open(args.prompts) as f:
        prompts = json.load(f)
    os.makedirs(args.out, exist_ok=True)

    layer_out = []
    hooks = [
        layer.register_forward_hook(lambda m, i, o: layer_out.append((o[0] if isinstance(o, tuple) else o).detach()))
        for layer in model.model.layers
    ]

    meta = {"newTokens": args.new_tokens, "positions": args.positions, "prompts": []}
    for index, prompt in enumerate(prompts):
        if "messages" in prompt:
            text = tok.apply_chat_template(prompt["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok.apply_chat_template(prompt["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if isinstance(ids, dict) or hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
        else:
            text = prompt["text"]
            ids = tok(text)["input_ids"]
        special = "messages" not in prompt
        input_ids = torch.tensor([ids])
        layer_out.clear()
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        n = min(args.positions, len(ids))
        hidden = torch.stack([h[0, :n] for h in layer_out]).numpy().astype("<f4")
        hidden.tofile(os.path.join(args.out, f"p{index}.hidden.f32"))
        out.logits[0, -1].numpy().astype("<f4").tofile(os.path.join(args.out, f"p{index}.logits.f32"))

        generated, margins = [], []
        past = out.past_key_values
        logits = out.logits[0, -1]
        for _ in range(args.new_tokens):
            top = torch.topk(logits, 2)
            generated.append(int(top.indices[0]))
            margins.append(float(top.values[0] - top.values[1]))
            layer_out.clear()
            with torch.no_grad():
                step = model(torch.tensor([[generated[-1]]]), past_key_values=past, use_cache=True)
            past = step.past_key_values
            logits = step.logits[0, -1]
        meta["prompts"].append(
            {
                "name": prompt.get("name", f"p{index}"),
                "text": text,
                "addSpecial": special,
                "ids": list(ids),
                "generated": generated,
                "margins": margins,
            }
        )
    for h in hooks:
        h.remove()
    meta["layers"] = len(model.model.layers)
    meta["hidden"] = model.config.hidden_size
    meta["vocab"] = model.config.vocab_size
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"wrote {len(prompts)} prompts to {args.out}")


if __name__ == "__main__":
    main()
