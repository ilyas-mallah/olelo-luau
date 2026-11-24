# Converter

Python tools that run on a developer machine. The runtime never needs them.

Needs Python with torch, transformers, safetensors, tokenizers and numpy. Set `OLELO_PYTHON` to such an interpreter, or
use any environment that has them.

- `convert.py <checkpoint dir> --out models/<name>.olelo [--recipe f32|bf16|q8|q4] [--q8-layout p8|v12]` writes a
  container, see [docs/container.md](../../docs/container.md). The checkpoint's `model_type` must be `llama` (SmolLM2
  included), `qwen2`, `qwen3`, `gemma3_text`, `lfm2` (LFM2 and LFM2.5) or `llama4_text` (dense layers only, MobileLLM-R1.5).
  `--tokenizer-only` writes only the header and tokenizer blobs.

  | Recipe | Matrices | Notes |
  |---|---|---|
  | `f32` (default) | float32 | the reference |
  | `bf16` | BF16 | |
  | `q8` | Q8_0, embedding and head included | equal to `llama-quantize`'s. `--q8-layout p8` (default) is the packed layout and runs on every host. `--q8-layout v12` is the vector layout, 1.375 times the data, and runs only where `buffer.readvectorf32x3` is fast, otherwise the load fails |
  | `q4` | Q4_0 layers, Q8_0 embedding and head | packed `p4` layout, equal to llama.cpp's Q4_0 |

  Norm weights and biases stay f32 in every recipe. `--q8-layout` applies to every Q8 matrix, so it also changes the
  embedding and the head of the `q4` recipe.
- `repack_q8v.py <in.olelo> <out.olelo>` copies a container with its `p8` Q8 matrices in the `v12` layout, so a converted
  model needs no second conversion. The copy is a new file. A container with no `p8` Q8 matrix is copied as it is.
- `repack_q4.py <container.olelo> [...]` rewrites the row major Q4 matrices of an older container to the packed `p4`
  layout in place. The converter writes `p4`, so only a container from before that needs it.
- `reference.py <checkpoint dir> <prompts.json> <out dir>` runs the Hugging Face model in float32 and writes the dumps
  the parity tests compare with. `prompts.json` holds the prompts used for SmolLM2.
- `make_fixtures.py <SmolLM2 checkpoint dir>` writes the committed fixtures under `tests/fixtures`: a small random model
  with padded shapes and a trained tokenizer, its dumps, and the SmolLM2 tokenizer with its corpus ids.

A real model check, from the repository root:

```sh
python tools/convert/convert.py <checkpoint dir> --out models/smollm2-135m.olelo
python tools/convert/reference.py <checkpoint dir> tools/convert/prompts.json out/ref/smollm2-135m
OLELO_MODEL=models/smollm2-135m.olelo OLELO_REF=out/ref/smollm2-135m lute run tests/run.luau parity/real
```

Chat template support is the ChatML form. The converter extracts it from the Jinja template and the tokenizer test
compares the rendered prompts with `apply_chat_template`.
