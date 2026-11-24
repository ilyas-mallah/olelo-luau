# Models

The models olelo-luau has run and checked against Hugging Face. The converter takes a checkpoint whose `model_type` is `llama`,
`qwen2`, `qwen3`, `gemma3_text`, `lfm2` or `llama4_text` and stops on any other ([converter](../tools/convert/README.md)).

| Model | Family | Formats | Checked against Hugging Face | Notes |
|---|---|---|---|---|
| `HuggingFaceTB/SmolLM2-135M-Instruct` | llama | f32, Q8, BF16, Q4 | Tokens, hidden states, logits and greedy tokens match in f32 ([Parity](benchmarks.md#parity)). Q8, BF16 and Q4 errors in [benchmarks.md](benchmarks.md#q8-quality) | Q4 equals llama.cpp's Q4_0 |
| Qwen2.5-0.5B-Instruct | qwen2 | f32, Q8, Q4 | Tiny random model in f32 and Q8 | Biases on q, k and v |
| Qwen3-0.6B | qwen3 | f32, Q8 | Tiny random model in f32 and Q8, token ids of a 51 text corpus | Chats with thinking off |
| Gemma 3 1B | gemma3_text | f32, Q8 | Tiny random model in f32 and Q8, token ids of the corpus | Refuses `rope_scaling` and logit soft capping. No measured rate yet |
| `LiquidAI/LFM2.5-230M` | lfm2 | f32, Q8 | f32 hidden 7.0e-5, logits 1.0e-5, greedy 48 of 48 on 6 prompts. Q8 logits 0.044 | Convolution layers, so no `--workers` |
| `LiquidAI/LFM2.5-350M` | lfm2 | f32, Q8 | f32 hidden 1.5e-4, logits 1.4e-5, greedy 48 of 48 on 6 prompts. Q8 logits 0.039 | Same as the 230M |
| `facebook/MobileLLM-R1.5-360M` | llama4_text | f32, Q8 | f32 hidden 8.4e-5, logits 2.1e-5, greedy 48 of 48 on 6 prompts. Q8 logits 0.049 | Dense layers only, the converter refuses NoPE layers, chunked attention shorter than the context, temperature tuning and experts. Answers open with `<think>` |

Errors are relative to the RMS of the reference, the worst of the 6 prompts. Q8 errors are RMS errors. Q8 greedy continuations
leave the reference only inside near ties (margin under 0.1). For Qwen2.5, Qwen3 and Gemma 3 the repository holds no error figures for the real checkpoint.
The table lists the tiny model and token id results instead.

## How a model is checked

[`reference.py`](../tools/convert/reference.py) runs the Hugging Face model in float32. For 6 prompts it dumps the token ids and the hidden
state after every layer for the first 16 positions. It also dumps the last prompt position's logits and a greedy continuation. The
[`real` parity test](../tests/parity/real.test.luau) compares the engine with the dumps on a converted model.
The f32 limits are 3e-3 for hidden states and 1e-3 for logits. A greedy token may differ only where the two best logits are nearly tied.

CI runs the same comparison on tiny random models of every family in [`tests/fixtures`](../tests/fixtures), in f32 and Q8.
The llama one also runs in BF16 and Q4. The tokenizer test compares token ids and chat prompts with Hugging Face on a text corpus.

Rates are in [benchmarks.md](benchmarks.md).
