# Changelog

## 0.1.0

First release, a snapshot of the pure Luau engine.

- Container format, converter, tokenizer (byte BPE, ChatML), ops, the Llama decoder, sampling, `Engine` and `Chat`, the
  Lute backend and a streaming chat example. SmolLM2-135M-Instruct matches the reference.
- Tensor layer: formats, buffer parts up to 1 GiB, tensors, activation quantization.
- Kernel generator in Luau, generated kernels for f32, BF16, Q8 and Q4, spill guard, reference kernels, registry and
  capability probe.
- Gemma 3: post norms, local and global layers, SentencePiece style BPE with byte fallback, the turn template.
- Qwen2.5 and Qwen3 (thinking off), the Qwen pre-tokenizer, q/k/v biases and per head q/k norm, tiny fixtures for both.
- Packed Q8 kernel (offset binary weights in tiles of 8 rows, f64 activations with a scale per group of 32, exact
  integer sums), `ops/prepare`, the `q8` converter recipe and Q8 models in the runtime.
- Q4_0 weights (equal to llama.cpp, with the Q8_0 embedding and head), BF16 weights, both with converter recipes and
  runtime support.
- Prefill tile for Q8 that handles two tokens per pass, and prefix reuse that keeps the common prefix of the cache.
- Tensor parallel inference: the division of a model among workers, `container.loadSlice`, the worker and the main worker
  with their protocol, the `Runner` the chat runs on, in process transports and worker VMs in Lute, `--workers` in the chat
  example.
- Roblox backend: bundler and build tool, chunk ModuleScript and HTTP range weight sources, Actor workers, tested against
  a Luau model of Roblox. Run in Roblox Studio 0.748 (Run mode, server only), not in a published experience.
- Q8 KV cache (a scale per head and position, keys centred per head) that works with workers, snapshots and prefix
  reuse, `kv = "q8"` on the engine. Quality and speed in docs/benchmarks.md.
- Sampler with a bounded selection for top k, a histogram for top p over a flat vocabulary, penalties, min p and stop
  strings, with `ignoreStop` for benchmarks.
- Scheduler for many conversations by opaque session id: lanes, a fair queue by lane seconds, LRU eviction with an exact
  rebuild from the token state, streaming, cancel, timeouts and limits. The layout rule picks replicas and worker groups
  from the machine, the model and the number of concurrent chats, and the Lute pool runs it. A thin Roblox chat service binds players to
  it, tested against the model of Roblox.
- Benchmarks of speed against context length, replica scaling to 10 replicas, memory bandwidth, scheduler overhead,
  batching decode steps (not built, no gain) and more chats than lanes.
- Tests for every kernel against its reference, CI script with three flag modes, benchmark harness.
- Q8 KV cache in a vector layout (`kvLayout` "v3", three int8 values to an f32 lane) and a generated unrolled attention
  kernel that reads it, chosen when the vector read is there. Its attention runs near the f32 kernel's rate where the
  scalar kernel was 3 to 5 times slower, at 1.5 times the bytes of the int8 rows. Numbers in docs/benchmarks.md.
- Roblox: the unrolled attention is one module per head size, so every module is under the 199,997 character limit and the
  build fails on one that is not. Workers wait for their `ready` attribute, `load` runs on a serial topic, replies are
  parallel with `Platform.restore`, `fromChunks` keeps 4 chunks, and the slice defaults to 0.012.
