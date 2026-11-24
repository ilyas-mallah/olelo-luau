# Tests

Each suite guards one property, which is the comment at the top of its file and the second column below, kept in step
with the files by `unit/index`. Run every command from the repository root.

- All suites: `lute run tests/run.luau [filter]`. It exits 1 when a case fails.
- All suites in every flag mode: `tools/ci.sh`. A host without `buffer.readvectorf32x3` skips the vector kernels and runs
  the rest, so the fallback path stays tested. `OLELO_FLAG_MODE` (`none`, `library`, `both`) tells the probe test
  which flags the run set. The cases that start worker VMs run only with `OLELO_VM=1`, which `ci.sh` sets for the flag
  lute, because a lute without the fix for the `vm.create` crash can end the process.
- The gen suites need `luau-compile` of the pinned Luau, named by `LUAU_COMPILE` or found on `PATH`, and skip without it.

| Suite | Property |
|---|---|
| `unit/formats` | Address helpers place every weight of every format at a distinct byte, scale offsets tile the scale region, and BF16 and f32 rounding match the formats. |
| `unit/store` | A matrix is cut into parts of whole tiles that fit the buffer limit, covering every row once. |
| `unit/tensor` | A tensor pads with zeros and reads back what it stores in every format across part boundaries. |
| `unit/quant` | Activation quantization is within half a step of the input, a zero row gets scale 1, and group quantization follows Q8_0. |
| `unit/prepare` | Prepared activations equal the reference conversions for every kernel form, ragged rows included. |
| `unit/json` | The JSON decoder reads scalars, escapes, surrogate pairs and nesting, and rejects malformed text. |
| `unit/container` | The container header, tensor table and blobs load padded as the kernels need, and bad files are refused with advice. |
| `unit/tokenizer` | Token ids equal the Hugging Face tokenizers on a corpus for SmolLM2 and the small trained vocabulary, chat prompts equal the chat template, and streaming never splits a character. |
| `unit/sampling` | The generator repeats the reference stream, and each sampler setting picks the tokens it should. |
| `unit/ops` | Norm, residual add, rope, the gated activations and the embedding lookup compute their definitions. |
| `unit/attention` | Both attention implementations match a plain computation, with grouped heads, a window and padded heads. |
| `unit/kvcache` | The Q8 KV cache quantizer stays within half a step, attention over it equals attention over the values it holds, a model on it stays within the rounding of the f32 cache, and a rewind does not change it. |
| `unit/shortconv` | The short convolution op matches its definition, and a model with such layers gives the logits of a straight run after passes of any size, a rewind and a snapshot, counts their history in the cache estimate and is refused workers. |
| `unit/split` | A division among workers covers every head, group of columns and row once, a KV head cut between workers is in both shares, and a slice of a matrix holds the weights of the whole in every format. |
| `unit/index` | This index, `tests/run.luau` and the suite files agree. |
| `kernels/f32_vector` | The f32 vector kernels match the reference within K x 2^-24 x sum of abs(w x). |
| `kernels/f32_buffer` | The 3 x readf32 fallback matches the reference within the same bound. |
| `kernels/f32_scalar` | The scalar f32 kernel matches the reference within the same bound. |
| `kernels/bf16` | The BF16 kernels match the reference within the same bound. |
| `kernels/q8` | The Q8 kernel equals the reference exactly, and the integer sum with unit scales. |
| `kernels/q4` | The Q4 kernel equals the reference exactly, and the integer sum with unit scales. |
| `kernels/registry` | The registry selects vector kernels only when the host can run them. |
| `kernels/probe` | The probe picks the vector kernel exactly when the vector read exists and is fast, as the flags of the run say. |
| `api/chat` | A reused cache gives the tokens of a run from scratch, and streaming, stop strings, cancel, limits and seeds behave. |
| `api/prefix` | Conversations that start with the same system prompt copy its cache instead of prefilling it and answer exactly as without sharing, on one VM and over workers, with an f32 or a Q8 cache. |
| `serve/scheduler` | Users on a pool of lanes take turns by the lane seconds they use, wait for slots, are dropped and rebuilt exactly, and their cancels, timeouts and limits end requests with their own finish. |
| `serve/layout` | The layout rule gives a group of workers sized by the model to few users, replicas to many, and queues on slots when memory or threads run out. |
| `serve/pool` | Replica VMs and groups of worker VMs behind the scheduler, opened from a layout, answer as a chat in one VM does. Needs a lute with the `vm.create` fix (`OLELO_VM=1`). |
| `parallel/parallel` | Tensor parallel inference over 1 to 4 workers, in this VM and in worker VMs, on an f32 or a Q8 KV cache, gives the logits, the greedy tokens, the rewinds and the chats of the single VM for every family and format. The worker VM cases need `OLELO_VM=1`. |
| `roblox/roblox` | The Roblox backend against a Luau model of Roblox that enforces the module size, Actor and phase rules: the bundle as ModuleScripts, chunk and HTTP weight sources, the example Script, and Actor workers with VMs of their own give the tokens and chats of the single VM. Says nothing about Roblox Studio's timing. |
| `roblox/chat_service` | The Roblox chat service over replica Actors: players get the answer of a chat in one VM, share the Actors by turns, are forgotten when they leave, and are answered over a RemoteEvent. Says nothing about Roblox Studio. |
| `parity/tiny` | A small random model with padded shapes matches the reference in token ids, hidden states, logits and greedy tokens. |
| `parity/tiny_formats` | The BF16, Q8 and Q4 containers of the same model stay within the error their rounding allows of the same reference, and prefill equals decode on each. |
| `parity/tiny_families` | Tiny Qwen2 (biases), Qwen3 (q/k norm) and Gemma 3 (post norms, local layers) models match their references in f32 and stay within Q8 error in Q8. |
| `parity/real` | A real converted model matches its reference dumps, when `OLELO_MODEL` and `OLELO_REF` are set. |
| `gen/generator` | The checked-in kernels are what the generator renders, with version and native directives. |
| `gen/spill` | The spill guard passes every generated kernel and fails a kernel past the cliff. |
| `gen/native` | Every module of the core is `--!native` and native lowering skips none of its functions. |
