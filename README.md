# olelo-luau

olelo-luau runs released chat language models on the CPU in pure Luau, in Lute and on Roblox servers. Its matrix kernels
are generated Luau that the compiler turns into native code. The core has no host imports, and two thin backends supply
files, workers and a clock.

This is the pure Luau version of Olelo. The next Olelo engine is a native C++ core with Luau embedded as its scripting
layer: the kernels run natively, and the generation loop, scheduling and tokenizing stay in Luau. It is a separate
codebase, and this repository does not share code with it.

## What it can do

- **Models.** SmolLM2-135M-Instruct, Qwen2.5-0.5B-Instruct, Qwen3-0.6B, Gemma 3 1B, LFM2.5-230M, LFM2.5-350M and
  MobileLLM-R1.5-360M chat in f32 and Q8 in Lute. SmolLM2 also runs in BF16 and Q4.
- **Correctness.** For SmolLM2, LFM2.5 and MobileLLM the tokens and f32 logits match the Hugging Face models. The Qwen and
  Gemma 3 families are checked on small random models of the same architecture. [docs/models.md](docs/models.md) lists
  each model, its formats and what it is checked against.
- **Quantization.** Q8_0 and Q4_0 weights equal llama.cpp's. A packed layout runs on every host, and a vector
  layout is faster where the vector read is fast. The KV cache is f32 or Q8.
- **Tensor parallel.** The layers divide among worker VMs in Lute ([docs/parallel.md](docs/parallel.md)).
- **Many chats at once.** A scheduler serves them from replica VMs and worker groups ([docs/serve.md](docs/serve.md)).
- **Roblox.** The backend ran in Roblox Studio 0.748 (Run mode, server only). SmolLM2-135M Q8 chatted at 14.1 tokens per
  second on one script and 25.4 with 4 Actors ([docs/roblox.md](docs/roblox.md)). It has not run in a published experience.
- **Tokenizers and templates.** Byte level BPE (GPT-2, Qwen and Llama 3 splits) and SentencePiece style BPE, with the
  ChatML, Gemma and Llama 3 chat templates.

## Speed and limits

Everything here is Luau running on the CPU, with no native library. At this speed the models that chat comfortably are
the small ones, 135M to 0.5B parameters, which suits running a model wherever only Luau is available.

Tokens per second in Lute with the vector flags, greedy, a 119 token prompt (116 for Qwen) and 48 new tokens (32 for
Qwen). Each cell is the median of 2 or 3 runs on a desktop with 16 logical CPUs. Rates varied by up to 2 times between
runs. Source: [docs/benchmarks.md](docs/benchmarks.md), "Q4 and Q8 on workers".

| Model | Format | One VM, decode | One VM, prefill | Best with workers, decode | Best with workers, prefill |
|---|---|---|---|---|---|
| SmolLM2-135M | Q8, vector | 16.55 | 23.54 | 19.24 (4 workers) | 43.94 (4 workers) |
| SmolLM2-135M | Q4 | 14.63 | 22.27 | 20.21 (3 workers) | 42.39 (8 workers) |
| Qwen2.5-0.5B | Q8, vector | 4.47 | 8.79 | 9.80 (8 workers) | 23.07 (8 workers) |
| Qwen2.5-0.5B | Q4 | 3.12 | 5.00 | 7.97 (8 workers) | 15.45 (6 workers) |

- **Workers.** The gain from more workers varies a lot. One run measured Qwen2.5-0.5B Q8 decoding 3.6 times faster at
  4 workers (3.1 to 11.2), and the run above measured 0.84 to 2.19 times at 2 to 8 workers.
- **Attention at long contexts.** It is a large part of a token. [docs/benchmarks.md](docs/benchmarks.md)
  has the rates by context length.
- **Running workers in Lute needs a Lute with the `vm.create` fix** ([tools/lute-fix/README.md](tools/lute-fix/README.md)).

## Try it

From a Hugging Face checkpoint to a chat. The first lines run in the repository root.

**1. A checkpoint.** A directory with `config.json`, `tokenizer.json`, `tokenizer_config.json` and `*.safetensors`.
The architecture is `llama` (SmolLM2 included), `qwen2`, `qwen3`, `gemma3_text`, `lfm2` (LFM2 and LFM2.5) or
`llama4_text` (dense layers, MobileLLM-R1.5). Any other `model_type` stops the converter with "unsupported model_type".

```sh
hf download HuggingFaceTB/SmolLM2-135M-Instruct --local-dir smollm2-135m
```

**2. Convert.** The converter needs Python with torch and transformers ([tools/convert/README.md](tools/convert/README.md)).
The repository holds no weights, so every model file comes from this step.

```sh
python tools/convert/convert.py smollm2-135m --out models/smollm2-135m-q8.olelo --recipe q8
```

| Recipe | Stores | Runs where |
|---|---|---|
| `f32` (default) | float32, the reference | everywhere |
| `q8` | Q8_0, packed layout | everywhere |
| `q8 --q8-layout v12` | Q8_0, vector layout, 1.4 times the file | only where the vector read is fast, otherwise the load fails |
| `q4` | Q4_0 layers, Q8_0 embedding and head | everywhere, the smallest |
| `bf16` | BF16 | everywhere |

`q8 --q8-layout v12` is the fastest for decode and prefill where it runs. Use `q8` when unsure.

**3. Chat in Lute.** The vector kernels need `buffer.readvectorf32x3` and its fast call, which Luau has behind two
flags. `tools/lute-fix/build.sh <lute checkout>` builds a Lute that sets them from `LUTE_SET_FFLAGS`. The Lute from
`foreman install` runs every container except `v12` ones, with slower kernels that read three scalars per vector.

```sh
export LUTE_SET_FFLAGS=DebugLuauBufferVectorLibrary=true,DebugLuauBufferVectorFastcalls=true
lute run examples/chat.luau models/smollm2-135m-q8.olelo
```

`examples/chat.luau <model.olelo>` takes `--prompt "text"` (answer once and exit), `--system "text"`, `--temp`, `--top-k`,
`--top-p`, `--min-p`, `--repeat`, `--max` (new tokens, 256), `--context` (positions, 2048), `--seed` and `--workers n`.
Without `--prompt` it reads lines, and `/reset`, `/stats` and `/quit` are commands.

With no sampling flag the example samples with temperature 0.7, top-p 0.9 and repetition penalty 1.1, and the load line
prints what it uses. `--temp 0` is greedy, and a small model loops in greedy mode. An answer cut off by `--max` ends mid
sentence, and the stats line shows finish `length`.

**4. Use it from Luau.** A script inside the repository, run like the chat, with the same calls as `examples/chat.luau`:

```luau
const engine = require("@src/api/engine")
const lute = require("@backends/lute")
const io = require("@std/io")

const e = engine.new({
    host = lute.host(),
    source = lute.fileSource("models/smollm2-135m-q8.olelo"),
    context = 2048,
})
const chat = e:chat({ system = "You answer in one sentence." })
for piece in chat:send("What is a hash table?", { temperature = 0.7, topP = 0.9, repetitionPenalty = 1.1, maxTokens = 128 }) do
    io.write(piece)
end
print(chat:stats().decodeRate)
```

The library samples greedily when `send` gets no options, which keeps tests and benchmarks deterministic.

**5. Run it on Roblox.** Convert with `q8` unless the server is known to have the vector read. Then bundle the source and
cut the container into chunks:

```sh
lute run tools/roblox/build.luau build/roblox
lute run tools/roblox/chunks.luau models/smollm2-135m-q8.olelo build/weights 65536
```

Rojo maps `build/roblox` (its `default.project.json`) to `ReplicatedStorage.olelo` and `build/weights` to a Folder
`ReplicatedStorage.weights`. A script reads the weights from those chunk ModuleScripts, or by HTTP range requests with
chunks of 4 MiB or more (`weights.fromHttp`). `examples/roblox/server.server.luau` is the Script to put in
`ServerScriptService`. Fill in `SIZE`, `CHUNK_BYTES` and `WORKERS` from `build/weights/info.json` (2 to 4 Actors is best).
[docs/roblox.md](docs/roblox.md) has the weight sources, the Actor and phase rules and the measured rates.

## Tests

`lute run tests/run.luau` runs the tests, and `tools/ci.sh` runs every check in all three flag modes
([CONTRIBUTING.md](CONTRIBUTING.md)).

## Documents

- [docs/architecture.md](docs/architecture.md) the layers and why
- [docs/models.md](docs/models.md) the models that run and how each is checked
- [docs/container.md](docs/container.md) the model file
- [docs/structure.md](docs/structure.md) what each directory owns, its contracts and how to extend it
- [docs/kernels.md](docs/kernels.md) the generated kernels
- [docs/parallel.md](docs/parallel.md) dividing a model among workers
- [docs/serve.md](docs/serve.md) serving many conversations, the KV cache formats and the layout rule
- [docs/roblox.md](docs/roblox.md) the Roblox backend, what was measured in Studio and what is unverified
- [docs/benchmarks.md](docs/benchmarks.md) measured rates and the method
- [CONTRIBUTING.md](CONTRIBUTING.md) checks and rules
- [CHANGELOG.md](CHANGELOG.md) what changed

## Source

https://github.com/ilyas-mallah/olelo-luau

## License

MIT, see [LICENSE](LICENSE).
