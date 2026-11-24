# Structure

What each directory owns, what it may import, and the contracts between layers. The design behind it is in
[architecture.md](architecture.md).

## Layers and imports

A layer imports only the layers listed for it. The core under `src/` imports nothing from lute, Roblox or `backends/`, so it
runs on any Luau host with `buffer`, `vector` and `bit32`.

| Layer | Path | Owns | Imports |
|---|---|---|---|
| util | `src/util/` | the JSON decoder | nothing |
| tensor | `src/tensor/` | formats, buffer parts, tensors, activation quantization | nothing |
| kernels | `src/kernels/` | generated kernels, references, registry, probe | tensor |
| container | `src/container/` | reads the `.olelo` file into tensors and blobs | util, tensor, host |
| ops | `src/ops/` | norm, rope, attention (plain, vector, and unrolled per head size in `generated/`), gated MLP, embedding lookup, activation preparation, linear, short convolution | tensor, kernels, runtime |
| runtime | `src/runtime/` | the time budget that yields to the host | host |
| tokenizer | `src/tokenizer/` | byte BPE, the pre-tokenizer, chat templates | container |
| sampling | `src/sampling/` | the random generator and the sampler | nothing |
| arch | `src/arch/` | what differs per model family | ops |
| model | `src/model/` | loaded weights or a worker's share of them, sessions with the KV cache, the forward pass, the memory plan, the `Runner` that runs it | tokenizer, arch and every layer below |
| parallel | `src/parallel/` | the division of a model among workers, the worker, the main worker, the in process transport (see [parallel.md](parallel.md)) | model, container, ops, host |
| api | `src/api/` | `Engine` and `Chat` | everything above |
| serve | `src/serve/` | the scheduler of many conversations over lanes, the layout rule, the replica unit (see [serve.md](serve.md)) | api, tokenizer, host |
| host | `src/host.luau` | the `Host` and `WeightSource` types | nothing |
| entry | `src/init.luau` | the public table | every layer |
| backends | `backends/<host>/` | one folder per host, builds a `Host`, a `WeightSource` and the transport to workers (Roblox: see [roblox.md](roblox.md)) | `src`, the host's libraries |
| generator | `tools/gen/` | kernel generator, spill guard | `src/kernels/types` for types, lute |
| converter | `tools/convert/` | Python, checkpoint to container, reference dumps and fixtures | none, a dev tool |
| tests | `tests/` | harness, suites, fixtures and the model of Roblox | `src`, `backends`, `tools`, lute |
| bench | `tools/bench/` | kernel and generation benchmarks | `src`, `backends/lute`, `tests/data`, lute |

Cross directory requires use the aliases in `.config.luau` (`@src`, `@backends`, `@tests`, `@tools`). A sibling uses
`./name`. An `init.luau` names its children with `@self/name`, because `init.luau` is the module of its folder and `./name`
there would look beside the folder. `tools/roblox/build.luau` rewrites the aliases, `@self` included, to instance paths
for Roblox, where a folder with an `init.luau` is a ModuleScript that holds its children.

## Naming

Files are lowercase `snake_case.luau`, a test is `<name>.test.luau`. A module returns one table of functions.
Functions are `camelCase`, constants `UPPER_SNAKE`.

Every folder of `src/` has an `init.luau`, its public face, so a caller writes `require("@src/ops")`. A face holds the
folder's types, re-exported with `export type X = types.X`, and the modules a caller uses, and no logic. Where a folder
has one module that is its whole job (`arch`, `container`, `model`, `tensor`, `tokenizer`, the backends), that module is
the `init.luau` and carries its logic beside the re-exports. Two generated folders differ. `ops/generated/init.luau` is
the generated aggregate of the unrolled attention, `byLanes` for the f32 cache and `q8ByLanes` for the Q8 cache. `kernels/generated/` has no `init.luau`, because
`kernels/registry.luau` is its face and pairs every generated module with its reference.

Types live in `<folder>/types.luau`, and a file requires `./types` and a caller `@src/<folder>/types`. A type stays in
its file only for a reason:

- `src/host.luau` is the host contract and holds nothing but its types.
- `src/model/config.luau` keeps `Config` because `arch/types.luau` needs it and `model/types.luau` needs `Arch`, so a
  `Config` in `model/types.luau` would be a require cycle.
- The other exported types that stay (`probe.Result`, `graph.LayerHook`, the `Options`, `Stats` and `Coordinator` of
  `parallel/coordinator.luau`, `sampler.Sampler`, the `Options` of `serve/layout.luau` and `serve/scheduler.luau`,
  `formats.Format`, `store.PartPlan`, `tokenizer.Stream`) are named by no other file.

## Luau policy

- **`--!native`.** Every file under `src/` starts with `--!native`, and the generated ones also with `--!optimize 2`.
  The choice is per file, not per function, so no function is forgotten. `tests/gen/native.test.luau` checks the header
  of every file under `src/` and compiles each with `luau-compile` to prove that native lowering skips no function.
- **`const`.** Luau 0.711 and later accept `const` and `const function`, and so does the pinned Lute. Use `const` for
  every binding that is assigned once, and `local` for accumulators and variables that change.
- **Strictness.** Strict everywhere. A generated module that calls `buffer.readvectorf32x3` is `--!nonstrict`, because
  the type checker does not know that function in 0.747.
- **The vector read.** Luau 0.747 has `buffer.readvectorf32x3` and its fast call behind `DebugLuauBufferVectorLibrary`
  and `DebugLuauBufferVectorFastcalls`. `tools/lute-fix/` builds a lute that sets them, and the probe selects kernels at
  startup, so code that needs the flags is never chosen without them.

## Contracts

### Tensor

`formats` names the formats (`f32`, `bf16`, `q8`, `q4`) and their layouts, and gives byte sizes and address helpers.
A tensor has `rows`, `cols`, `kpad` (the stored row length, a multiple of the kernel's K step) and a list of parts.
A part is `{ buf, offset, scales, scaleOffset, rowStart, rowCount }`. A part never exceeds 2^30 bytes and is cut on a
tile row boundary. Rows past `rows` up to a whole tile are stored, and for Q4 their nibbles are filled with the byte
`0x88` and for Q8 the bytes with `0x80`, which are weight zero.

### Container and weight sources

`docs/container.md` has the file format. A `WeightSource` is `size()` and `read(offset, length, dst, dstOffset)` over the
logical container bytes, and a backend builds one from a file, chunks or HTTP bodies. Weights are read straight into the
buffers of a tensor's parts.

### Activations

Every activation row is stored padded to a multiple of 24 floats (`model/config.luau`, `ACT_PAD`), the least common
multiple of the kernels' column steps and row tiles. One buffer is then the output of a product and the input of the
next, and the padding stays zero because only real values are written. A forward pass over `n` tokens runs the products
on `n` rounded up to the kernel's tile tokens, and the extra row is never read.

### Architecture plug-ins

An `Arch` (`src/arch/types.luau`) is a table of fields the shared block reads: bias on q, k and v, per head q and k norm,
the norm form, the embedding scale, the MLP activation and the attention window per layer. A new family adds a table, the
converter's role names for its checkpoint, and a parity fixture. It never edits `model/graph.luau` unless it needs a new
op.

Families in `src/arch/`: `llama` (and SmolLM2), `qwen2` (biases on q, k and v), `qwen3` (per head q and k norm) and
`gemma3` (a norm after each block, local layers with a window and their own rope base, `1 + w` norms, a score scale from
`attnScalar`), `lfm2` (short convolution layers, see below) and `llama4` (the Llama decoder with a q and k norm that has
no weight, from the dense Llama 4 text layout of MobileLLM-R1.5). The Llama 4 checkpoint rotates adjacent pairs of a head,
so the converter reorders the rows of q and k to the half split form, which leaves every score unchanged, and the rope
op stays one. Its tokenizer is the Llama 3 one: the `llama3` pre-tokenizer, merges skipped for a piece that is a whole
token, and the `llama3` chat template with `<|begin_of_text|>` first.

The Qwen pre-tokenizer is the `qwen2` kind in `tokenizer/pretok.luau`. Qwen3 chats with thinking off: the converter stores the
empty think block as the template's `generationPrefix`, which `templates.generationPrompt` appends after the assistant
header. A chat keeps its cache between turns, so earlier turns keep their empty think block, where the Hugging Face
template drops it from history. `Chat:load(messages)` renders a history the way the template does, keeps the cache
positions that hold the same tokens and prefills the rest, which also serves editing or resuming a conversation. Text is not Unicode normalized (NFC) before the split, so a decomposed accent tokenizes differently from the
reference.

A short convolution layer (`config.layerTypes`, `ops/shortconv.luau`) has `Layer.conv` in place of the attention
projections. The session keeps its `B * x` of every position in `cache.conv[layer]` and `convBlock` in `model/graph.luau`
runs it between the same norm and residual as an attention block. Its keys and values are empty buffers.

Gemma's tokenizer is the `spm` kind: the text is one piece, spaces become U+2581, a character outside the vocabulary
becomes its byte tokens and `tokenizer/bpe.luau` merges with a heap, lowest rank first and leftmost among equals. Added
tokens such as runs of newlines are found before the split, longest first. `Engine:tokenize` adds the model's `<bos>`,
the chat prompt carries its own.

### Kernel

```luau
type Kernel = (
	w: buffer, wOff: number, ws: buffer?, wsOff: number,   -- weight part and its scales
	x: buffer, xs: buffer?,                                 -- activations as the kernel's `activation` says, see below
	y: buffer, yBase: number, yStride: number,              -- output, token t row i at yBase + (t * yStride + i) * 4
	row0: number, rows: number, K: number, T: number       -- rows in whole tiles, K the stored row length
) -> ()
```

A kernel writes `y` and returns nothing. Scales are f32, one per 32 weights. Q4 stores nibbles, the weight being the
nibble minus 8, in the packed layout of [kernels.md](kernels.md).

A registry entry's `activation` says what `x` and `xs` hold: `f32` rows with no `xs`, `int8` rows with one f32 scale per
token, `packed`, the Q8 tile kernel's f64 values with a sum and a scale per group of 32, or `vsplat`, the vector Q8
kernel's splat vectors (`ops/prepare.luau` and `ops/prepare_vector.luau` write them, [kernels.md](kernels.md) has the
layouts).

### Registry

`registry.all()` lists entries `{ id, format, layout, R, S, U, kstep, needs, fn, reference }`. `registry.select(format,
caps, prefill, layout)` returns the first entry of a preference list that the capabilities allow. `probe.run()` returns
`vectorRead` and the measured rates.

## Adding things

- **A kernel tile.** Add a row to `tools/gen/tiles.luau`, run `lute run tools/gen/main.luau`, run the guard, and add
  the tile to the preference list in `registry.luau`. The kernel tests cover every registry entry.
- **A format.** Add it to `formats`, a family module under `tools/gen/families/`, a reference under
  `src/kernels/reference/`, and a test file under `tests/kernels/`.
- **An architecture.** Add `src/arch/<name>.luau`, register it in `arch/init.luau`, add its tensor roles to
  `tools/convert/arch.py`, make a tiny random fixture with `tools/convert/make_fixtures.py`, and add a parity test.
- **A backend.** Add `backends/<host>/init.luau` that returns `host()` and a `WeightSource` constructor. The core stays
  unchanged, and the backend is the only place that imports host libraries.
- **A test suite.** Add the file and list it in `tests/run.luau`. There is no discovery.
