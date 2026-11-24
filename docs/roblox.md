# Roblox backend

The backend in `backends/roblox/` runs the engine on a Roblox server. It ran in Roblox Studio 0.748 (Luau 0.748), Run
mode, server only: experiments R1 to R7, an EditableImage check and SmolLM2-135M Q8 chatting end to end, measured on a
desktop under a total CPU load of 50 to 100 percent. **Only Studio Run mode was tested. A published experience
has not been.** The bundle `tools/roblox/build.luau` writes has every change those runs needed except the one in
[Not in the backend](#not-in-the-backend). The runs used a patched copy, and this bundle itself has not been loaded in
Studio yet.

The tests in `tests/roblox/` run the backend against a Luau model of Roblox (`tests/roblox/mock.luau`). They show the
wiring is right and the results equal the single VM's. The model enforces the rules of this page that decide whether code
runs at all, see [What the model checks](#what-the-model-checks). It says nothing about timing or limits.

## Pieces

| File | Owns |
|---|---|
| `host.luau` | the `Host`, built from a `Platform` of `clock`, `wait`, `spawn`, `print` and `warn`, so the backend imports no service |
| `weights.luau` | weight sources: chunk ModuleScripts that return base64 text, and HTTP range requests |
| `worker.luau` | `bind`, which makes an Actor answer the requests of [parallel.md](parallel.md) as messages |
| `workers.luau` | the main side: one Actor per worker and a `Transport` to each |
| `roblox.luau` | `load(platform, options)`, which returns an `Engine` |
| `chat_service.luau` | a thin binding of the scheduler of [serve.md](serve.md): players to session ids, `bindPlayers`, `bindRemote`, one `pump` per frame |
| `bootstrap.server.luau` | the Script every worker Actor runs |

## Build

`lute run tools/roblox/build.luau [out]` writes the module tree to `out/src` and a `default.project.json` that Rojo maps
into `ReplicatedStorage.olelo`. A Roblox script finds a module by its place in the tree, so the bundler rewrites
`require("@src/ops/norm")` to `require(script.Parent.Parent.ops.norm)`, `require("@self/host")` in an `init.luau` to `require(script.host)`, and `const` to `local`. `src/` is the root module
and `backends/roblox/` the folder `roblox` in it. The bootstrap Script is the template that `workers.spawn` clones into
each Actor. `examples/roblox/server.server.luau` shows a server Script that loads a model and answers one prompt.

A ModuleScript holds at most 199,997 characters of source, the largest size accepted in Studio. A larger module
arrives empty and `require` fails with "Module code did not return exactly one value". The generator writes the unrolled
attention as one module per head size, `attention_unrolled_4`, `_22`, `_32`, `_43` and `_86` (8.7 to 111.5 KB) for the
f32 cache and `attention_q8_unrolled_4`, `_22`, `_32`, `_43` and `_86` (11.3 to 138.4 KB) for the Q8 cache, and
`ops/generated` (its `init.luau`) gathers them under `byLanes` and `q8ByLanes`. `build.luau` fails when any module is over the limit, and so does
`chunks.luau` for a chunk. The bundle is 103 files: 102 modules and the bootstrap Script.

## Weights

A container is cut into chunks of a fixed size. Chunk `n` is the ModuleScript named `n` in one folder, which returns the
bytes as base64 text. `lute run tools/roblox/chunks.luau model.olelo out [chunkBytes]` writes those files and `info.json`
with the numbers `SIZE` and `CHUNK_BYTES` of the example Script. Rojo maps the folder into a Folder.
`weights.fromModules` requires a chunk, decodes it into a buffer and keeps the last 4. A worker reads its weights and
their scales as two streams far apart in the container, so one cached chunk reloaded at every tile, and 2 workers
exceeded the HTTP request limit in 56 s. `weights.fromHttp` asks `HttpService:RequestAsync` for `bytes=a-b` of
a URL and accepts a `206` or a `200` with the whole file. It runs on a server only and yields.
`worker.SourceSpec` describes either source in a form a message can carry, so each worker opens its share by itself.

A script has about ten seconds of work per resumption before Studio stops it with "Script timeout: exhausted allowed execution time".
Decoding the chunk modules of a model takes longer than that in one go, and an HTTP source never reaches it because each request
yields. `weights.paced` wraps a source so that `read` waits a frame once `LOAD_SLICE` (0.1 s) of work has passed since the last wait.
`roblox.load` and the workers wrap the source they open, so a load from modules stays inside the limit on one script and in
each Actor.

| Source | Chunk size | Measured |
|---|---|---|
| ModuleScript | 65,536 bytes (default) fits, 149,991 bytes of base64 is the most one module carries | `require` of 150 KB 0.60 ms median, 2.26 ms first. Base64 decode 3.3 ms per 150 KB, 44 MB/s |
| HTTP | 4 MiB or more (the example sets 4,194,304) | bodies up to 1 GiB accepted, 71 to 95 MB/s on localhost. The request limit stopped 64 KiB ranged requests after 39 s, so 152 MB needs 37 requests of 4 MiB. The 65,536 default of `chunks.luau` is for modules |

A string of 253 symbols without NUL, quote, backslash, CR or LF carries 199,990 bytes per module, a third more than
base64, and reads at 1.5 GB/s against 44 MB/s for base64 decode. `chunks.luau` writes base64 only.

## Workers

`workers.start` creates an Actor per worker with a clone of the bootstrap Script and an ObjectValue `Root` pointing at the
bundle. The Script runs in the Actor's VM, so each worker has its own module cache and its own copy of the code, and
sends nothing but messages. The messages are:

- `hello` with the main Actor, which tells the worker where to send replies.
- `req` with an id, the op of [parallel.md](parallel.md) and its arguments.
- `rep` with the id, a flag and the value or the error text.

Buffers are copied on both sides, as a message does. The model of Roblox checks that the engine over 1 to 3 Actors gives
the greedy tokens and the chat of the single VM, also when a message takes two frames to arrive.

### Rules Roblox enforces

| Rule | Consequence |
|---|---|
| `Actor:BindToMessage` works only in a script that is a descendant of that Actor | `place.main` is an Actor with the main script as a child, and the Script that calls `roblox.load` runs under it. `examples/roblox/server.server.luau` starts itself again under an Actor. A bare Actor in `ServerScriptService` errors at once |
| A message sent before the worker script has called `BindToMessage` is dropped | the worker sets the attribute `ready` after its binds. `workers.spawn` waits for it (30 s limit), then a frame, before `hello` |
| `HttpService.RequestAsync` and `require` are refused in the parallel phase. `Instance.SetAttribute` and `Connection.Disconnect` are not safe there | `load` and `replica` go on the serial topic `sreq`, `attn`, `mlp` and the other ops on the parallel `req` |
| A message to a serial handler waits for the frame | tensor parallel needs the main script in the parallel phase, and the replies are bound with `BindToMessageParallel`. A serial main took a frame per exchange and 61 exchanges make a token: 2.0 tokens per second with 2 workers against 8.6 |
| A thread woken from a parallel reply handler continues in the parallel phase | `join` calls `Platform.restore` after it wakes, which the script sets to `task.desynchronize` or `task.synchronize`. The script's `wait` calls `task.desynchronize` after `task.wait` when it works in the parallel phase, and it synchronizes before it fires events or destroys Actors |
| A message handler stops the whole frame loop while it runs | the engine works in slices, see below |

The setup (workers, `load`, engine) runs in the serial phase and `task.desynchronize()` comes before `chat:send`.

### Slices

`slice` is the seconds of work between yields to the frame. Studio holds 60 fps with 12 ms of work per frame for 1, 4 and
8 Actors, 16 ms drops to 55 to 58 fps and 24 ms to 37 to 39 fps. `roblox.load` defaults to 0.012 (`roblox.SLICE`). On one
script SmolLM2 Q8 decoded 6.9 tokens per second at 12 ms against 4.6 at 8 ms (frames at 16.8 ms), and 8.5 with no
slice, which freezes the frames. With 4 workers the two settings were not clearly different.

`Chat.send` pulls one step of the reply per call of the iterator and runs no coroutine, so the budget's `task.wait`
yields the caller's thread and the iterator form returns its pieces on one script as `onText` does.

### Worker count

2 to 4 Actors. Decode costs about 0.85 ms per exchange at every Actor count and a token has 61 exchanges, so decode tops
out near 19 tokens per second and more Actors only add capacity for prefill. 8 Actors did worse than 4 (5.5 against 6 to
12.5 tokens per second on the packed container, with total CPU at 100 percent).

### The probe

A timed choice of the vector kernels flipped in 2 of the Studio runs (1 of 7 single script runs, which changed the text, and
1 `v12` worker run, where the load failed because that container needs the vector read). `probe.run` now repeats a failed
timing up to 3 times and keeps the best ratio. The main script's answer goes to the workers in `load`, so every VM runs
the same kernels. The probe has not been run in Studio since that change.

## Measured

Studio 0.748, Run mode, server only. Rates in MMAC/s are medians of 5 to 15 windows of 0.15 s.

| Question | Result |
|---|---|
| R1 kernel rates, 1536x576 | `f32_vector` 1491, `f32_buffer` 684, `f32_scalar` 487, `q8_p8` 1173, `q8_v12/R12S1U32` 2039, `q4_p4` 1276. The vector read is 3.1 times the scalar and 2.1 times the buffer kernel, and `q8_v12` is 1.7 times `q8_p8`. The probe picks the vector kernels |
| R2 Actors | an Actor runs a native script at the main script's rate, in the parallel and the serial phase. Without `--!native` or `@native` a kernel runs 13 to 26 times slower. 8 Actors reach 4.1 times one for `q8_p8` in the parallel phase and 1.1 times in the serial phase |
| R3 message round trip, main in the parallel phase | 25 to 30 us for one Actor, 250 us for 8, about 30 us per extra Actor. In the serial phase up to a frame (16.7 ms) |
| R4 buffer in a message | 22 us at 2 KiB, 423 us at 192 KiB, 1.8 ms at 1 MiB, 1.1 to 1.2 GB/s from 1 MiB. A SharedTable costs as much as the copy |
| R5 frame time | see Slices |
| R6 ModuleScript and HTTP | see Weights |
| R7 buffer memory | 3 GiB in buffers of 128 MiB without failure, stopped by the test's own cap. A single buffer over 1 GiB fails with "block too big". The ceiling was not found. A container is held in parts of at most 1 GiB |
| EditableImage | `WritePixelsBuffer` and `ReadPixelsBuffer` round trip a 32x32 RGBA image with every byte unchanged. A 1024x1024 write takes 1.0 ms, a read 2.4 ms. The backend does not use it |
| `CFrame` as a matmul path | no. 450 to 780 MMAC/s as a weight table against 1700 to 3000 for the kernels |

SmolLM2-135M-Instruct Q8, one 36 token prompt, greedy, 32 new tokens, context 512, HTTP range source in 4 MiB chunks,
slice 12 ms. Vector is the `v12` container, kernels `q8_v12/R12S1U32` for decode and `R6S2U32` for
prefill. Packed is the `p8` container. Medians of 2 to 4 runs, tokens per second:

| Container, configuration | Decode | Prefill |
|---|---|---|
| vector, one script | 14.1 | 22.4 |
| vector, 2 Actors | 19.8 | 43.9 |
| vector, 4 Actors | 25.4 | 62.2 |
| packed, one script | 7.9 | 15.1 |
| packed, 2 Actors | 12.9 | 23.9 |
| packed, 4 Actors | 15.8 | 35.2 |

Rates vary with CPU load. An earlier set of runs under heavier load gave 12.4 on one script and 19.6 with 4 Actors for
the vector container.

Loading takes 3 to 4 s on one script and 4 to 14 s with 1 to 6 Actors. Two runs with the same settings gave different
text when the probe chose the scalar read in one of them, because the attention path differs and rounding moves a near
tie.

### Where the time goes

One decode exchange, `q8` vector, medians in microseconds at 2, 4 and 8 Actors: worker compute 414, 265, 193. Delivery
97, 150, 216. Return 90, 147, 195. Wake of the joining thread 135, 144, 139. Main work between exchanges 102, 127, 131.
Compute shrinks with the Actor count and delivery plus return grows, so an exchange costs about 0.85 to 0.92 ms at each
count. The slowest Actor is 21 to 42 us behind the mean, so a straggler does not set the step time. The MicroProfiler
scopes (`debug.profilebegin`) that measured this are not in the modules. Each call logs a warning while the profiler is
off and decode fell from 17 to 4.3 tokens per second.

## Native code

Native lowering is decided per function. Studio 0.748 skipped these vector Q8 tiles (the module's other functions stay
native):

- `q8_v12/R12S2U32` runs at 125 MMAC/s against 2330 for `R12S1U32`. Main selects `R6S2U32` for vector Q8 prefill, which
  lowers natively in Studio, see [kernels.md](kernels.md).
- In a probe of 23 unrolled tiles, 1 token tiles failed from 4 row vectors with 22 steps and 2 token tiles failed at 4 rows
  for every unroll and at 1 row with 22 steps. The register count and the line count do not predict it, and the Lute
  `luau-compile` output does not either, so the Roblox lowering is checked in Studio.

## Not in the backend

| Change | Why |
|---|---|
| `chunks.luau` writing the 253 symbol encoding, and `weights.fromModules` reading it | a third more bytes per module and 34 times the decode rate. The Studio run built the string in a script. Rojo 7.7.0 refuses a file that is not UTF-8 (`rojo build` stops with "File was not valid UTF-8"), and the raw bytes 128 to 255 of the 253 symbols are not, so a chunk file cannot carry it. Only the 123 ASCII symbols (1 to 127 without quote, backslash, CR and LF) pass, which is 6.9 bits per character against 6 for base64 |

## What the model checks

`tests/roblox/roblox.test.luau` runs the bundle in `tests/roblox/mock.luau`, which enforces these rules and fails
without the change that follows them:

| Rule or change | Test |
|---|---|
| a module over 199,997 characters arrives empty, a module returns one value | every bundled module is under the limit, the attention aggregate gathers a module per head size, and requiring an empty module fails |
| `BindToMessage` only from a script under the Actor | `spawn` called outside the main Actor fails, the example Script starts itself under an Actor |
| a message sent before the topic is bound is dropped | `spawn` waits for `ready` and no message is lost, and gives up on a worker that never binds |
| `require`, `RequestAsync`, `SetAttribute` and `Destroy` are refused in the parallel phase | `load` is sent on `sreq` and the rest on `req`, the engine over Actors with the main script in the parallel phase gives the single VM's tokens |
| a thread woken by a parallel reply handler continues in the parallel phase | `Platform.restore` returns it, and without it `require` is refused |
| `fromChunks` keeps 4 chunks | two interleaved streams load each chunk once |
| the iterator form on one script with a slice | `Chat.send` returns the pieces `onText` gets |
| default slice | the budget slice is 0.012 |

The model cannot show what depends on Roblox itself: the HTTP request rate limit, that a serial handler waits for the
frame (so the 2.0 against 8.6 tokens per second is not reproduced), what native lowering skips, the rates, the frame time,
how a module reaches Studio (Rojo or another sync), and whether the real `task.synchronize` and `task.desynchronize`
behave as the model assumes.

## Chat service

The scheduler core knows no Player or Actor. `workers.startReplicas` makes replica Actors (the first `replica` request makes
an Actor a unit with an engine of its own, `worker.bind`) and a transport to each, which are the lanes. `chat_service.new`
takes those lanes, keys a player by its `UserId`, pumps the scheduler in a loop thread once per frame, closes a
conversation on `PlayerRemoving`, and answers a RemoteEvent with the pieces and the end of each reply.
`tests/roblox/chat_service.test.luau` runs it over replica Actors of the model. A step costs at least a frame, so the
decode rate of a lane is bounded by the frame rate unless `stepTokens` is raised, which is unmeasured.

## Not verified

- A published experience. Studio Run mode on one desktop is all that ran, so a live server's CPU, memory, request limit and
  allowed HTTP domains are unmeasured.
- The vector read on a live server. The Studio runs set `DebugLuauBufferVectorLibrary`, `DebugLuauBufferVectorFastcalls`
  and the native code generation limits in Studio's `ClientAppSettings.json`. Where the flags are off the probe picks the
  scalar kernels, which are about 3 times slower (R1), and a `v12` container does not load, so convert with the packed
  layout there.
- A chunk ModuleScript source for a whole container. The end to end runs read over HTTP, and R6 gives the per chunk costs.
- Rates without CPU load. Four worker runs of one build ranged from 5.8 to 12.5 tokens per second under CPU load,
  and 8 Actors with the current kernels were timed only with the profiler on.
- The Allow Mesh/Image APIs setting for EditableImage, which was neither read nor changed.
