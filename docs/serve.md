# Serving many conversations

The core is host agnostic: sessions are opaque string ids, nothing in it knows Lute, Roblox or a Player. A host gives it
lanes behind a transport, calls `pump` in a loop, and reads replies. The measurements are in
[benchmarks.md](benchmarks.md).

| Piece | File | Job |
|---|---|---|
| Chat steps | `src/api/chat.luau` | `begin` starts a reply and `step` runs one prefill chunk or some tokens, so a caller decides who goes next. `snapshot` and `resume` rebuild a conversation exactly from its token state. |
| Unit | `src/serve/unit.luau` | Conversations on one engine, a replica or a group of workers, by session id, as plain data over a transport. |
| Scheduler | `src/serve/scheduler.luau` | Users, queue, fairness, streaming, cancel, timeouts, placement and eviction over lanes. |
| Layout | `src/serve/layout.luau` | How many replicas against worker groups, from the model, threads, memory and users. |
| Prefix store | `src/api/prefix.luau` | One prefill of a shared system prompt for every conversation that starts with it. |
| Lute pool | `backends/lute/pool.luau` | The lanes of a layout as replica VMs or groups of worker VMs. |
| Roblox | `backends/roblox/chat_service.luau` | A thin binding: players to session ids, replica Actors as lanes. |

## Lanes and units

A lane is a unit behind a `host.Transport`, with one call in flight. Its unit is a replica (the whole model in one VM, no
exchanges) or the main worker of a tensor parallel group (`parallel.md`). Everything crossing a transport is a copy, so
a request carries the sampling options and text, never a callback.

## Scheduler

```luau
const sched = scheduler.new({ lanes = lanes, clock = clock, maxQueued = 64, queueTimeout = 30, requestTimeout = 120 })
const request = sched:submit("user7", "hello", { maxTokens = 64, onText = send, onDone = finished })
request:cancel()
sched:close("user7")
sched:run(wait) -- or call sched:pump() in the host's own loop
```

- **No thread.** `pump` collects what the lanes finished, enforces cancels and timeouts, places waiting users and gives
  each idle lane its next step. The host decides how often it runs.
- **Fairness.** Every user has a virtual time that grows with the seconds of lane time their steps took. An idle lane
  runs the step of the resident user with the least virtual time, and a waiting user takes the least loaded lane. A long
  prompt or reply cannot starve the others.
- **Slots.** A lane holds `slots` conversations. When none is free and a user waits, the least recently used idle
  conversation is dropped, and rebuilt from its token state when its owner comes back (`Chat:snapshot`, `resume`), with
  the shared prefix copied rather than run again.
- **Limits.** `maxPerUser` requests per user, `maxQueued` waiting requests (the rest end with `rejected`), `queueTimeout`
  and `requestTimeout`, and `cancel`. A request always ends with a `Reply` whose `finish` says why: `stop`, `length`,
  `context`, `text` (a stop string), `cancelled`, `timeout`, `queue_timeout`, `rejected`, `closed` or `error`.

## Prefix sharing

The runner saves the first N positions of a sequence under a key and copies them into an empty sequence. The store keeps
the token lists, finds the longest saved prefix of a new prompt and drops the least recently used. The cache lives where
the model runs, in the VM or in each worker, so no cache crosses a transport. A chat saves its ChatML system prefix after
its first prefill. Sharing is between the conversations of one lane, so each lane prefills a prefix once. Moving a prefix between lanes
would pay: 40 tokens of system prompt are 1.9 MB of f32 cache on SmolLM2-135M (47,520 B per position) against 2.5 s of
prefill at 16 tokens per second, but it needs export and import ops on every unit, worker and Actor, which are not built.

## KV cache formats

`kv = "f32"` stores `[kv head][position][hdPad]` f32. `kv = "q8"` stores int8 with an f32 scale per head and position
(amax over 127), for keys and values separately, and keys less a centre per head that is the
mean of the sequence's first 32 keys when that shrinks their largest value by 10% or more. A constant shift of all keys
changes every score of a query by the same amount, which the softmax does not see. The centre belongs to the cache: it is
copied with a snapshot and a saved prefix, and set again when a sequence starts from position 0.

A Q8 row has two layouts, `kvLayout` on the engine. `i8` is `headDim` bytes a row and is read by the scalar
`attention_q8`. `v3` packs three int8 values (offset binary) into each f32 lane as `c + b / 256 + a / 65536`, which is
exact in f32, so one vector read takes nine values and the generated `attention_q8_unrolled_<lanes>` kernel runs at about
the f32 kernel's rate (`ops/kvquant`, `tools/gen/attention_q8.luau`). A row is padded to whole vectors, 96 bytes for a head
of 64. The engine picks `v3` when the vector read is there and a generated kernel has the head size, else `i8`.

Bytes per position of the whole model (`2 * layers * kvHeads * x`):

| Format | x | SmolLM2-135M (30 layers, 3 KV heads) | Qwen2.5-0.5B (24 layers, 2 KV heads) |
|---|---|---|---|
| f32 | `hdPad * 4` (264 for headDim 64) | 47,520 B | 25,344 B |
| Q8, `i8` | `headDim + 4` (68) | 12,240 B | 6,528 B |
| Q8, `v3` | `vectors * 12 + 4` (100) | 18,000 B | 9,600 B |
| Q4, not built | `headDim / 2 + 4` (36) | 6,480 B | 3,456 B |

Plus a centre per layer for Q8, `kvHeads * headDim * 4` bytes. Quality, speed and the Q4 verdict are in
[benchmarks.md](benchmarks.md).

## Layout rule

`layout.choose(machine, model, users)` returns lanes, workers per lane and slots per lane. With `R` the replicas that fit
the threads and the memory (a replica takes the engine's estimate plus 150 MB for its VM):

- At least `R` users: `R` replicas, every lane one worker, `ceil(users / R)` slots. Replicas add up to the threads and
  workers do not.
- Fewer users: one lane per user, each a group of `min(wide, R // users)` workers, where `wide` is one worker per 50 MB of
  weights up to 4. The 50 MB is a fit to two models (SmolLM2-135M Q8 stops gaining at 3 workers, Qwen2.5-0.5B Q8 gains 3.6
  times at 4 and 3.7 at 8) and a default to change when more models are measured.

## Hosts

Lute: `pool.open(path, layout.choose(...), options)` returns the lanes and `close`. Roblox: `workers.startReplicas` makes
replica Actors, `chat_service.new` maps players to ids and pumps the scheduler once per frame. It is tested against
`tests/roblox/mock.luau` and has not run in Roblox Studio. The engine and Actor rates that did are in [roblox.md](roblox.md).
