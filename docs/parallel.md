# Tensor parallel inference

A model runs in one VM, or divided among workers that each hold a share of the weights. The division is the same on
every host. Only the line between the workers differs.

## Division

`parallel/split.luau` gives each of P workers:

| Part | Share |
|---|---|
| Query heads | a run of heads, the first workers one head longer when P does not divide them |
| KV heads | the ones those query heads read, so a KV head whose group of query heads is cut between two workers is in both |
| MLP | a run of columns, in groups of 32 |
| Output matrix | a run of vocabulary rows, in groups of 8 |

A worker holds the matching weights only. `q`, `k`, `v`, `gate` and `up` keep the rows of its share, `o` and `down`
keep the columns. A quantized model needs a head dimension that is a multiple of 32, so that `o` is cut on a quantized
group. `container.loadSlice` reads only the bytes of a share from the container, so a worker never holds a whole matrix.
Its KV cache holds its KV heads only.

The main worker holds the embedding table, the norm weights, the residual stream, the token list and the sampler. It
holds no layer weights. Per layer it normalizes the stream, sends the rows to every worker, adds the partial results of
the `o` projection, applies the post norm of Gemma 3 where there is one, adds the result to the stream, and does the same
for the MLP. Two exchanges per layer and one for the logits.

A worker's partial result is a sum over its columns of `o` or `down`. The sum of the partial results equals the whole
product up to the order of the additions, and equal to it exactly when one worker holds everything. A quantized activation is
quantized per group of 32 with the same values whatever the division, so the division does not change the rounding.

## Messages

A request is a call of a function of the worker with the arguments below, a reply is its return value. All sizes of
rows are `count` rows at the stride of an activation row, which is the hidden size rounded up to 24.

| Op | Request | Reply |
|---|---|---|
| `load` | container path, index, number of workers, context, chunk, vector flag | bytes of weights held |
| `attn` | sequence, layer, first position, count, normalized rows | the partial sum of the output projection |
| `mlp` | sequence, layer, count, normalized rows | the partial sum of the down projection |
| `logits` | sequence, one normalized row | the worker's rows of the logits |
| `trim` | sequence, length | none, forgets the positions from `length` |
| `free` | sequence | none |
| `stats` | none | seconds and calls per op, bytes in and out, bytes of weights |

A worker creates the cache of a sequence on its first request. `attn` of layer 1 must start where its cache ends. The
`logits` reply is the whole part of the vocabulary and not a list of candidates, which keeps the sampler exact.

## Ownership

1. A buffer has one owner, the VM that created it. No buffer is shared between VMs.
2. A buffer passed to a transport is copied on both hosts, in the request and in the reply. The sender may reuse it as
   soon as the call returns.
3. A worker creates its weights and cache from its own reads of the container. They never cross a boundary.
4. A worker writes its partial result to a buffer of its own and returns a copy. The main worker adds the copies into its
   residual stream and drops them.
5. The main worker holds the stream, the tokens and the sampling state. A worker holds nothing of the conversation
   except the cache rows of its heads.

## Transports

A `host.Transport` is `call(op, ...)` returning a future, `join(futures)` returning the replies in order, and `close()`.

- `parallel/transport.luau` runs workers in the calling VM and copies like a real transport. The tests use it.
- `backends/lute/init.luau` `spawnWorkers(path, n, options)` starts n worker VMs with `vm.create` from
  `backends/lute/worker.luau`. Each worker opens the container file itself. A request runs in a thread of its own,
  since a call into another VM yields its caller, and `join` polls the scheduler. It needs a Lute with the fix for the
  `vm.create` crash (`tools/lute-fix/README.md`).

A pass that waits for workers yields to the host's scheduler, so `Chat.send` does not wrap it in a coroutine of its own.
Over workers it runs the whole reply in the caller's thread and hands the text out afterwards. `onText` still streams.

## Use

```luau
const transports = lute.spawnWorkers(path, 4, { context = 2048, chunk = 32 })
const engine = engineModule.new({ host = host, source = lute.fileSource(path), transports = transports })
```

`context` and `chunk` must match the engine's. `engine:close()` ends the workers. The chat example takes `--workers n`.
Measurements are in [benchmarks.md](benchmarks.md).
