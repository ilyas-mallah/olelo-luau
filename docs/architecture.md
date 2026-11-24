# Architecture

olelo-luau runs a released decoder language model on the CPU in pure Luau. The speed comes from generated kernels that Luau
compiles natively, and from the buffer layout they read.

## Layers

| Layer | Owns |
|---|---|
| tensor | buffers, parts of at most 1 GiB, formats and layouts |
| kernels | matrix products per format, the reference, kernel selection |
| ops | the math of one step over tensors |
| model | the decoder loop, memory plan, KV cache |
| arch | what differs per model family |
| tokenizer | text to ids and back, chat templates |
| sampling | logits to a token id |
| runtime | step order, yielding, worker calls |
| api | `Engine` and `Chat`, what callers use |
| serve | many conversations at once: lanes, a queue and the layout rule, see [serve.md](serve.md) |

A layer imports only the layers below it, and [structure.md](structure.md) has the contracts of the layers that exist. Hosts differ only in the files they read, the workers they start and the clock, and each supplies
those through a small table, so the core never imports a host.

## Layers that are not attention

LFM2 replaces most attention layers with a gated short convolution: the input projection gives two gates and a value, the
value times the first gate is filtered along the positions by a filter of 3 taps per channel, and the second gate times
the result goes through the output projection. The state of such a layer is the last 2 values of `B * x`, but the runtime
keeps `B * x` of every position instead, in a per layer history the size of one activation row per position. That makes
the layer stateless the way attention over a KV cache is, so a rewind, a snapshot and a restore need no recomputation, at
a cost of `hidden * 4` bytes a position for each such layer. Models with these layers do not divide among workers.

## Why a generator

A fast Luau matrix kernel is unrolled straight line code with constant offsets, one function per tile. Writing a few
dozen of those by hand does not scale, and one stray extra live value pushes a function past the native spill limit.
`tools/gen/` writes them from tile data, and `tools/gen/guard.luau` fails the build when a function would run
interpreted. See [kernels.md](kernels.md).

## Why one buffer layout

Luau caps a buffer at 1 GiB, so a tensor is a list of parts cut on tile row boundaries, and a kernel works on a row range
inside one part. Rows are padded so a kernel has no tail, and quantized weights keep their scales in a second buffer.
