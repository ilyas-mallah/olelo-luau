# Container format

The `.olelo` file is what the runtime reads. The converter (`tools/convert/convert.py`) writes it from a Hugging Face
checkpoint, so the runtime never parses safetensors or GGUF and never repacks weights.

## Layout

| Bytes | Content |
|---|---|
| 0 to 7 | magic `OLELO` and three zero bytes |
| 8 to 11 | container version, u32 little endian, now 1 |
| 12 to 15 | header length `H`, u32 |
| 16 to 16 + H | header, JSON |
| next multiple of 64 | the data area, offsets in the header count from here |

Every tensor part and blob starts on a multiple of 64 bytes in the data area. The runtime refuses another version with
the version it reads, so a new converter and an old runtime fail with a clear message.

## Header

- `converter`: the converter name and version.
- `arch`: the architecture plug-in: `llama`, `qwen2`, `qwen3`, `gemma3`, `lfm2` or `llama4`.
- `config`: `hidden`, `layers`, `heads`, `kvHeads`, `headDim`, `intermediate`, `vocab` (the rows of the embedding),
  `ropeTheta`, `rmsEps`, `maxPositions`, `tieEmbeddings`, `activation`. Gemma 3 adds `slidingWindow`, `slidingPattern`,
  `ropeLocalTheta` and `attnScalar`. LFM2 adds `layerTypes` (`attention` or `conv` for each layer) and `convKernel`.
- `tokenizer`: `kind`, `specials` (id and text) and `eos` ids. `bytebpe` adds `pretokenizer` (`gpt2` or `qwen2`),
  `splitDigits`, `addPrefixSpace` and `ignoreMerges` (a piece that is a whole token skips the merges). A `pretokenizer` is `gpt2`,
  `qwen2` or `llama3` (the Qwen split with numbers in pieces of up to three digits). `bos` and `addBos` say which token
  the tokenizer puts before a text, for `bytebpe` as for `spm`.
- `chat`: the declarative chat template. `kind` `chatml` has the default system message and an optional
  `generationPrefix`, `llama3` (header markers, `<|eot_id|>` ends a turn) has the default system message, and `gemma`
  has none. Every kind may have `opening`, the text that starts a conversation, the beginning of sequence token the
  template writes.
- `tensors`: name, `format`, `layout`, `rows`, `cols`, `kpad`, `rowTile` and `parts` of `{ offset, rowStart, rows, bytes }`,
  and for a quantized tensor also `scaleOffset` and `scaleBytes` per part.
- `blobs`: name to `{ offset, bytes }`.

## Tensors

Names are the roles the converter maps checkpoint names to: `embed`, `final_norm`, `head` (absent when tied), and per
layer `layers.N.` followed by `attn_norm`, `q`, `k`, `v`, `o`, `mlp_norm`, `gate`, `up`, `down`, and where the
architecture has them `q_bias`, `k_bias`, `v_bias`, `q_norm`, `k_norm`, `post_attn_norm`, `post_mlp_norm`. A short convolution layer has `attn_norm`, `conv_in` (the input projection, 3 times
`hidden` rows), `conv_taps` (the filter of every channel, `convKernel` f32 values each, one `vector`), `conv_out`,
`mlp_norm`, `gate`, `up` and `down`, and no attention tensors. A matrix has `rows`
padded to a multiple of `rowTile` and columns padded to `kpad`, with zeros, so a kernel has no tail. A part holds whole
tiles of rows and at most 2^30 bytes of weights. A `vector` tensor is the unpadded f32 values of a norm weight.

## Blobs

- `tokenizer.vocab`: u32 count, then per id a u16 length and the token text. For `bytebpe` the bytes are mapped to
  printable characters, for `spm` the text is as the tokenizer file has it, a space is U+2581 and a byte is `<0xXX>`.
- `tokenizer.merges`: u32 count, then per merge in rank order the u32 ids of the two parts and of the result.
- `unicode`: three range lists, letters, numbers and whitespace, each a u32 count and then pairs of u32 first and last
  code point. The pre-tokenizer reads them because Luau patterns work on bytes.

## Formats

The header names the format and layout per tensor, so a recipe adds a format without a new version.

- `f32`, layout `rowmajor`: rows of `kpad` floats, rows padded to 8 and columns to 24 with zeros.
- `q8`, layout `p8`: Q8_0 in the packed tile layout of [kernels.md](kernels.md). Rows padded to 8, columns to 32. A part
  stores its weights, one byte each, at `offset` and its scales, 4 bytes each, at `scaleOffset`.

- `q8`, layout `v12`: the same weights in the vector layout of [kernels.md](kernels.md), 3 weights per f32 lane. Rows
  padded to 12, columns to 32. `convert.py --q8-layout v12` writes it, `repack_q8v.py` copies a `p8` container into it.
  The runtime reads it only where the vector read is fast.
- `bf16`, layout `rowmajor`: rows of `kpad` 16 bit values, rows padded to 8 and columns to 24.
- `q4`, layout `p4`: Q4_0 in the packed tile layout of [kernels.md](kernels.md). Rows padded to 8, columns to 32. A part
  stores its weights at `offset` and its scales, 4 bytes each, at `scaleOffset`. A container with the older layout
  `rowmajor` is converted with `tools/convert/repack_q4.py`.

`convert.py --recipe f32`, `bf16`, `q8` and `q4` write them. Q8 stores every matrix, the embedding and the output matrix
included. Q4 stores the layer matrices as Q4_0 and the embedding and the output matrix as Q8_0, and a runtime model needs
only its layer matrices to share a format. Norm weights and biases stay f32.
