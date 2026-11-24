--!nonstrict
-- Times attention over a synthetic cache: `lute run tools/bench/attention.luau [module] [heads] [kvHeads] [headDim]
-- [positions] [tokens] [repeats]`. `module` is a file of `src/ops` such as `attention_vector`. Prints one `ATT` line
-- with nanoseconds per query head and position, the median of the repeats, and a checksum of the output.
const data = require("@tests/data")
const process = require("@std/process")
const opsTypes = require("@src/ops/types")

const name = process.args[2] or "attention_vector"
const heads = tonumber(process.args[3]) or 9
const kvHeads = tonumber(process.args[4]) or 3
const headDim = tonumber(process.args[5]) or 64
const positions = tonumber(process.args[6]) or 140
const tokens = tonumber(process.args[7]) or 1
const repeats = tonumber(process.args[8]) or 15

const hdPad = (headDim + 2) // 3 * 3
const modules = {
    attention = require("@src/ops/attention"),
    attention_vector = require("@src/ops/attention_vector"),
    attention_unrolled = {
        attend = require("@src/ops/generated").byLanes[hdPad // 3],
    },
}
const attend = modules[name].attend
const context = 1024
const rng = data.newRng(7)
const q = buffer.create(tokens * heads * hdPad * 4)
const keys = buffer.create(kvHeads * context * hdPad * 4)
const values = buffer.create(kvHeads * context * hdPad * 4)
const function fill(b: buffer, rows: number)
    for i = 0, rows - 1 do
        for d = 0, headDim - 1 do
            buffer.writef32(b, (i * hdPad + d) * 4, data.nextFloat(rng, 1))
        end
    end
end
fill(q, tokens * heads)
fill(keys, kvHeads * context)
fill(values, kvHeads * context)
const shape: opsTypes.Shape = {
    heads = heads,
    kvHeads = kvHeads,
    group = heads // kvHeads,
    headBase = 0,
    kvBase = 0,
    headDim = headDim,
    hdPad = hdPad,
    context = context,
    scale = 1 / math.sqrt(headDim),
    window = nil,
}
const stride = heads * headDim
const out = buffer.create(tokens * stride * 4)
const scores = buffer.create(context * 4)
const acc = table.create(hdPad, 0)
const pos0 = positions - tokens

const times = {}
for r = 1, repeats do
    const t0 = os.clock()
    local calls = 0
    while os.clock() - t0 < 0.1 do
        for _ = 1, 20 do
            attend(shape, q, tokens, keys, values, pos0, out, stride, scores, acc)
        end
        calls += 20
    end
    times[r] = (os.clock() - t0) / calls
end
table.sort(times)
local sum = 0
for i = 0, tokens * stride - 1 do
    sum += buffer.readf32(out, i * 4) * ((i % 7) + 1)
end
local pairs = 0
for t = 0, tokens - 1 do
    pairs += heads * (pos0 + t + 1)
end
print(
    string.format(
        "ATT %s heads=%d kv=%d hd=%d positions=%d tokens=%d ns_per_head_pos=%.1f us_per_call=%.1f csum=%.9g",
        name,
        heads,
        kvHeads,
        headDim,
        positions,
        tokens,
        times[(#times + 1) // 2] / pairs * 1e9,
        times[(#times + 1) // 2] * 1e6,
        sum
    )
)
