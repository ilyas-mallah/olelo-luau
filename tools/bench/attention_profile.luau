--!nonstrict
--!native
-- Where the attention of one decode token goes, by part, for the shapes of SmolLM2-135M and Qwen2.5-0.5B:
--
--   lute run tools/bench/attention_profile.luau [kv=f32|q8v|q8] [shapes=smol,qwen] [positions=128,512,1024,2048] [repeats=5] [spread=1]
--
-- Runs the attention op over every layer's own synthetic cache, like a decode pass, so the caches stream from memory.
-- The f32 cache runs the unrolled vector kernel ended after the dot products, after the softmax and whole (the
-- generator's `stage`), and the parts are the differences. With `kv=q8v` it runs the same stages of the kernel over the
-- vector layout of the Q8 cache (`attention_q8_unrolled`). With `kv=q8` it runs a replica of the scalar `attention_q8`
-- over the int8 layout with the weight scale pass as its own part, checked against the real kernel. Prints one `APART`
-- line per shape and position, in microseconds per token over all layers, and the MMAC/s of each part. Run it pinned to
-- CPUs 14 and 15, see docs/benchmarks.md.
const data = require("@tests/data")
const fs = require("@std/fs")
const gen = require("@tools/gen/attention")
const genQ8 = require("@tools/gen/attention_q8")
const kvquant = require("@src/ops/kvquant")
const process = require("@std/process")
const unrolled = require("@src/ops/generated")

const kv = process.args[2] or "f32"
const shapeNames = {}
for piece in string.gmatch(process.args[3] or "smol,qwen", "[^,]+") do
    table.insert(shapeNames, piece)
end
const positions = {}
for piece in string.gmatch(process.args[4] or "128,512,1024,2048", "[^,]+") do
    table.insert(positions, tonumber(piece))
end
const repeats = tonumber(process.args[5]) or 5
const spread = tonumber(process.args[6]) or 1

const SHAPES = {
    smol = { heads = 9, kvHeads = 3, headDim = 64, layers = 30 },
    qwen = { heads = 14, kvHeads = 2, headDim = 64, layers = 24 },
}

--- The replica of `attention_q8.attend`, ended after `stage` parts: 1 scores, 2 softmax, 3 weight scales, 4 values.
const function attendQ8(stage, s, q, tokens, keys, values, pos0, out, outStride, scores)
    const keyScales, valueScales = s.keyScales, s.valueScales
    const hd, hdPad, context = s.headDim, s.hdPad, s.context
    const group, headBase, kvBase = s.group, s.headBase, s.kvBase
    const scale = s.scale
    const hd4 = hd // 4 * 4
    const hd8 = hd // 8 * 8
    for t = 0, tokens - 1 do
        const last = pos0 + t
        const first = 0
        const count = last - first
        for h = 0, s.heads - 1 do
            const g = (headBase + h) // group - kvBase
            const qb = ((t * s.heads) + h) * hdPad * 4
            const rowBase = g * context
            local maxScore = -math.huge
            for j = 0, count do
                const row = rowBase + first + j
                const kb = row * hd
                local a, b = 0, 0
                for d = 0, hd4 - 1, 4 do
                    const qo = qb + d * 4
                    a += buffer.readf32(q, qo) * buffer.readi8(keys, kb + d) + buffer.readf32(q, qo + 4) * buffer.readi8(
                        keys,
                        kb + d + 1
                    )
                    b += buffer.readf32(q, qo + 8) * buffer.readi8(keys, kb + d + 2) + buffer.readf32(q, qo + 12) * buffer.readi8(
                        keys,
                        kb + d + 3
                    )
                end
                for d = hd4, hd - 1 do
                    a += buffer.readf32(q, qb + d * 4) * buffer.readi8(keys, kb + d)
                end
                const dot = (a + b) * buffer.readf32(keyScales, row * 4) * scale
                buffer.writef32(scores, j * 4, dot)
                if dot > maxScore then
                    maxScore = dot
                end
            end
            if stage == 1 then
                continue
            end
            local total = 0
            for j = 0, count do
                const e = math.exp(buffer.readf32(scores, j * 4) - maxScore)
                buffer.writef32(scores, j * 4, e)
                total += e
            end
            if stage == 2 then
                continue
            end
            const inv = 1 / total
            const vRow = rowBase + first
            for j = 0, count do
                buffer.writef32(
                    scores,
                    j * 4,
                    buffer.readf32(scores, j * 4) * inv * buffer.readf32(valueScales, (vRow + j) * 4)
                )
            end
            if stage == 3 then
                continue
            end
            const ob = (t * outStride + h * hd) * 4
            const vBase = vRow * hd
            for d = 0, hd8 - 1, 8 do
                local v0, v1, v2, v3, v4, v5, v6, v7 = 0, 0, 0, 0, 0, 0, 0, 0
                for j = 0, count do
                    const w = buffer.readf32(scores, j * 4)
                    const vb = vBase + j * hd + d
                    v0 += w * buffer.readi8(values, vb)
                    v1 += w * buffer.readi8(values, vb + 1)
                    v2 += w * buffer.readi8(values, vb + 2)
                    v3 += w * buffer.readi8(values, vb + 3)
                    v4 += w * buffer.readi8(values, vb + 4)
                    v5 += w * buffer.readi8(values, vb + 5)
                    v6 += w * buffer.readi8(values, vb + 6)
                    v7 += w * buffer.readi8(values, vb + 7)
                end
                const o = ob + d * 4
                buffer.writef32(out, o, v0)
                buffer.writef32(out, o + 4, v1)
                buffer.writef32(out, o + 8, v2)
                buffer.writef32(out, o + 12, v3)
                buffer.writef32(out, o + 16, v4)
                buffer.writef32(out, o + 20, v5)
                buffer.writef32(out, o + 24, v6)
                buffer.writef32(out, o + 28, v7)
            end
            for d = hd8, hd - 1 do
                local v = 0
                for j = 0, count do
                    v += buffer.readf32(scores, j * 4) * buffer.readi8(values, vBase + j * hd + d)
                end
                buffer.writef32(out, ob + d * 4, v)
            end
        end
    end
end

--- The three f32 stage functions for a head size, from modules the generator renders into `out/`.
const function stageModules(lanes, generator, prefix)
    fs.createDirectory("out/attention_profile", { makeParents = true })
    const fns = {}
    for _, stage in { "dots", "softmax" } do
        const path = `out/attention_profile/{prefix}{stage}.luau`
        fs.writeStringToFile(path, generator.renderModule(stage))
        fns[stage] = require(`../../out/attention_profile/{prefix}{stage}`).byLanes[lanes]
    end
    return fns
end

const rng = data.newRng(11)
local checked = false
for _, name in shapeNames do
    const sh = SHAPES[name]
    const hdPad = (sh.headDim + 2) // 3 * 3
    const maxPos = positions[#positions]
    const context = maxPos + 8
    const group = sh.heads // sh.kvHeads
    const q = buffer.create(sh.heads * hdPad * 4)
    for i = 0, sh.heads - 1 do
        for d = 0, sh.headDim - 1 do
            buffer.writef32(q, (i * hdPad + d) * 4, data.nextFloat(rng, 1) * spread)
        end
    end
    const keys, values, keyScales, valueScales = {}, {}, {}, {}
    for l = 1, sh.layers do
        if kv == "q8v" then
            const rowB = kvquant.rowBytes(sh.headDim, "v3")
            keys[l] = buffer.create(sh.kvHeads * context * rowB)
            values[l] = buffer.create(sh.kvHeads * context * rowB)
            keyScales[l] = buffer.create(sh.kvHeads * context * 4)
            valueScales[l] = buffer.create(sh.kvHeads * context * 4)
            for i = 0, sh.kvHeads * context - 1 do
                buffer.writef32(keyScales[l], i * 4, 0.01 + data.nextInt(rng, 100) * 1e-4)
                buffer.writef32(valueScales[l], i * 4, 0.01 + data.nextInt(rng, 100) * 1e-4)
                for lane = 0, rowB // 4 - 1 do
                    for _, b in { keys[l], values[l] } do
                        const c, bb, a =
                            1 + data.nextInt(rng, 255), 1 + data.nextInt(rng, 255), 1 + data.nextInt(rng, 255)
                        buffer.writef32(b, i * rowB + lane * 4, c + bb / 256 + a / 65536)
                    end
                end
            end
        elseif kv == "f32" then
            keys[l] = buffer.create(sh.kvHeads * context * hdPad * 4)
            values[l] = buffer.create(sh.kvHeads * context * hdPad * 4)
            for i = 0, sh.kvHeads * context - 1 do
                for d = 0, sh.headDim - 1 do
                    buffer.writef32(keys[l], (i * hdPad + d) * 4, data.nextFloat(rng, 1))
                    buffer.writef32(values[l], (i * hdPad + d) * 4, data.nextFloat(rng, 1))
                end
            end
        else
            keys[l] = buffer.create(sh.kvHeads * context * sh.headDim)
            values[l] = buffer.create(sh.kvHeads * context * sh.headDim)
            keyScales[l] = buffer.create(sh.kvHeads * context * 4)
            valueScales[l] = buffer.create(sh.kvHeads * context * 4)
            for i = 0, sh.kvHeads * context - 1 do
                buffer.writef32(keyScales[l], i * 4, 0.01 + data.nextInt(rng, 100) * 1e-4)
                buffer.writef32(valueScales[l], i * 4, 0.01 + data.nextInt(rng, 100) * 1e-4)
                for d = 0, sh.headDim - 1 do
                    buffer.writei8(keys[l], i * sh.headDim + d, data.nextInt(rng, 255) - 127)
                    buffer.writei8(values[l], i * sh.headDim + d, data.nextInt(rng, 255) - 127)
                end
            end
        end
    end
    const shape = {
        heads = sh.heads,
        kvHeads = sh.kvHeads,
        group = group,
        headBase = 0,
        kvBase = 0,
        headDim = sh.headDim,
        hdPad = hdPad,
        context = context,
        scale = 1 / math.sqrt(sh.headDim),
        window = nil,
    }
    const out = buffer.create(sh.heads * sh.headDim * 4)
    const scores = buffer.create(context * 4)
    const acc = table.create(hdPad, 0)
    local stages
    if kv == "q8v" then
        const lanes = hdPad // 3
        const fns = stageModules(lanes, genQ8, "q8_")
        const function run(fn)
            return function(l, p)
                shape.keyScales, shape.valueScales = keyScales[l], valueScales[l]
                fn(shape, q, 1, keys[l], values[l], p - 1, out, sh.heads * sh.headDim, scores, acc)
            end
        end
        stages = {
            { "scores", run(fns.dots) },
            { "softmax", run(fns.softmax) },
            { "values", run(unrolled.q8ByLanes[lanes]) },
        }
    elseif kv == "f32" then
        const lanes = hdPad // 3
        const fns = stageModules(lanes, gen, "")
        stages = {
            {
                "scores",
                function(l, p)
                    fns.dots(shape, q, 1, keys[l], values[l], p - 1, out, sh.heads * sh.headDim, scores, acc)
                end,
            },
            {
                "softmax",
                function(l, p)
                    fns.softmax(shape, q, 1, keys[l], values[l], p - 1, out, sh.heads * sh.headDim, scores, acc)
                end,
            },
            {
                "values",
                function(l, p)
                    unrolled.byLanes[lanes](
                        shape,
                        q,
                        1,
                        keys[l],
                        values[l],
                        p - 1,
                        out,
                        sh.heads * sh.headDim,
                        scores,
                        acc
                    )
                end,
            },
        }
    else
        shape.keyScales = nil
        stages = {}
        for i, label in { "scores", "softmax", "weight scales", "values" } do
            table.insert(stages, {
                label,
                function(l, p)
                    shape.keyScales, shape.valueScales = keyScales[l], valueScales[l]
                    attendQ8(i, shape, q, 1, keys[l], values[l], p - 1, out, sh.heads * sh.headDim, scores)
                end,
            })
        end
    end
    if kv == "q8" and not checked then
        checked = true
        const ok, real = pcall(function()
            return require("../../src/ops/attention_q8")
        end)
        if ok then
            const a, b = buffer.create(sh.heads * sh.headDim * 4), buffer.create(sh.heads * sh.headDim * 4)
            shape.keyScales, shape.valueScales = keyScales[1], valueScales[1]
            real.attend(shape, q, 1, keys[1], values[1], 127, a, sh.heads * sh.headDim, scores, acc)
            attendQ8(4, shape, q, 1, keys[1], values[1], 127, b, sh.heads * sh.headDim, scores)
            print(
                "replica equals attention_q8:",
                buffer.readstring(a, 0, buffer.len(a)) == buffer.readstring(b, 0, buffer.len(b))
            )
        else
            print("attention_q8 is not in this tree, the replica is unchecked")
        end
    end
    for _, p in positions do
        -- Seconds per decode token over every layer for each stage. The stages alternate within a repeat, so a slow
        -- stretch hits all of them, and the fastest repeat counts, because CPU contention only adds time.
        const best = {}
        for i in stages do
            best[i] = math.huge
        end
        for r = 1, repeats do
            for i, st in stages do
                const t0 = os.clock()
                local passes = 0
                repeat
                    for l = 1, sh.layers do
                        st[2](l, p)
                    end
                    passes += 1
                until os.clock() - t0 >= 0.1
                best[i] = math.min(best[i], (os.clock() - t0) / passes)
            end
        end
        const macs = sh.layers * sh.heads * p * sh.headDim
        const exps = sh.layers * sh.heads * p
        const parts = {}
        for i, st in stages do
            parts[i] = best[i] - (best[i - 1] or 0)
        end
        const line = { string.format("APART kv=%s shape=%s pos=%d total_us=%.0f", kv, name, p, best[#best] * 1e6) }
        for i, st in stages do
            table.insert(line, string.format("%s_us=%.0f", (string.gsub(st[1], " ", "_")), parts[i] * 1e6))
        end
        table.insert(line, string.format("scores_MMACs=%.0f", macs / parts[1] / 1e6))
        table.insert(line, string.format("softmax_Mexps=%.0f", exps / parts[2] / 1e6))
        table.insert(line, string.format("values_MMACs=%.0f", macs / parts[#parts] / 1e6))
        table.insert(line, string.format("ns_per_head_pos=%.1f", best[#best] / (sh.layers * sh.heads * p) * 1e9))
        print(table.concat(line, " "))
    end
end
