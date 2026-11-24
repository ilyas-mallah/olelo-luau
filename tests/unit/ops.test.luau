const embed = require("@src/ops/embed")
const harness = require("@tests/harness")
const mlp = require("@src/ops/mlp")
const norm = require("@src/ops/norm")
const rope = require("@src/ops/rope")
const tensor = require("@src/tensor")

const function floats(values: { number }): buffer
    const b = buffer.create(#values * 4)
    for i, v in values do
        buffer.writef32(b, (i - 1) * 4, v)
    end
    return b
end

return function(h: harness.Harness)
    h.case("rms norm", function()
        const x = floats({ 3, 4, 0, 0, 1, 1, 1, 1 })
        const w = floats({ 1, 2, 0.5, 1 })
        const out = buffer.create(32)
        norm.rms(x, 4, w, out, 4, 4, 2, 0, false)
        -- Row 0 has mean square 6.25, so rms 2.5.
        h.near(buffer.readf32(out, 0), 3 / 2.5, 1e-6)
        h.near(buffer.readf32(out, 4), 2 * 4 / 2.5, 1e-6)
        h.near(buffer.readf32(out, 8), 0, 1e-9)
        h.near(buffer.readf32(out, 16), 1, 1e-6)
        h.near(buffer.readf32(out, 20), 2, 1e-6)
        h.near(buffer.readf32(out, 24), 0.5, 1e-6)
    end)

    h.case("rms norm with the plus one weight", function()
        const x = floats({ 2, 2 })
        const out = buffer.create(8)
        norm.rms(x, 2, floats({ 0, 1 }), out, 2, 2, 1, 0, true)
        h.near(buffer.readf32(out, 0), 1, 1e-6)
        h.near(buffer.readf32(out, 4), 2, 1e-6)
    end)

    h.case("residual add keeps the padding", function()
        const hh = floats({ 1, 2, 3, 9 })
        norm.add(hh, 4, floats({ 10, 20, 30, 0 }), 4, 3, 1)
        h.equal(buffer.readf32(hh, 0), 11)
        h.equal(buffer.readf32(hh, 8), 33)
        h.equal(buffer.readf32(hh, 12), 9)
    end)

    h.case("rope rotates pairs, keeps the length and is the identity at position 0", function()
        const r = rope.new(4, 10000, 8)
        const src = floats({ 1, 2, 3, 4 })
        const dst = buffer.create(16)
        rope.apply(r, src, 4, 1, 4, 1, 0, dst, 0, 16, 16)
        for i = 0, 3 do
            h.near(buffer.readf32(dst, i * 4), i + 1, 1e-6)
        end
        rope.apply(r, src, 4, 1, 4, 1, 3, dst, 0, 16, 16)
        local before, after = 0, 0
        for i = 0, 3 do
            before += (i + 1) ^ 2
            after += buffer.readf32(dst, i * 4) ^ 2
        end
        h.near(after, before, 1e-4)
        -- Element 0 pairs with element 2 at angle 3 * 10000^0 = 3.
        h.near(buffer.readf32(dst, 0), 1 * math.cos(3) - 3 * math.sin(3), 1e-5)
        h.near(buffer.readf32(dst, 8), 3 * math.cos(3) + 1 * math.sin(3), 1e-5)
    end)

    h.case("rope writes the destination layout and leaves the padding", function()
        const r = rope.new(4, 10000, 8)
        const src = floats({ 1, 2, 3, 4, 5, 6, 7, 8 })
        const dst = buffer.create(2 * 8 * 4)
        buffer.fill(dst, 0, 0)
        -- Two heads of 4 values, stored 6 apart.
        rope.apply(r, src, 8, 2, 4, 1, 0, dst, 0, 48, 24)
        h.equal(buffer.readf32(dst, 24), 5)
        h.equal(buffer.readf32(dst, 16), 0)
        h.equal(buffer.readf32(dst, 20), 0)
    end)

    h.case("silu and gelu gates", function()
        const gate = floats({ 1, -2, 0 })
        const up = floats({ 2, 3, 4 })
        const out = buffer.create(12)
        mlp.gated("silu", gate, 3, up, 3, out, 3, 3, 1)
        h.near(buffer.readf32(out, 0), 1 / (1 + math.exp(-1)) * 2, 1e-6)
        h.near(buffer.readf32(out, 4), -2 / (1 + math.exp(2)) * 3, 1e-6)
        h.equal(buffer.readf32(out, 8), 0)
        mlp.gated("gelu_tanh", floats({ 1, 100 }), 2, floats({ 1, 1 }), 2, out, 2, 2, 1)
        h.near(buffer.readf32(out, 0), 0.8411919906, 1e-5)
        h.near(buffer.readf32(out, 4), 100, 1e-4)
    end)

    h.case("Q8 embedding rows dequantize across parts", function()
        const t = tensor.new({ format = "q8", rows = 16, cols = 40, rowTile = 8, kstep = 32, maxPartBytes = 8 * 64 })
        h.equal(#t.parts, 2)
        for row = 0, 15 do
            for k = 0, 39 do
                tensor.setQuant(t, row, k, (row * 5 + k) % 256 - 128)
            end
            for g = 0, 1 do
                tensor.setScale(t, row, g, (row + 1) / 8 + g)
            end
        end
        const dst = buffer.create(40 * 4 + 8)
        embed.row(t, 11, dst, 8)
        for k = 0, 39 do
            const want = ((11 * 5 + k) % 256 - 128) * ((11 + 1) / 8 + k // 32)
            h.equal(buffer.readf32(dst, 8 + k * 4), want, `k {k}`)
        end
    end)

    h.case("Q8 embedding rows dequantize in the vector layout", function()
        const t = tensor.new({
            format = "q8",
            layout = "v12",
            rows = 30,
            cols = 40,
            rowTile = 12,
            kstep = 32,
            maxPartBytes = 12 * 64 * 2,
        })
        for row = 0, 29 do
            for k = 0, 39 do
                tensor.setQuant(t, row, k, (row * 5 + k) % 256 - 128)
            end
            for g = 0, 1 do
                tensor.setScale(t, row, g, (row + 1) / 8 + g)
            end
        end
        const dst = buffer.create(40 * 4 + 8)
        for _, row in { 0, 13, 29 } do
            embed.row(t, row, dst, 8)
            for k = 0, 39 do
                const want = ((row * 5 + k) % 256 - 128) * ((row + 1) / 8 + k // 32)
                h.equal(buffer.readf32(dst, 8 + k * 4), want, `row {row} k {k}`)
            end
        end
    end)

    h.case("embedding rows copy across parts", function()
        const t = tensor.new({ format = "f32", rows = 16, cols = 5, rowTile = 8, kstep = 6, maxPartBytes = 8 * 6 * 4 })
        h.equal(#t.parts, 2)
        for row = 0, 15 do
            for k = 0, 4 do
                tensor.setFloat(t, row, k, row * 10 + k)
            end
        end
        const dst = buffer.create(40)
        embed.row(t, 11, dst, 8)
        h.equal(buffer.readf32(dst, 8), 110)
        h.equal(buffer.readf32(dst, 8 + 16), 114)
        h.fails(function()
            embed.row(t, 16, dst, 0)
        end)
    end)
end
