-- A tensor pads its rows to a tile and its columns to the kernel step with zeros, reads back what it stores in every
-- format across part boundaries, and keeps Q4 padding at the weight zero.
const harness = require("@tests/harness")
const tensor = require("@src/tensor")
const tensorTypes = require("@src/tensor/types")

return function(h: harness.Harness)
    h.case("padding rounds rows to the tile and columns to the step", function()
        const t = tensor.new({ format = "f32", rows = 37, cols = 50, rowTile = 8, kstep = 24 })
        h.equal(t.rowsPadded, 40)
        h.equal(t.kpad, 72)
        h.equal(tensor.getFloat(t, 39, 71), 0)
    end)

    h.case("f32 and BF16 weights read back across parts", function()
        for _, format in { "f32", "bf16" } do
            const t = tensor.new({
                format = format :: tensorTypes.FormatName,
                rows = 40,
                cols = 30,
                rowTile = 4,
                kstep = 6,
                maxPartBytes = 4 * 30 * 4,
            })
            h.truthy(#t.parts > 1, "expected several parts")
            for row = 0, 39 do
                for k = 0, 29 do
                    tensor.setFloat(t, row, k, (row * 7 + k) % 200)
                end
            end
            for row = 0, 39 do
                for k = 0, 29 do
                    h.equal(tensor.getFloat(t, row, k), (row * 7 + k) % 200, `{format} row {row} k {k}`)
                end
            end
        end
    end)

    h.case("Q8 weights and scales read back across parts", function()
        const t = tensor.new({ format = "q8", rows = 24, cols = 70, rowTile = 8, kstep = 32, maxPartBytes = 8 * 96 })
        h.truthy(#t.parts > 1)
        for row = 0, 23 do
            for k = 0, 69 do
                tensor.setQuant(t, row, k, (row * 7 + k * 3) % 256 - 128)
            end
            for g = 0, 2 do
                tensor.setScale(t, row, g, row + g / 4)
            end
        end
        for row = 0, 23 do
            for k = 0, 69 do
                h.equal(tensor.getQuant(t, row, k), (row * 7 + k * 3) % 256 - 128, `row {row} k {k}`)
            end
            for g = 0, 2 do
                h.equal(tensor.getScale(t, row, g), row + g / 4)
            end
        end
    end)

    h.case("Q8 padding is the weight zero and extremes survive in both layouts", function()
        for _, layout in { "p8", "il4e" } do
            const t =
                tensor.new({ format = "q8", rows = 5, cols = 40, rowTile = 8, kstep = 32, layout = layout :: any })
            h.equal(tensor.getQuant(t, 7, 63), 0, layout)
            tensor.setQuant(t, 2, 3, -128)
            tensor.setQuant(t, 2, 4, 127)
            h.equal(tensor.getQuant(t, 2, 3), -128, layout)
            h.equal(tensor.getQuant(t, 2, 4), 127, layout)
            h.equal(tensor.getQuant(t, 3, 3), 0, layout)
        end
    end)

    h.case("vector Q8 weights and scales read back across parts, in tiles of 12 rows", function()
        const t = tensor.new({
            format = "q8",
            rows = 30,
            cols = 70,
            rowTile = 12,
            kstep = 32,
            layout = "v12",
            maxPartBytes = 12 * 132,
        })
        h.equal(#t.parts, 3)
        h.equal(t.rowsPadded, 36)
        for row = 0, 29 do
            for k = 0, 69 do
                tensor.setQuant(t, row, k, (row * 7 + k * 3) % 256 - 128)
            end
            for g = 0, 2 do
                tensor.setScale(t, row, g, row + g / 4)
            end
        end
        for row = 0, 29 do
            for k = 0, 69 do
                h.equal(tensor.getQuant(t, row, k), (row * 7 + k * 3) % 256 - 128, `row {row} k {k}`)
            end
            for g = 0, 2 do
                h.equal(tensor.getScale(t, row, g), row + g / 4)
            end
        end
        -- The padding rows and columns are the weight zero, and the weights next to a set one stay zero.
        h.equal(tensor.getQuant(t, 35, 3), 0)
        h.equal(tensor.getQuant(t, 0, 95), 0)
    end)

    h.case("Q4 nibbles are independent and padding is the weight zero", function()
        const t = tensor.new({ format = "q4", rows = 8, cols = 40, rowTile = 8, kstep = 32 })
        h.equal(tensor.getQuant(t, 7, 63), 8)
        for row = 0, 7 do
            for k = 0, 39 do
                tensor.setQuant(t, row, k, (row + k) % 16)
            end
        end
        for row = 0, 7 do
            for k = 0, 39 do
                h.equal(tensor.getQuant(t, row, k), (row + k) % 16, `row {row} k {k}`)
            end
        end
    end)

    h.case("the vector layout needs rows in twelves", function()
        h.fails(function()
            tensor.new({ format = "q8", rows = 8, cols = 32, rowTile = 8, kstep = 32, layout = "v12" })
        end)
    end)

    h.case("the interleaved layout needs rows in fours and quantized steps whole groups", function()
        h.fails(function()
            tensor.new({ format = "q8", rows = 8, cols = 32, rowTile = 6, kstep = 32 })
        end)
        h.fails(function()
            tensor.new({ format = "q8", rows = 8, cols = 32, rowTile = 8, kstep = 8 })
        end)
    end)
end
