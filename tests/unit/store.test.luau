-- A matrix is cut into parts of whole tiles that each fit the buffer limit, the parts cover every row exactly once,
-- and a row is found in the part that holds it.
const formats = require("@src/tensor/formats")
const harness = require("@tests/harness")
const store = require("@src/tensor/store")
const tensor = require("@src/tensor")

return function(h: harness.Harness)
    h.case("a matrix that fits is one part", function()
        const plan = store.plan(64, 100, 8, 1048576)
        h.equal(#plan, 1)
        h.equal(plan[1].rowStart, 0)
        h.equal(plan[1].rowCount, 64)
    end)

    h.case("parts are whole tiles, cover every row once, and fit the limit", function()
        for _, limit in { 1000, 4096, 8192, 100000 } do
            const plan = store.plan(200, 100, 8, limit)
            local next = 0
            for _, p in plan do
                h.equal(p.rowStart, next, "a gap or overlap between parts")
                h.equal(p.rowCount % 8, 0, "a part is not whole tiles")
                h.truthy(p.rowCount * 100 <= limit, "a part is over the limit")
                next += p.rowCount
            end
            h.equal(next, 200)
        end
    end)

    h.case("a part holds as many whole tiles as fit", function()
        -- 1000 byte limit, 100 byte rows: 10 rows fit, so one tile of 8.
        const plan = store.plan(24, 100, 8, 1000)
        h.equal(#plan, 3)
        h.equal(plan[1].rowCount, 8)
    end)

    h.case("one tile that cannot fit is an error", function()
        h.fails(function()
            store.plan(8, 1000, 8, 4000)
        end)
    end)

    h.case("rows must be a multiple of the tile", function()
        h.fails(function()
            store.plan(10, 100, 8, 100000)
        end)
    end)

    h.case("partIndex finds the part of every row", function()
        const plan = store.plan(48, 100, 8, 1700)
        const parts = store.create(plan, "f32", 25)
        h.truthy(#parts > 1)
        for row = 0, 47 do
            const part = parts[store.partIndex(parts, row)]
            h.truthy(row >= part.rowStart and row < part.rowStart + part.rowCount, "row in the wrong part")
        end
        h.fails(function()
            store.partIndex(parts, 48)
        end)
    end)

    h.case("a Q8 tensor splits at a small limit and no part spans two buffers", function()
        const t = tensor.new({ format = "q8", rows = 100, cols = 70, rowTile = 8, kstep = 32, maxPartBytes = 96 * 16 })
        h.equal(t.kpad, 96)
        h.equal(t.rowsPadded, 104)
        h.truthy(#t.parts > 1)
        local rows = 0
        for _, part in t.parts do
            h.equal(buffer.len(part.buf), part.rowCount * 96)
            h.equal(buffer.len(part.scales :: buffer), part.rowCount * 3 * 4)
            h.truthy(buffer.len(part.buf) <= 96 * 16, "a part buffer is over the limit")
            rows += part.rowCount
        end
        h.equal(rows, 104)
    end)

    h.case("the default limit is one part for a model sized matrix", function()
        const t = tensor.new({ format = "f32", rows = 1536, cols = 576, rowTile = 8, kstep = 24 })
        h.equal(#t.parts, 1)
        h.equal(buffer.len(t.parts[1].buf), 1536 * 576 * 4)
        h.truthy(buffer.len(t.parts[1].buf) <= formats.MAX_BUFFER_BYTES)
    end)

    h.case("a matrix past the limit splits into parts that each fit", function()
        -- 1048576 rows of 3072 f32 is a 12 GiB matrix, which no single buffer holds.
        const plan = store.plan(1048576, 3072 * 4, 8, formats.MAX_BUFFER_BYTES)
        h.truthy(#plan >= 3)
        for _, p in plan do
            h.truthy(p.rowCount * 3072 * 4 <= formats.MAX_BUFFER_BYTES)
            h.equal(p.rowCount % 8, 0)
        end
    end)
end
