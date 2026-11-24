-- The address helpers place every weight of every format and layout at a distinct byte of the right size, the scale
-- offsets tile the scale region, and the BF16 and f32 rounding helpers round as the formats do.
const formats = require("@src/tensor/formats")
const harness = require("@tests/harness")
const tensorTypes = require("@src/tensor/types")

--- Checks that (row, k) over `rows` x `kpad` maps one to one onto `bytes` bytes, counting nibbles for Q4.
const function bijective(
    name: tensorTypes.FormatName,
    rows: number,
    kpad: number,
    bytes: number,
    perByte: number,
    layout: tensorTypes.LayoutName?
): string?
    const seen: { [number]: boolean } = {}
    for row = 0, rows - 1 do
        for k = 0, kpad - 1 do
            local offset, nibble = formats.weightOffset(name, row, k, kpad, layout)
            const key = offset * perByte + nibble
            if offset < 0 or offset >= bytes or seen[key] then
                return `row {row} k {k} maps to offset {offset} nibble {nibble}`
            end
            seen[key] = true
        end
    end
    return nil
end

return function(h: harness.Harness)
    h.case("the buffer limit is 2^30 bytes, the Luau VM's own", function()
        h.equal(formats.MAX_BUFFER_BYTES, 2 ^ 30)
    end)

    h.case("row sizes", function()
        h.equal(formats.rowBytes("f32", 24), 96)
        h.equal(formats.rowBytes("bf16", 36), 72)
        h.equal(formats.rowBytes("q8", 64), 64)
        h.equal(formats.rowBytes("q4", 64), 32)
        h.equal(formats.scaleRowBytes("f32", 24), 0)
        h.equal(formats.scaleRowBytes("q8", 64), 8)
        h.equal(formats.scaleRowBytes("q4", 96), 12)
    end)

    h.case("f32 and BF16 rows are row major", function()
        h.equal(formats.weightOffset("f32", 3, 5, 24), (3 * 24 + 5) * 4)
        h.equal(formats.weightOffset("bf16", 3, 5, 36), (3 * 36 + 5) * 2)
    end)

    h.case("Q8 packs a tile of 8 rows by column", function()
        -- Rows 0 to 7 are 8 bytes of one column, then the next column, then the next tile of 8 rows.
        h.equal(formats.weightOffset("q8", 0, 0, 64), 0)
        h.equal(formats.weightOffset("q8", 1, 0, 64), 1)
        h.equal(formats.weightOffset("q8", 7, 0, 64), 7)
        h.equal(formats.weightOffset("q8", 0, 1, 64), 8)
        h.equal(formats.weightOffset("q8", 8, 0, 64), 8 * 64)
        h.equal(formats.weightOffset("q8", 9, 3, 64), 8 * 64 + 3 * 8 + 1)
    end)

    h.case("Q8 scales of a tile sit together per group", function()
        -- 2 groups: a tile has 8 scales for group 0, then 8 for group 1, then the next tile.
        h.equal(formats.scaleOffset(0, 0, 64, "p8"), 0)
        h.equal(formats.scaleOffset(3, 0, 64, "p8"), 3 * 4)
        h.equal(formats.scaleOffset(0, 1, 64, "p8"), 8 * 4)
        h.equal(formats.scaleOffset(8, 0, 64, "p8"), 16 * 4)
        h.equal(formats.scaleOffset(5, 1, 64, "p8"), (8 + 5) * 4)
    end)

    h.case("the interleaved Q8 layout of 4 rows by element stays available", function()
        h.equal(formats.weightOffset("q8", 1, 0, 64, "il4e"), 1)
        h.equal(formats.weightOffset("q8", 0, 1, 64, "il4e"), 4)
        h.equal(formats.weightOffset("q8", 4, 0, 64, "il4e"), 4 * 64)
        h.equal(formats.scaleOffset(3, 1, 64, "il4e"), (3 * 2 + 1) * 4)
    end)

    h.case("the row major Q4 layout puts element j low and element j + 16 high in one byte", function()
        local offset, nibble = formats.weightOffset("q4", 0, 3, 64, "rowmajor")
        h.equal(offset, 3)
        h.equal(nibble, 0)
        offset, nibble = formats.weightOffset("q4", 0, 19, 64, "rowmajor")
        h.equal(offset, 3)
        h.equal(nibble, 1)
        offset, nibble = formats.weightOffset("q4", 0, 32, 64, "rowmajor")
        h.equal(offset, 16)
        h.equal(nibble, 0)
        offset, nibble = formats.weightOffset("q4", 2, 0, 64, "rowmajor")
        h.equal(offset, 2 * 32)
        h.equal(nibble, 0)
    end)

    h.case(
        "the packed Q4 layout has a word per column with the even row of pair m in nibble m, the odd one in m + 4",
        function()
            -- Row 0, column 3: word 3 of group 0, nibble 0. Row 1: nibble 4, the low nibble of byte 2.
            local offset, nibble = formats.weightOffset("q4", 0, 3, 64)
            h.equal(offset, 12)
            h.equal(nibble, 0)
            offset, nibble = formats.weightOffset("q4", 1, 3, 64)
            h.equal(offset, 12 + 2)
            h.equal(nibble, 0)
            offset, nibble = formats.weightOffset("q4", 2, 3, 64)
            h.equal(offset, 12)
            h.equal(nibble, 1)
            offset, nibble = formats.weightOffset("q4", 7, 3, 64)
            h.equal(offset, 12 + 3)
            h.equal(nibble, 1)
            -- The second group starts 128 bytes on, the second tile of rows 4 * kpad bytes on.
            offset = formats.weightOffset("q4", 0, 32, 64)
            h.equal(offset, 128)
            offset = formats.weightOffset("q4", 8, 0, 64)
            h.equal(offset, 4 * 64)
        end
    )

    h.case(
        "the vector Q8 layout keeps 3 columns of a row in one f32 lane, 11 steps a group, 4 row vectors a step",
        function()
            -- Columns j, j + 11 and j + 22 of a group are the fields of the lane of step j.
            local offset, field = formats.weightOffset("q8", 0, 0, 64, "v12")
            h.equal(offset, 0)
            h.equal(field, 0)
            offset, field = formats.weightOffset("q8", 0, 11, 64, "v12")
            h.equal(offset, 0)
            h.equal(field, 1)
            offset, field = formats.weightOffset("q8", 0, 22, 64, "v12")
            h.equal(offset, 0)
            h.equal(field, 2)
            -- The next step is 48 bytes on, the next row of a vector 4, the next row vector 12.
            h.equal(formats.weightOffset("q8", 0, 1, 64, "v12"), 48)
            h.equal(formats.weightOffset("q8", 1, 0, 64, "v12"), 4)
            h.equal(formats.weightOffset("q8", 3, 0, 64, "v12"), 12)
            h.equal(formats.weightOffset("q8", 11, 0, 64, "v12"), 3 * 12 + 2 * 4)
            -- A group of a tile is 528 bytes, a tile of 2 groups 1056.
            h.equal(formats.weightOffset("q8", 0, 32, 64, "v12"), 528)
            h.equal(formats.weightOffset("q8", 12, 0, 64, "v12"), 1056)
            h.equal(formats.rowBytes("q8", 64, "v12"), 88)
            h.equal(formats.scaleOffset(5, 0, 64, "v12"), 5 * 4)
            h.equal(formats.scaleOffset(5, 1, 64, "v12"), (12 + 5) * 4)
            h.equal(formats.scaleOffset(12, 0, 64, "v12"), 12 * 2 * 4)
        end
    )

    h.case("a lane holds three bytes exactly and reads them back", function()
        for _, bytes in { { 0, 0, 0 }, { 255, 255, 255 }, { 1, 0, 254 }, { 128, 128, 128 }, { 37, 201, 90 } } do
            const a, b, c = formats.bytesOf(formats.laneOf(bytes[1], bytes[2], bytes[3]))
            h.equal(a, bytes[1])
            h.equal(b, bytes[2])
            h.equal(c, bytes[3])
        end
    end)

    h.case("every layout is a one to one map onto the row bytes", function()
        h.equal(bijective("f32", 8, 24, 8 * 96, 1), nil)
        h.equal(bijective("bf16", 8, 36, 8 * 72, 1), nil)
        h.equal(bijective("q8", 8, 64, 8 * 64, 1), nil)
        h.equal(bijective("q4", 8, 64, 8 * 32, 2, "rowmajor"), nil)
        h.equal(bijective("q4", 16, 64, 16 * 32, 2), nil)
        -- The vector layout has 3 weights to a lane, and the slot of column 32 of a group stays unused.
        h.equal(bijective("q8", 24, 64, 24 * 88, 3, "v12"), nil)
    end)

    h.case("scale offsets tile the scale region", function()
        const seen: { [number]: boolean } = {}
        for row = 0, 7 do
            for group = 0, 2 do
                const offset = formats.scaleOffset(row, group, 96)
                h.truthy(offset >= 0 and offset < 8 * 3 * 4 and not seen[offset], "scale offset clash")
                seen[offset] = true
            end
        end
    end)

    h.case("half precision rounds to nearest even, with subnormals", function()
        h.equal(formats.roundF16(0), 0)
        h.equal(formats.roundF16(0.5), 0.5)
        h.equal(formats.roundF16(65504), 65504)
        h.equal(formats.roundF16(0.1), 0.0999755859375)
        h.equal(formats.roundF16(1 / 3), 0.333251953125)
        h.equal(formats.roundF16(-0.1), -0.0999755859375)
        -- Halfway cases go to the even neighbour.
        h.equal(formats.roundF16(1 + 2 ^ -11), 1)
        h.equal(formats.roundF16(1 + 3 * 2 ^ -11), 1 + 2 ^ -9)
        -- Below 2^-14 the spacing is 2^-24.
        h.equal(formats.roundF16(2 ^ -24 * 16.6), 2 ^ -24 * 17)
        h.equal(formats.roundF16(2 ^ -26), 0)
    end)

    h.case("BF16 rounds to nearest even and converts back exactly", function()
        h.equal(formats.bf16FromF32(1), 0x3F80)
        h.equal(formats.bf16ToF32(0x3F80), 1)
        h.equal(formats.bf16FromF32(-2.5), 0xC020)
        -- 1 + 2^-8 is halfway between two BF16 values and rounds to the even one, which is 1.
        h.equal(formats.bf16FromF32(1 + 2 ^ -8), 0x3F80)
        h.equal(formats.bf16FromF32(1 + 2 ^ -8 + 2 ^ -9), 0x3F81)
        for _, v in { 0, 0.15625, 3.140625, -1e-3, 65536 } do
            const bits = formats.bf16FromF32(v)
            h.equal(formats.bf16FromF32(formats.bf16ToF32(bits)), bits)
        end
    end)

    h.case("f32 rounding is the nearest f32", function()
        h.equal(formats.roundF32(0.1), 0.10000000149011612)
        h.equal(formats.roundF32(1 + 2 ^ -24), 1)
    end)
end
