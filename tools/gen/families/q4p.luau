-- The packed Q4 kernel: 8 rows of one column are one u32 of nibbles, and a mask picks two of them 16 bits apart, so one
-- double multiply by the activation adds two products. Layout and exactness are argued in docs/kernels.md. A tile is 8
-- rows by 1 token, or 4 rows by 2 tokens, and a group of 32 columns per pass. The activations are the f64 form the packed
-- Q8 kernel reads, with 128 times their group sum and the group scale.

const emit = require("../emit")
const tile = require("../tile")

const GROUP = 32
--- The weight of a field pair of mask m is 16^m (low + 2^16 high), and the offset 8 of the nibbles is cancelled by
--- starting the sum at 8 * sum(x) * (2^16 + 1) * 16^m negated. The stored group sum is 128 sum(x), so the factor is
--- -(2^16 + 1) * 16^m / 16.
const PAIR = 65537

--- The masks of the four pairs, pair m holding nibbles m and m + 4.
const MASKS = { "0x000F000F", "0x00F000F0", "0x0F000F00", "0xF000F000" }

--- Columns one loop pass covers, which the stored row length is a multiple of.
const function kstep(_: tile.Tile): number
    return GROUP
end

--- A double literal of a power of two times an integer, written exactly.
const function literal(v: number): string
    return string.format("%.17g", v)
end

--- The text of the kernel function for `t`.
const function render(t: tile.Tile): string
    const R, S = t.R, t.S
    assert((R == 8 and S == 1) or (R == 4 and S == 2), "the packed Q4 tile is 8 rows by 1 token or 4 rows by 2 tokens")
    assert(t.U == GROUP, "the unroll is a whole group")
    assert(t.order == "c2", "the packed Q4 kernel loads two columns at a time")
    -- Pairs this tile computes: all four, or the two of the band it is on, known at run time.
    const pairs = R // 2
    const e = emit.new()
    emit.open(e, `const function {t.name}({tile.SIGNATURE}): ()`)
    emit.line(e, "const wsb = ws :: buffer")
    emit.line(e, "const xsb = xs :: buffer")
    emit.line(e, `const ng = K // {GROUP}`)
    emit.line(e, "const ng4 = ng * 4")
    emit.line(e, "const ng8 = ng * 8")
    emit.line(e, "const half = K // 2")
    emit.open(e, `for i = row0, row0 + rows - 1, {R} do`)
    if R == 8 then
        emit.line(e, "const wb = wOff + i * half")
        emit.line(e, "const sb = wsOff + i * ng4")
    else
        -- Band 0 is rows 0 to 3 of the tile, pairs 0 and 1, band 1 is rows 4 to 7, pairs 2 and 3. The mask of a pair
        -- moves up 8 bits for band 1, which scales the sum by 256, undone by `inv` in the epilogue.
        emit.line(e, "const band = i % 8")
        emit.line(e, "const wb = wOff + (i - band) * half")
        emit.line(e, "const sb = wsOff + (i - band) * ng4 + band * 4")
        emit.line(e, "const m0 = if band == 0 then 0x000F000F else 0x0F000F00")
        emit.line(e, "const m1 = if band == 0 then 0x00F000F0 else 0xF000F000")
        emit.line(e, `const init0 = if band == 0 then {literal(-PAIR / 16)} else {literal(-PAIR * 16)}`)
        emit.line(e, "const init1 = init0 * 16")
        emit.line(e, "const inv0 = if band == 0 then 1 else 0.00390625")
        emit.line(e, "const inv1 = inv0 * 0.0625")
        emit.line(e, "const invh0 = inv0 * 1.52587890625e-05")
        emit.line(e, "const invh1 = inv1 * 1.52587890625e-05")
    end
    emit.open(e, `for t = 0, T - 1, {S} do`)
    for s = 0, S - 1 do
        emit.line(e, `const xo{s} = {emit.plusFactor("t", s)} * K * 8`)
        emit.line(e, `const xg{s} = {emit.plusFactor("t", s)} * ng8`)
        for r = 0, R - 1 do
            emit.line(e, `local t{r}_{s} = 0`)
        end
    end
    emit.open(e, "for g = 0, ng - 1 do")
    emit.line(e, "const sr = sb + g * 32")
    for s = 0, S - 1 do
        emit.line(e, `const xm{s} = buffer.readf32(xsb, xg{s} + g * 8)`)
        emit.line(e, `const xd{s} = buffer.readf32(xsb, xg{s} + g * 8 + 4)`)
        for p = 0, pairs - 1 do
            if R == 8 then
                emit.line(e, `local q{p}_{s} = xm{s} * {literal(-PAIR * 16 ^ p / 16)}`)
            else
                emit.line(e, `local q{p}_{s} = xm{s} * init{p}`)
            end
        end
        emit.line(e, `const xb{s} = xo{s} + g * 256`)
    end
    emit.line(e, "const pk = wb + g * 128")
    for u0 = 0, GROUP - 1, 2 do
        for s = 0, S - 1 do
            for u = u0, u0 + 1 do
                emit.line(e, `const xv{u}_{s} = buffer.readf64(x, {emit.plus(`xb{s}`, 8 * u)})`)
            end
        end
        for u = u0, u0 + 1 do
            emit.line(e, `const wd{u} = buffer.readu32(w, {emit.plus("pk", 4 * u)})`)
            for p = 0, pairs - 1 do
                const mask = if R == 8 then MASKS[p + 1] else `m{p}`
                emit.line(e, `const d{p}_{u} = bit32.band(wd{u}, {mask})`)
            end
        end
        for s = 0, S - 1 do
            for u = u0, u0 + 1 do
                for p = 0, pairs - 1 do
                    emit.line(e, `q{p}_{s} += d{p}_{u} * xv{u}_{s}`)
                end
            end
        end
    end
    for s = 0, S - 1 do
        for p = 0, pairs - 1 do
            -- Pair p holds row 2p in its low field and row 2p + 1 in its high one.
            const invh = if R == 8 then literal(2 ^ (-16 - 4 * p)) else `invh{p}`
            const inv = if R == 8 then literal(2 ^ (-4 * p)) else `inv{p}`
            emit.line(e, `const hi{p}_{s} = math.floor(q{p}_{s} * {invh} + 0.5)`)
            emit.line(
                e,
                `t{2 * p}_{s} += buffer.readf32(wsb, {emit.plus("sr", 4 * 2 * p)}) * xd{s} * (q{p}_{s} * {inv} - hi{p}_{s} * 65536)`
            )
            emit.line(
                e,
                `t{2 * p + 1}_{s} += buffer.readf32(wsb, {emit.plus("sr", 4 * (2 * p + 1))}) * xd{s} * hi{p}_{s}`
            )
        end
    end
    emit.close(e)
    for s = 0, S - 1 do
        for r = 0, R - 1 do
            emit.line(
                e,
                `buffer.writef32(y, yBase + ({emit.plusFactor("t", s)} * yStride + {emit.plus("i", r)}) * 4, t{r}_{s})`
            )
        end
    end
    emit.close(e)
    emit.close(e)
    emit.close(e)
    return emit.text(e)
end

return {
    kstep = kstep,
    render = render,
}
