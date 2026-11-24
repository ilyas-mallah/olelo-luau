-- The packed Q8 kernel: offset binary int8 weights, two rows summed in one double per field pair, f64 activations
-- with a scale per group of 32 like the weights.
-- Layout and exactness are argued in docs/kernels.md. A tile is 8 rows by 1 token, or 4 rows by 2 tokens, and a group
-- of 32 columns per pass. The 4 row tile reads one word of each tile column, the word of its band, and shares the
-- field extraction between its two tokens.

const emit = require("../emit")
const tile = require("../tile")

const GROUP = 32
--- 2^-24, and the constant that cancels the offset: a field pair weighs 2^24 + 1.
const INV_FIELD = "5.9604644775390625e-08"
const FIELD = "16777216"
const PAIR = "-16777217"

--- Columns one loop pass covers, which the stored row length is a multiple of.
const function kstep(_: tile.Tile): number
    return GROUP
end

--- The text of the kernel function for `t`. `U` is the columns between loads.
const function render(t: tile.Tile): string
    const R, S = t.R, t.S
    assert((R == 8 and S == 1) or (R == 4 and S == 2), "the packed Q8 tile is 8 rows by 1 token or 4 rows by 2 tokens")
    assert(t.U == GROUP, "the unroll is a whole group")
    assert(t.order == "c2", "the packed Q8 kernel loads two columns at a time")
    -- Bands of 4 rows: both of an 8 row tile, or the one the 4 row pass is on, whose offset is known at run time.
    const bands = R // 4
    const e = emit.new()
    emit.open(e, `const function {t.name}({tile.SIGNATURE}): ()`)
    emit.line(e, "const wsb = ws :: buffer")
    emit.line(e, "const xsb = xs :: buffer")
    emit.line(e, `const ng = K // {GROUP}`)
    emit.line(e, "const ng4 = ng * 4")
    emit.line(e, "const ng8 = ng * 8")
    emit.open(e, `for i = row0, row0 + rows - 1, {R} do`)
    if R == 8 then
        emit.line(e, "const wb = wOff + i * K")
        emit.line(e, "const sb = wsOff + i * ng4")
    else
        emit.line(e, "const band = i % 8")
        emit.line(e, "const wb = wOff + (i - band) * K + band")
        emit.line(e, "const sb = wsOff + (i - band) * ng4 + band * 4")
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
        emit.line(e, `const xm{s} = buffer.readf32(xsb, xg{s} + g * 8) * {PAIR}`)
        emit.line(e, `const xd{s} = buffer.readf32(xsb, xg{s} + g * 8 + 4)`)
        for b = 0, bands - 1 do
            for a = 0, 1 do
                emit.line(e, `local q{b}_{a}_{s} = xm{s}`)
            end
        end
        emit.line(e, `const xb{s} = xo{s} + g * 256`)
    end
    emit.line(e, "const pk = wb + g * 256")
    for u0 = 0, GROUP - 1, 2 do
        for s = 0, S - 1 do
            for u = u0, u0 + 1 do
                emit.line(e, `const xv{u}_{s} = buffer.readf64(x, {emit.plus(`xb{s}`, 8 * u)})`)
            end
        end
        for b = 0, bands - 1 do
            for u = u0, u0 + 1 do
                emit.line(e, `const wd{b}_{u} = buffer.readu32(w, {emit.plus("pk", 8 * u + 4 * b)})`)
                emit.line(e, `const da{b}_{u} = bit32.band(wd{b}_{u}, 0xFF0000FF)`)
                emit.line(e, `const db{b}_{u} = bit32.band(bit32.lrotate(wd{b}_{u}, 16), 0xFF0000FF)`)
            end
        end
        for b = 0, bands - 1 do
            for s = 0, S - 1 do
                for u = u0, u0 + 1 do
                    emit.line(e, `q{b}_0_{s} += da{b}_{u} * xv{u}_{s}`)
                    emit.line(e, `q{b}_1_{s} += db{b}_{u} * xv{u}_{s}`)
                end
            end
        end
    end
    for s = 0, S - 1 do
        for b = 0, bands - 1 do
            for a = 0, 1 do
                -- Field a of band 0 holds rows 0 low and 3 high, of band 1 rows 2 low and 1 high.
                const lo = if a == 0 then 0 else 2
                const hi = if a == 0 then 3 else 1
                emit.line(e, `const hi{b}_{a}_{s} = math.floor(q{b}_{a}_{s} * {INV_FIELD} + 0.5)`)
                emit.line(
                    e,
                    `t{4 * b + lo}_{s} += buffer.readf32(wsb, {emit.plus("sr", 4 * (4 * b + lo))}) * xd{s} * (q{b}_{a}_{s} - hi{b}_{a}_{s} * {FIELD})`
                )
                emit.line(
                    e,
                    `t{4 * b + hi}_{s} += buffer.readf32(wsb, {emit.plus("sr", 4 * (4 * b + hi))}) * xd{s} * hi{b}_{a}_{s}`
                )
            end
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
