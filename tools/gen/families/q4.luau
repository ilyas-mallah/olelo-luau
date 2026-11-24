-- Q4 kernels: two 4 bit weights per byte, a group of 32 in 16 bytes with element j in the low nibble and element j + 16
-- in the high nibble, the weight being the nibble minus 8. The minus 8 is applied once per group through the sum of the
-- group's activations, which the kernel computes for every token before the row loop. The activations are int8 with
-- a scale per group of 32, as llama.cpp quantizes them for Q4_0, so a group adds `wscale * xscale * (p - 8 * sx)`.

const emit = require("../emit")
const tile = require("../tile")

const GROUP = 32
const HALF = 16

--- Columns one loop pass covers, which the stored row length is a multiple of.
const function kstep(_: tile.Tile): number
    return GROUP
end

--- The text of the kernel function for `t`.
const function render(t: tile.Tile): string
    const R, S, U = t.R, t.S, t.U
    assert(HALF % U == 0, "the unroll divides half a group")
    assert(t.order == "all", "the Q4 kernel loads a row's bytes before its multiplies")
    const e = emit.new()

    emit.open(e, `const function {t.name}({tile.SIGNATURE}): ()`)
    emit.line(e, "const wsb = ws :: buffer")
    emit.line(e, "const xsb = xs :: buffer")
    emit.line(e, `const ng = K // {GROUP}`)
    emit.line(e, "const ng4 = ng * 4")
    emit.line(e, "const K2 = K // 2")
    emit.line(e, "const xg = buffer.create(T * ng4)")
    emit.open(e, "for t = 0, T - 1 do")
    emit.open(e, "for g = 0, ng - 1 do")
    emit.line(e, `const b = t * K + g * {GROUP}`)
    emit.line(e, "local sum = 0")
    emit.line(e, `for j = 0, {GROUP - 1} do sum += buffer.readi8(x, b + j) end`)
    emit.line(e, "buffer.writei32(xg, t * ng4 + g * 4, sum)")
    emit.close(e)
    emit.close(e)
    emit.open(e, `for i = row0, row0 + rows - 1, {R} do`)
    for r = 0, R - 1 do
        emit.line(e, `const wr{r} = wOff + {emit.plusFactor("i", r)} * K2`)
    end
    emit.open(e, `for t = 0, T - 1, {S} do`)
    for s = 0, S - 1 do
        emit.line(e, `const xo{s} = {emit.plusFactor("t", s)} * K`)
    end
    for r = 0, R - 1 do
        for s = 0, S - 1 do
            emit.line(e, `local t{r}_{s} = 0`)
        end
    end
    emit.open(e, "for g = 0, ng - 1 do")
    for r = 0, R - 1 do
        emit.line(e, `const sr{r} = wsOff + {emit.plusFactor("i", r)} * ng4 + g * 4`)
        emit.line(e, `const wg{r} = wr{r} + g * {HALF}`)
        for s = 0, S - 1 do
            emit.line(e, `local p{r}_{s} = 0`)
        end
    end
    for s = 0, S - 1 do
        emit.line(e, `const xg{s} = xo{s} + g * {GROUP}`)
    end
    if U == HALF then
        emit.line(e, "const j = 0")
    else
        emit.open(e, `for j = 0, {HALF - 1}, {U} do`)
    end
    for s = 0, S - 1 do
        for u = 0, U - 1 do
            emit.line(e, `const xl{s}_{u} = buffer.readi8(x, {emit.plus(`xg{s} + j`, u)})`)
            emit.line(e, `const xh{s}_{u} = buffer.readi8(x, xg{s} + j + {u + HALF})`)
        end
    end
    for r = 0, R - 1 do
        for u = 0, U - 1 do
            emit.line(e, `const b{r}_{u} = buffer.readu8(w, {emit.plus(`wg{r} + j`, u)})`)
            emit.line(e, `const lo{r}_{u} = bit32.band(b{r}_{u}, 15)`)
            emit.line(e, `const hi{r}_{u} = bit32.rshift(b{r}_{u}, 4)`)
        end
        for s = 0, S - 1 do
            for u = 0, U - 1 do
                emit.line(e, `p{r}_{s} += lo{r}_{u} * xl{s}_{u} + hi{r}_{u} * xh{s}_{u}`)
            end
        end
    end
    if U ~= HALF then
        emit.close(e)
    end
    for s = 0, S - 1 do
        emit.line(e, `const sx{s} = buffer.readi32(xg, {emit.plusFactor("t", s)} * ng4 + g * 4)`)
        emit.line(e, `const xd{s} = buffer.readf32(xsb, ({emit.plusFactor("t", s)} * ng + g) * 4)`)
    end
    for r = 0, R - 1 do
        for s = 0, S - 1 do
            emit.line(e, `t{r}_{s} += buffer.readf32(wsb, sr{r}) * xd{s} * (p{r}_{s} - 8 * sx{s})`)
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
