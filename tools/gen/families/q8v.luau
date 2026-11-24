-- The vector Q8 kernel: an f32 lane holds three offset binary weights exactly, (a + 256 b + 65536 c) / 65536, so one
-- `buffer.readvectorf32x3` brings 9 weights, 3 rows by 3 columns. The stored tile is 12 rows (4 vectors of 3 rows). A
-- kernel pass covers 12 rows by 1 token, or 6 rows by 2 tokens, because 4 row vectors by 2 tokens needs more than 23
-- spill slots on the Windows x64 ABI and is skipped by native lowering there. A group of 32 columns is 11 steps. The
-- activations are splat vectors, see `quant.splatActivations`.
-- Layout and exactness are argued in docs/kernels.md.

const emit = require("../emit")
const tile = require("../tile")

const GROUP = 32
const STEPS = 11
const ROW_VECTORS = 4
--- Bytes of one group of one token in the activations, and of one group of a tile in the weights and the scales.
const X_GROUP = 12 + STEPS * 36
const W_GROUP = STEPS * ROW_VECTORS * 12
const S_GROUP = ROW_VECTORS * 12

--- Columns one loop pass covers, which the stored row length is a multiple of.
const function kstep(_: tile.Tile): number
    return GROUP
end

--- The text of the kernel function for `t`.
const function render(t: tile.Tile): string
    const R, S = t.R, t.S
    assert(
        (R == 12 and S == 1) or (R == 6 and S == 2),
        "the vector Q8 pass is 12 rows by 1 token or 6 rows by 2 tokens"
    )
    const vectors = R // 3
    --- A pass of 6 rows is the first or the second half of a stored tile, `half` says which.
    const halves = R == 6
    assert(t.U == GROUP, "the unroll is a whole group")
    const e = emit.new()
    emit.open(e, `const function {t.name}({tile.SIGNATURE}): ()`)
    emit.line(e, "const xsb = xs :: buffer")
    emit.line(e, "const wsb = ws :: buffer")
    emit.line(e, `const ng = K // {GROUP}`)
    emit.line(e, `const tw = ng * {W_GROUP}`)
    emit.line(e, `const ts = ng * {S_GROUP}`)
    emit.open(e, `for i = row0, row0 + rows - 1, {R} do`)
    emit.line(e, `const wb = wOff + i // 12 * tw`)
    emit.line(e, `const sb = wsOff + i // 12 * ts`)
    if halves then
        emit.line(e, "const hv = i % 12 // 3")
    end
    emit.open(e, `for t = 0, T - 1, {S} do`)
    for s = 0, S - 1 do
        emit.line(e, `const xo{s} = {emit.plusFactor("t", s)} * ng * {X_GROUP}`)
        emit.line(e, `const xd{s} = {emit.plusFactor("t", s)} * ng * 12`)
        for r = 0, vectors - 1 do
            emit.line(e, `local tv{r}_{s} = vector.zero`)
        end
    end
    emit.open(e, "for g = 0, ng - 1 do")
    emit.line(e, if halves then `const pw = wb + g * {W_GROUP} + hv * 12` else `const pw = wb + g * {W_GROUP}`)
    emit.line(e, if halves then `const sr = sb + g * {S_GROUP} + hv * 12` else `const sr = sb + g * {S_GROUP}`)
    for s = 0, S - 1 do
        emit.line(e, `const px{s} = xo{s} + g * {X_GROUP}`)
        emit.line(e, `const init{s} = buffer.readvectorf32x3(x, px{s})`)
        for r = 0, vectors - 1 do
            emit.line(e, `local acc{r}_{s} = init{s}`)
        end
    end
    for j = 0, STEPS - 1 do
        emit.open(e, "do")
        for s = 0, S - 1 do
            const base = 12 + j * 36
            emit.line(e, `const xa{s} = buffer.readvectorf32x3(x, {emit.plus(`px{s}`, base)})`)
            emit.line(e, `const xb{s} = buffer.readvectorf32x3(x, {emit.plus(`px{s}`, base + 12)})`)
            emit.line(e, `const xc{s} = buffer.readvectorf32x3(x, {emit.plus(`px{s}`, base + 24)})`)
        end
        for r = 0, vectors - 1 do
            emit.open(e, "do")
            emit.line(e, `const v = buffer.readvectorf32x3(w, {emit.plus("pw", (j * ROW_VECTORS + r) * 12)})`)
            emit.line(e, "const c = vector.floor(v)")
            emit.line(e, "const m = v * 256")
            emit.line(e, "const f = vector.floor(m)")
            emit.line(e, "const a = m - f")
            for s = 0, S - 1 do
                emit.line(e, `acc{r}_{s} = acc{r}_{s} + ((a * xa{s} + f * xb{s}) + c * xc{s})`)
            end
            emit.close(e)
        end
        emit.close(e)
    end
    for s = 0, S - 1 do
        emit.line(e, `const xg{s} = buffer.readvectorf32x3(xsb, xd{s} + g * 12)`)
    end
    for r = 0, vectors - 1 do
        emit.line(e, `const sv{r} = buffer.readvectorf32x3(wsb, {emit.plus("sr", 12 * r)})`)
        for s = 0, S - 1 do
            emit.line(e, `tv{r}_{s} = tv{r}_{s} + acc{r}_{s} * (sv{r} * xg{s})`)
        end
    end
    emit.close(e)
    for s = 0, S - 1 do
        for r = 0, vectors - 1 do
            emit.line(
                e,
                `buffer.writevectorf32x3(y, yBase + ({emit.plusFactor("t", s)} * yStride + {emit.plus("i", 3 * r)}) * 4, tv{r}_{s})`
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
