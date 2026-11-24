-- f32 and BF16 kernels with scalar double math. BF16 weights are converted in pairs through a scratch buffer: the two
-- halves of a 32 bit read are shifted into the high half of two f32 slots and read back as f32.

const emit = require("../emit")
const tile = require("../tile")

--- Columns one loop pass covers, which the stored row length is a multiple of.
const function kstep(t: tile.Tile): number
    return t.U
end

--- The text of the kernel function for `t`.
const function render(t: tile.Tile): string
    const R, S, U = t.R, t.S, t.U
    const fmt = t.fmt :: string
    const wsz = if fmt == "f32" then 4 else 2
    assert(fmt == "f32" or U % 2 == 0, "BF16 converts pairs, so U must be even")
    const e = emit.new()

    emit.open(e, `const function {t.name}({tile.SIGNATURE}): ()`)
    if fmt == "bf16" then
        emit.line(e, `const sc = buffer.create({4 * R * U})`)
    end
    emit.open(e, `for i = row0, row0 + rows - 1, {R} do`)
    for r = 0, R - 1 do
        emit.line(e, `const wr{r} = wOff + {emit.plusFactor("i", r)} * K * {wsz}`)
    end
    emit.open(e, `for t = 0, T - 1, {S} do`)
    for s = 0, S - 1 do
        emit.line(e, `const xo{s} = {emit.plusFactor("t", s)} * K * 4`)
    end
    for r = 0, R - 1 do
        for s = 0, S - 1 do
            emit.line(e, `local p{r}_{s} = 0`)
        end
    end
    emit.open(e, `for k = 0, K - 1, {U} do`)
    emit.line(e, `const kw = k * {wsz}`)
    emit.line(e, "const kx = k * 4")
    for r = 0, R - 1 do
        emit.line(e, `const a{r} = wr{r} + kw`)
    end
    for s = 0, S - 1 do
        emit.line(e, `const xb{s} = xo{s} + kx`)
    end
    for s = 0, S - 1 do
        for u = 0, U - 1 do
            emit.line(e, `const xv{s}_{u} = buffer.readf32(x, {emit.plus(`xb{s}`, 4 * u)})`)
        end
    end
    for r = 0, R - 1 do
        if fmt == "f32" then
            for u = 0, U - 1 do
                emit.line(e, `const wv{r}_{u} = buffer.readf32(w, {emit.plus(`a{r}`, 4 * u)})`)
            end
        else
            for u = 0, U - 1, 2 do
                const i0 = r * U + u
                emit.line(e, `const wd{r}_{u} = buffer.readu32(w, {emit.plus(`a{r}`, 2 * u)})`)
                emit.line(e, `buffer.writeu32(sc, {4 * i0}, bit32.lshift(wd{r}_{u}, 16))`)
                emit.line(e, `buffer.writeu32(sc, {4 * (i0 + 1)}, bit32.band(wd{r}_{u}, 0xFFFF0000))`)
                emit.line(e, `const wv{r}_{u} = buffer.readf32(sc, {4 * i0})`)
                emit.line(e, `const wv{r}_{u + 1} = buffer.readf32(sc, {4 * (i0 + 1)})`)
            end
        end
        for s = 0, S - 1 do
            for u = 0, U - 1 do
                emit.line(e, `p{r}_{s} += wv{r}_{u} * xv{s}_{u}`)
            end
        end
    end
    emit.close(e)
    for s = 0, S - 1 do
        for r = 0, R - 1 do
            emit.line(
                e,
                `buffer.writef32(y, yBase + ({emit.plusFactor("t", s)} * yStride + {emit.plus("i", r)}) * 4, p{r}_{s})`
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
