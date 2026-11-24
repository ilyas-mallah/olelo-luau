-- f32 and BF16 kernels over 3 lane vectors. Weights and activations are read as vectors, either with
-- `buffer.readvectorf32x3` (`rv`) or built from three `readf32` calls (`buf`), and each row keeps one accumulator
-- vector per token that is summed across lanes at the end.

const emit = require("../emit")
const tile = require("../tile")

const LANES = 3

--- Columns one loop pass covers, which the stored row length is a multiple of.
const function kstep(t: tile.Tile): number
    return LANES * t.U
end

--- The text of the kernel function for `t`.
const function render(t: tile.Tile): string
    const R, S, U = t.R, t.S, t.U
    const fmt = t.fmt :: string
    const src = t.src :: string
    const wsz = if fmt == "f32" then 4 else 2
    assert(fmt ~= "bf16" or (LANES * U) % 2 == 0, "BF16 converts pairs, so 3 x U must be even")
    assert(fmt == "f32" or src == "rv", "the BF16 kernel reads its converted scratch with the vector read")
    const e = emit.new()

    const function readVec(buf: string, base: string, element: number): string
        if src == "rv" then
            return `buffer.readvectorf32x3({buf}, {emit.plus(base, 4 * element * LANES)})`
        end
        const lanes = {}
        for l = 0, LANES - 1 do
            table.insert(lanes, `buffer.readf32({buf}, {emit.plus(base, 4 * (element * LANES + l))})`)
        end
        return `vector.create({table.concat(lanes, ", ")})`
    end

    -- Loads the weight vectors `us` of row `r`. BF16 first converts the whole pass of the row into the scratch buffer.
    const function loadWeights(r: number, us: { number })
        for _, u in us do
            if fmt == "f32" then
                emit.line(e, `const wv{r}_{u} = {readVec("w", `a{r}`, u)}`)
            else
                if u == 0 then
                    for pp = 0, LANES * U - 1, 2 do
                        const sb = r * U * LANES + pp
                        emit.line(e, `const wd{r}_{pp} = buffer.readu32(w, {emit.plus(`a{r}`, 2 * pp)})`)
                        emit.line(e, `buffer.writeu32(sc, {4 * sb}, bit32.lshift(wd{r}_{pp}, 16))`)
                        emit.line(e, `buffer.writeu32(sc, {4 * (sb + 1)}, bit32.band(wd{r}_{pp}, 0xFFFF0000))`)
                    end
                end
                const base = r * U * LANES + u * LANES
                emit.line(e, `const wv{r}_{u} = buffer.readvectorf32x3(sc, {4 * base})`)
            end
        end
    end

    const function multiply(r: number, u: number)
        for s = 0, S - 1 do
            emit.line(e, `p{r}_{s} = p{r}_{s} + wv{r}_{u} * xv{s}_{u}`)
        end
    end

    const all: { number } = {}
    for u = 0, U - 1 do
        table.insert(all, u)
    end

    emit.open(e, `const function {t.name}({tile.SIGNATURE}): ()`)
    if fmt == "bf16" then
        emit.line(e, `const sc = buffer.create({4 * R * U * LANES})`)
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
            emit.line(e, `local p{r}_{s} = vector.create(0, 0, 0)`)
        end
    end
    emit.open(e, `for k = 0, K - 1, {LANES * U} do`)
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
            emit.line(e, `const xv{s}_{u} = {readVec("x", `xb{s}`, u)}`)
        end
    end
    if t.order == "all" then
        for r = 0, R - 1 do
            loadWeights(r, all)
        end
        for r = 0, R - 1 do
            for u = 0, U - 1 do
                multiply(r, u)
            end
        end
    elseif t.order == "row" then
        for r = 0, R - 1 do
            loadWeights(r, all)
            for u = 0, U - 1 do
                multiply(r, u)
            end
        end
    elseif t.order == "elem" then
        for r = 0, R - 1 do
            for u = 0, U - 1 do
                loadWeights(r, { u })
                multiply(r, u)
            end
        end
    else
        error(`unknown load order {t.order}`)
    end
    emit.close(e)
    for s = 0, S - 1 do
        for r = 0, R - 1 do
            const out = `p{r}_{s}.x + p{r}_{s}.y + p{r}_{s}.z`
            emit.line(
                e,
                `buffer.writef32(y, yBase + ({emit.plusFactor("t", s)} * yStride + {emit.plus("i", r)}) * 4, {out})`
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
