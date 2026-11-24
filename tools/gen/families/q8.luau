-- Q8 kernels: int8 weights with an f32 scale per group of 32, int8 activations with an f32 scale per token. Each
-- group's integer sum is scaled once. This is the family a faster Q8 kernel replaces: add a family file with the same
-- `kstep` and `render`, register it in `generate.luau` and point the Q8 tiles in `tiles.luau` at it.

const emit = require("../emit")
const tile = require("../tile")

const GROUP = 32

--- Columns one loop pass covers, which the stored row length is a multiple of.
const function kstep(_: tile.Tile): number
    return GROUP
end

--- The text of the kernel function for `t`.
const function render(t: tile.Tile): string
    const R, S, U = t.R, t.S, t.U
    const layout = t.layout :: string
    assert(GROUP % U == 0, "the unroll divides the group")
    assert(layout == "rm" or layout == "il4e", "unknown layout")
    assert(layout == "rm" or R % 4 == 0, "the interleaved layout takes 4 rows at a time")
    const e = emit.new()

    --- The weight address of row `r`, unrolled element `u`, in the layout.
    const function weightAddress(r: number, u: number): string
        if layout == "rm" then
            return emit.plus(`a{r}`, u)
        end
        const b, rr = r // 4, r % 4
        return emit.plus(`pk{b}`, 4 * u + rr)
    end

    emit.open(e, `const function {t.name}({tile.SIGNATURE}): ()`)
    emit.line(e, "const wsb = ws :: buffer")
    emit.line(e, "const xsb = xs :: buffer")
    emit.line(e, `const ng = K // {GROUP}`)
    emit.line(e, "const ng4 = ng * 4")
    emit.open(e, `for i = row0, row0 + rows - 1, {R} do`)
    if layout == "rm" then
        for r = 0, R - 1 do
            emit.line(e, `const wr{r} = wOff + {emit.plusFactor("i", r)} * K`)
        end
    else
        for b = 0, R // 4 - 1 do
            emit.line(e, `const wb{b} = wOff + {emit.plusFactor("i", 4 * b)} * K`)
        end
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
    emit.open(e, `for g = 0, ng - 1 do`)
    emit.line(e, `const gk = g * {GROUP}`)
    for r = 0, R - 1 do
        emit.line(e, `const sr{r} = wsOff + {emit.plusFactor("i", r)} * ng4 + g * 4`)
        for s = 0, S - 1 do
            emit.line(e, `local p{r}_{s} = 0`)
        end
    end
    if U == GROUP then
        emit.line(e, "const k = gk")
    else
        emit.open(e, `for k = gk, gk + {GROUP - 1}, {U} do`)
    end
    for s = 0, S - 1 do
        emit.line(e, `const xb{s} = xo{s} + k`)
    end
    if layout == "rm" then
        for r = 0, R - 1 do
            emit.line(e, `const a{r} = wr{r} + k`)
        end
    else
        for b = 0, R // 4 - 1 do
            emit.line(e, `const pk{b} = wb{b} + k * 4`)
        end
    end
    for s = 0, S - 1 do
        for u = 0, U - 1 do
            emit.line(e, `const xv{s}_{u} = buffer.readi8(x, {emit.plus(`xb{s}`, u)})`)
        end
    end
    for r = 0, R - 1 do
        if t.order == "elem" then
            for u = 0, U - 1 do
                emit.line(e, `const wv{r}_{u} = buffer.readi8(w, {weightAddress(r, u)})`)
                for s = 0, S - 1 do
                    emit.line(e, `p{r}_{s} += wv{r}_{u} * xv{s}_{u}`)
                end
            end
        elseif string.sub(t.order, 1, 1) == "g" then
            const chunk = tonumber(string.sub(t.order, 2)) :: number
            for u0 = 0, U - 1, chunk do
                for u = u0, u0 + chunk - 1 do
                    emit.line(e, `const wv{r}_{u} = buffer.readi8(w, {weightAddress(r, u)})`)
                end
                for u = u0, u0 + chunk - 1 do
                    for s = 0, S - 1 do
                        emit.line(e, `p{r}_{s} += wv{r}_{u} * xv{s}_{u}`)
                    end
                end
            end
        elseif t.order == "all" then
            for u = 0, U - 1 do
                emit.line(e, `const wv{r}_{u} = buffer.readi8(w, {weightAddress(r, u)})`)
            end
            for s = 0, S - 1 do
                for u = 0, U - 1 do
                    emit.line(e, `p{r}_{s} += wv{r}_{u} * xv{s}_{u}`)
                end
            end
        else
            error(`unknown load order {t.order}`)
        end
    end
    if U ~= GROUP then
        emit.close(e)
    end
    for r = 0, R - 1 do
        for s = 0, S - 1 do
            emit.line(e, `t{r}_{s} += buffer.readf32(wsb, sr{r}) * p{r}_{s}`)
        end
    end
    emit.close(e)
    for s = 0, S - 1 do
        emit.line(e, `const xsv{s} = buffer.readf32(xsb, {emit.plusFactor("t", s)} * 4)`)
        for r = 0, R - 1 do
            emit.line(
                e,
                `buffer.writef32(y, yBase + ({emit.plusFactor("t", s)} * yStride + {emit.plus("i", r)}) * 4, t{r}_{s} * xsv{s})`
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
