--!native
-- Splits a matrix into parts that each fit one buffer. A part is a run of whole tiles of rows and never spans two buffers.

const formats = require("./formats")
const types = require("./types")

type Part = types.Part

--- A planned part, before its buffers exist.
export type PartPlan = {
    rowStart: number,
    rowCount: number,
}

--- Cuts `rows` rows of `rowBytes` bytes into parts of at most `maxBytes` bytes on multiples of `rowTile` rows. `rows`
--- is a multiple of `rowTile`. Errors when one tile of rows is larger than `maxBytes`.
const function plan(rows: number, rowBytes: number, rowTile: number, maxBytes: number): { PartPlan }
    assert(rowTile >= 1 and rows % rowTile == 0, "rows must be a multiple of the tile rows")
    const perPart = (maxBytes // rowBytes) // rowTile * rowTile
    if perPart < rowTile then
        error(`a tile of {rowTile} rows of {rowBytes} bytes does not fit in a buffer of {maxBytes} bytes`)
    end
    const parts: { PartPlan } = {}
    local start = 0
    while start < rows do
        const count = math.min(perPart, rows - start)
        table.insert(parts, { rowStart = start, rowCount = count })
        start += count
    end
    return parts
end

--- Allocates the buffers of a plan for a format with `kpad` stored weights per row. A plan with no rows gives no parts.
const function create(
    partPlans: { PartPlan },
    name: types.FormatName,
    kpad: number,
    layout: types.LayoutName?
): { Part }
    const rowBytes = formats.rowBytes(name, kpad, layout)
    const scaleBytes = formats.scaleRowBytes(name, kpad)
    const parts: { Part } = {}
    for _, p in partPlans do
        local scales: buffer? = nil
        if scaleBytes > 0 then
            scales = buffer.create(p.rowCount * scaleBytes)
        end
        table.insert(parts, {
            buf = buffer.create(p.rowCount * rowBytes),
            offset = 0,
            scales = scales,
            scaleOffset = 0,
            rowStart = p.rowStart,
            rowCount = p.rowCount,
        })
    end
    return parts
end

--- The index of the part that holds `row`, by binary search over the parts in row order.
const function partIndex(parts: { Part }, row: number): number
    local lo, hi = 1, #parts
    while lo < hi do
        const mid = (lo + hi + 1) // 2
        if parts[mid].rowStart <= row then
            lo = mid
        else
            hi = mid - 1
        end
    end
    const part = parts[lo]
    assert(part ~= nil and row >= part.rowStart and row < part.rowStart + part.rowCount, "row outside the tensor")
    return lo
end

return {
    plan = plan,
    create = create,
    partIndex = partIndex,
}
