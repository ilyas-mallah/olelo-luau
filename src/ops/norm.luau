--!native
-- Normalization and the residual add. Rows are tokens of f32 values in buffers with a row stride in floats.

--- `out = x / rms(x) * w` for `tokens` rows of `n` values. `plusOne` multiplies by `1 + w`, as Gemma's norm does.
const function rms(
    x: buffer,
    xStride: number,
    w: buffer,
    out: buffer,
    outStride: number,
    n: number,
    tokens: number,
    eps: number,
    plusOne: boolean
)
    for t = 0, tokens - 1 do
        const xo = t * xStride * 4
        const oo = t * outStride * 4
        local sum = 0
        for i = 0, n - 1 do
            const v = buffer.readf32(x, xo + i * 4)
            sum += v * v
        end
        const scale = 1 / math.sqrt(sum / n + eps)
        for i = 0, n - 1 do
            local weight = buffer.readf32(w, i * 4)
            if plusOne then
                weight += 1
            end
            buffer.writef32(out, oo + i * 4, buffer.readf32(x, xo + i * 4) * scale * weight)
        end
    end
end

--- `h += delta` for `tokens` rows of `n` values.
const function add(h: buffer, hStride: number, delta: buffer, deltaStride: number, n: number, tokens: number)
    for t = 0, tokens - 1 do
        const ho = t * hStride * 4
        const dO = t * deltaStride * 4
        for i = 0, n - 1 do
            buffer.writef32(h, ho + i * 4, buffer.readf32(h, ho + i * 4) + buffer.readf32(delta, dO + i * 4))
        end
    end
end

--- `x += bias` for `tokens` rows of `n` values.
const function addBias(x: buffer, xStride: number, bias: buffer, n: number, tokens: number)
    for t = 0, tokens - 1 do
        const xo = t * xStride * 4
        for i = 0, n - 1 do
            buffer.writef32(x, xo + i * 4, buffer.readf32(x, xo + i * 4) + buffer.readf32(bias, i * 4))
        end
    end
end

--- Normalizes in place each of `heads` runs of `headDim` values of `tokens` rows, with the weight `w` of `headDim`
--- values shared by every head, as Qwen3 does to q and k.
const function rmsHeads(
    x: buffer,
    xStride: number,
    w: buffer,
    heads: number,
    headDim: number,
    tokens: number,
    eps: number,
    plusOne: boolean
)
    for t = 0, tokens - 1 do
        for h = 0, heads - 1 do
            const at = (t * xStride + h * headDim) * 4
            local sum = 0
            for i = 0, headDim - 1 do
                const v = buffer.readf32(x, at + i * 4)
                sum += v * v
            end
            const scale = 1 / math.sqrt(sum / headDim + eps)
            for i = 0, headDim - 1 do
                local weight = buffer.readf32(w, i * 4)
                if plusOne then
                    weight += 1
                end
                buffer.writef32(x, at + i * 4, buffer.readf32(x, at + i * 4) * scale * weight)
            end
        end
    end
end

return {
    rms = rms,
    addBias = addBias,
    rmsHeads = rmsHeads,
    add = add,
}
