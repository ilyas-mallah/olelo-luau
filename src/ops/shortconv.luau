--!native
-- The gated short convolution of LFM2. The input projection yields three runs of `hidden` values per token: a gate B, a
-- gate C and a value x. The block convolves `B * x` along the positions with one short filter per channel, looking back
-- `kernel - 1` positions, and returns `C` times the result. `B * x` of every position is kept in a history, so a
-- sequence can be rewound or restored like a KV cache.

--- Writes `B * x` of `tokens` rows into the history at the positions `pos0` and after, then the block's output
--- `C * conv(history)` into `out`. A row of `bcx` holds B, C and x one after the other. `taps` holds the filter of
--- channel `c` at `c * kernel`, its last tap on the current position. Positions before 0 count as zero. A token of
--- `out` is `outStride` floats apart and the history has `hidden` floats per position.
const function run(
    bcx: buffer,
    bcxStride: number,
    hidden: number,
    taps: buffer,
    kernel: number,
    history: buffer,
    pos0: number,
    tokens: number,
    out: buffer,
    outStride: number
)
    for t = 0, tokens - 1 do
        const row = t * bcxStride * 4
        const at = (pos0 + t) * hidden * 4
        for c = 0, hidden - 1 do
            buffer.writef32(
                history,
                at + c * 4,
                buffer.readf32(bcx, row + c * 4) * buffer.readf32(bcx, row + (2 * hidden + c) * 4)
            )
        end
    end
    const reach = kernel - 1
    for t = 0, tokens - 1 do
        const row = t * bcxStride * 4
        const oo = t * outStride * 4
        const position = pos0 + t
        -- The first tap that lands on a position that exists.
        const first = math.max(0, reach - position)
        for c = 0, hidden - 1 do
            local sum = 0
            for k = first, reach do
                sum += buffer.readf32(taps, (c * kernel + k) * 4) * buffer.readf32(
                    history,
                    ((position - reach + k) * hidden + c) * 4
                )
            end
            buffer.writef32(out, oo + c * 4, buffer.readf32(bcx, row + (hidden + c) * 4) * sum)
        end
    end
end

return {
    run = run,
}
