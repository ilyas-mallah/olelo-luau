--!native
-- The gated MLP activation: `act = f(gate) * up`.

const types = require("./types")

type Activation = types.Activation

const function silu(x: number): number
    return x / (1 + math.exp(-x))
end

const SQRT_2_OVER_PI = math.sqrt(2 / math.pi)

const function geluTanh(x: number): number
    const inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
    -- tanh(a) = 1 - 2 / (exp(2a) + 1), which stays finite for large a.
    return 0.5 * x * (2 - 2 / (math.exp(2 * inner) + 1))
end

--- `act[t][i] = f(gate[t][i]) * up[t][i]` for `tokens` rows of `n` values.
const function gated(
    kind: Activation,
    gate: buffer,
    gateStride: number,
    up: buffer,
    upStride: number,
    act: buffer,
    actStride: number,
    n: number,
    tokens: number
)
    for t = 0, tokens - 1 do
        const g, u, a = t * gateStride * 4, t * upStride * 4, t * actStride * 4
        if kind == "silu" then
            for i = 0, n - 1 do
                buffer.writef32(act, a + i * 4, silu(buffer.readf32(gate, g + i * 4)) * buffer.readf32(up, u + i * 4))
            end
        else
            for i = 0, n - 1 do
                buffer.writef32(
                    act,
                    a + i * 4,
                    geluTanh(buffer.readf32(gate, g + i * 4)) * buffer.readf32(up, u + i * 4)
                )
            end
        end
    end
end

return {
    gated = gated,
}
