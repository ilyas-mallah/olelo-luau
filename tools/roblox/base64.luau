-- Base64 text of bytes, the encoding of a chunk ModuleScript. `backends/roblox/weights` decodes it.
const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

--- The base64 text of the bytes of `data`, with padding.
const function encode(data: string): string
    const parts: { string } = {}
    for i = 1, #data, 3 do
        const a, b, c = string.byte(data, i, i + 2)
        const n = a * 65536 + (b or 0) * 256 + (c or 0)
        const s1, s2, s3, s4 = n // 262144, n // 4096 % 64, n // 64 % 64, n % 64
        table.insert(
            parts,
            string.sub(ALPHABET, s1 + 1, s1 + 1)
                .. string.sub(ALPHABET, s2 + 1, s2 + 1)
                .. (if b ~= nil then string.sub(ALPHABET, s3 + 1, s3 + 1) else "=")
                .. (if c ~= nil then string.sub(ALPHABET, s4 + 1, s4 + 1) else "=")
        )
    end
    return table.concat(parts)
end

return {
    encode = encode,
}
