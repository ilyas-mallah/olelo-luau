--!native
-- Weight sources for Roblox, where a file cannot be read. The container is cut into chunks of a fixed size. A chunk is a
-- ModuleScript that returns its bytes as a base64 string, or the body of an HTTP range request. The sizes each accepts
-- are in docs/roblox.md.

const host = require("@src/host")

--- The 6 bit value of each base64 character by its byte, 255 for a byte that is not one.
const DECODE: { number } = table.create(256, 255)
do
    const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    for i = 1, 64 do
        DECODE[string.byte(alphabet, i) + 1] = i - 1
    end
end

--- The bytes of the base64 text `text`, padding optional, in a new buffer.
const function decodeBase64(text: string): buffer
    const length = #text
    local padding = 0
    if length >= 1 and string.byte(text, length) == 61 then
        padding = 1
        if length >= 2 and string.byte(text, length - 1) == 61 then
            padding = 2
        end
    end
    const bytes = (length - padding) * 3 // 4
    const out = buffer.create(bytes)
    local at = 0
    local i = 1
    while i + 3 <= length - padding do
        const a, b, c, d = string.byte(text, i, i + 3)
        const bits = DECODE[a + 1] * 262144 + DECODE[b + 1] * 4096 + DECODE[c + 1] * 64 + DECODE[d + 1]
        buffer.writeu8(out, at, bits // 65536)
        buffer.writeu8(out, at + 1, bits // 256 % 256)
        buffer.writeu8(out, at + 2, bits % 256)
        at += 3
        i += 4
    end
    const rest = length - padding - i + 1
    if rest == 3 then
        const a, b, c = string.byte(text, i, i + 2)
        const bits = DECODE[a + 1] * 262144 + DECODE[b + 1] * 4096 + DECODE[c + 1] * 64
        buffer.writeu8(out, at, bits // 65536)
        buffer.writeu8(out, at + 1, bits // 256 % 256)
    elseif rest == 2 then
        const a, b = string.byte(text, i, i + 1)
        buffer.writeu8(out, at, (DECODE[a + 1] * 262144 + DECODE[b + 1] * 4096) // 65536)
    end
    return out
end

--- How many chunks `fromChunks` keeps. A worker reads its weights and their scales as two streams far apart in the
--- container, and a small read such as a norm vector needs one chunk.
const KEPT = 4

--- A source over `size` bytes cut into chunks of `chunkBytes`, chunk `i` from 0 being what `load(i)` returns. The last
--- `KEPT` chunks loaded are kept.
const function fromChunks(size: number, chunkBytes: number, load: (index: number) -> buffer): host.WeightSource
    const keptIndex: { number } = {}
    const keptData: { buffer } = {}
    const function chunk(index: number): buffer
        for k, kept in keptIndex do
            if kept == index then
                return keptData[k]
            end
        end
        const data = load(index)
        if #keptIndex >= KEPT then
            table.remove(keptIndex, 1)
            table.remove(keptData, 1)
        end
        table.insert(keptIndex, index)
        table.insert(keptData, data)
        return data
    end
    return {
        size = function(): number
            return size
        end,
        read = function(offset: number, length: number, dst: buffer, dstOffset: number)
            if offset < 0 or offset + length > size then
                error(`read of {length} bytes at {offset} is outside the {size} bytes of the container`, 0)
            end
            local done = 0
            while done < length do
                const at = offset + done
                const index = at // chunkBytes
                const inside = at - index * chunkBytes
                const take = math.min(length - done, chunkBytes - inside)
                const data = chunk(index)
                if inside + take > buffer.len(data) then
                    error(`chunk {index} holds {buffer.len(data)} bytes and the read needs {inside + take}`, 0)
                end
                buffer.copy(dst, dstOffset + done, data, inside, take)
                done += take
            end
        end,
    }
end

--- A source over the ModuleScripts named "1", "2" and so on in `folder`, each returning one chunk as base64 text.
--- `require` is the host's, so a test can give its own.
const function fromModules(
    folder: any,
    size: number,
    chunkBytes: number,
    require: (module: any) -> any
): host.WeightSource
    return fromChunks(size, chunkBytes, function(index: number): buffer
        const module = folder:FindFirstChild(tostring(index + 1))
        if module == nil then
            error(`the folder has no chunk {index + 1}`, 0)
        end
        const text = require(module)
        if type(text) ~= "string" then
            error(`chunk {index + 1} does not return a string`, 0)
        end
        return decodeBase64(text :: string)
    end)
end

--- The request a range read of an HTTP source makes, and the reply it needs. `HttpService:RequestAsync` has this shape.
export type Http = {
    RequestAsync: (self: Http, options: { [string]: any }) -> { Success: boolean, StatusCode: number, Body: string },
}

--- A source over a file at `url` that answers range requests, in chunks of `chunkBytes`. Server scripts only.
const function fromHttp(http: Http, url: string, size: number, chunkBytes: number): host.WeightSource
    return fromChunks(size, chunkBytes, function(index: number): buffer
        const first = index * chunkBytes
        const last = math.min(size, first + chunkBytes) - 1
        const response = http:RequestAsync({
            Url = url,
            Method = "GET",
            Headers = { Range = `bytes={first}-{last}` },
        })
        if not response.Success or (response.StatusCode ~= 206 and response.StatusCode ~= 200) then
            error(`the request for bytes {first} to {last} of {url} failed with status {response.StatusCode}`, 0)
        end
        if response.StatusCode == 200 and #response.Body ~= last - first + 1 then
            -- A server that ignores the range sends the whole file.
            return buffer.fromstring(string.sub(response.Body, first + 1, last + 1))
        end
        return buffer.fromstring(response.Body)
    end)
end

--- Seconds of work between the yields of a load. A script that runs about ten seconds without yielding is stopped with
--- "Script timeout", which decoding the chunk modules of a model reaches.
const LOAD_SLICE = 0.1

--- `source` that waits a frame once `slice` seconds have passed since the last wait, as `read` returns. The engine reads
--- its weights while it loads, so this keeps a load of chunk modules inside the script time limit.
const function paced(
    source: host.WeightSource,
    clock: () -> number,
    wait: (number?) -> number,
    slice: number
): host.WeightSource
    local last = clock()
    return {
        size = source.size,
        read = function(offset: number, length: number, dst: buffer, dstOffset: number)
            source.read(offset, length, dst, dstOffset)
            if clock() - last >= slice then
                wait()
                last = clock()
            end
        end,
    }
end

return {
    KEPT = KEPT,
    LOAD_SLICE = LOAD_SLICE,
    paced = paced,
    decodeBase64 = decodeBase64,
    fromChunks = fromChunks,
    fromModules = fromModules,
    fromHttp = fromHttp,
}
