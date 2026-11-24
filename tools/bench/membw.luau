-- Memory copy bandwidth of one process, the hardware ceiling the decode rate is compared with.
--
--   lute run tools/bench/membw.luau [mb=256] [seconds=4]
--
-- Copies one buffer of `mb` MB onto another with `buffer.copy` (a memcpy) for `seconds` and prints `MEMBW` with the GB/s
-- moved (bytes copied) and the GB/s of memory traffic (read plus write). Run several at once, one per CPU, for the
-- limit of the whole memory system, see docs/benchmarks.md.
const process = require("@std/process")

const mb = tonumber(process.args[2]) or 256
const seconds = tonumber(process.args[3]) or 4
const size = mb * 1048576
const src, dst = buffer.create(size), buffer.create(size)
for i = 0, size - 1, 4096 do
    buffer.writeu8(src, i, 1)
    buffer.writeu8(dst, i, 2)
end

const start = os.clock()
local copied = 0
while os.clock() - start < seconds do
    buffer.copy(dst, 0, src, 0, size)
    copied += size
end
const elapsed = os.clock() - start
print(
    string.format(
        "MEMBW mb=%d copied_gbps=%.2f traffic_gbps=%.2f",
        mb,
        copied / elapsed / 1e9,
        2 * copied / elapsed / 1e9
    )
)
