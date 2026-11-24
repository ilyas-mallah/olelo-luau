--!native
--!nonstrict
-- A 4 row by 2 token vector tile of the Q8 kernel, one step. Native on x64 and a64, skipped on the x64_ms target.
local function tile(
    w: buffer,
    wOff: number,
    ws: buffer?,
    wsOff: number,
    x: buffer,
    xs: buffer?,
    y: buffer,
    yBase: number,
    yStride: number,
    row0: number,
    rows: number,
    K: number,
    T: number
): ()
    local xsb = xs :: buffer
    local wsb = ws :: buffer
    local ng = K // 32
    local tw = ng * 528
    local ts = ng * 48
    for i = row0, row0 + rows - 1, 12 do
        local wb = wOff + i // 12 * tw
        local sb = wsOff + i // 12 * ts
        for t = 0, T - 1, 2 do
            local xo0 = t * ng * 408
            local xd0 = t * ng * 12
            local tv0_0 = vector.zero
            local tv1_0 = vector.zero
            local tv2_0 = vector.zero
            local tv3_0 = vector.zero
            local xo1 = (t + 1) * ng * 408
            local xd1 = (t + 1) * ng * 12
            local tv0_1 = vector.zero
            local tv1_1 = vector.zero
            local tv2_1 = vector.zero
            local tv3_1 = vector.zero
            for g = 0, ng - 1 do
                local pw = wb + g * 528
                local sr = sb + g * 48
                local px0 = xo0 + g * 408
                local init0 = buffer.readvectorf32x3(x, px0)
                local acc0_0 = init0
                local acc1_0 = init0
                local acc2_0 = init0
                local acc3_0 = init0
                local px1 = xo1 + g * 408
                local init1 = buffer.readvectorf32x3(x, px1)
                local acc0_1 = init1
                local acc1_1 = init1
                local acc2_1 = init1
                local acc3_1 = init1
                do
                    local xa0 = buffer.readvectorf32x3(x, px0 + 12 + 0 * 36)
                    local xb0 = buffer.readvectorf32x3(x, px0 + 12 + 0 * 36 + 12)
                    local xc0 = buffer.readvectorf32x3(x, px0 + 12 + 0 * 36 + 24)
                    local xa1 = buffer.readvectorf32x3(x, px1 + 12 + 0 * 36)
                    local xb1 = buffer.readvectorf32x3(x, px1 + 12 + 0 * 36 + 12)
                    local xc1 = buffer.readvectorf32x3(x, px1 + 12 + 0 * 36 + 24)
                    do
                        local v = buffer.readvectorf32x3(w, pw + (0 * 4 + 0) * 12)
                        local c = vector.floor(v)
                        local m = v * 256
                        local f = vector.floor(m)
                        local a = m - f
                        acc0_0 = acc0_0 + ((a * xa0 + f * xb0) + c * xc0)
                        acc0_1 = acc0_1 + ((a * xa1 + f * xb1) + c * xc1)
                    end
                    do
                        local v = buffer.readvectorf32x3(w, pw + (0 * 4 + 1) * 12)
                        local c = vector.floor(v)
                        local m = v * 256
                        local f = vector.floor(m)
                        local a = m - f
                        acc1_0 = acc1_0 + ((a * xa0 + f * xb0) + c * xc0)
                        acc1_1 = acc1_1 + ((a * xa1 + f * xb1) + c * xc1)
                    end
                    do
                        local v = buffer.readvectorf32x3(w, pw + (0 * 4 + 2) * 12)
                        local c = vector.floor(v)
                        local m = v * 256
                        local f = vector.floor(m)
                        local a = m - f
                        acc2_0 = acc2_0 + ((a * xa0 + f * xb0) + c * xc0)
                        acc2_1 = acc2_1 + ((a * xa1 + f * xb1) + c * xc1)
                    end
                    do
                        local v = buffer.readvectorf32x3(w, pw + (0 * 4 + 3) * 12)
                        local c = vector.floor(v)
                        local m = v * 256
                        local f = vector.floor(m)
                        local a = m - f
                        acc3_0 = acc3_0 + ((a * xa0 + f * xb0) + c * xc0)
                        acc3_1 = acc3_1 + ((a * xa1 + f * xb1) + c * xc1)
                    end
                end
                local xg0 = buffer.readvectorf32x3(xsb, xd0 + g * 12)
                local xg1 = buffer.readvectorf32x3(xsb, xd1 + g * 12)
                local sv0 = buffer.readvectorf32x3(wsb, sr + 0)
                tv0_0 = tv0_0 + acc0_0 * (sv0 * xg0)
                tv0_1 = tv0_1 + acc0_1 * (sv0 * xg1)
                local sv1 = buffer.readvectorf32x3(wsb, sr + 12)
                tv1_0 = tv1_0 + acc1_0 * (sv1 * xg0)
                tv1_1 = tv1_1 + acc1_1 * (sv1 * xg1)
                local sv2 = buffer.readvectorf32x3(wsb, sr + 24)
                tv2_0 = tv2_0 + acc2_0 * (sv2 * xg0)
                tv2_1 = tv2_1 + acc2_1 * (sv2 * xg1)
                local sv3 = buffer.readvectorf32x3(wsb, sr + 36)
                tv3_0 = tv3_0 + acc3_0 * (sv3 * xg0)
                tv3_1 = tv3_1 + acc3_1 * (sv3 * xg1)
            end
            buffer.writevectorf32x3(y, yBase + (t * yStride + i + 0) * 4, tv0_0)
            buffer.writevectorf32x3(y, yBase + (t * yStride + i + 3) * 4, tv1_0)
            buffer.writevectorf32x3(y, yBase + (t * yStride + i + 6) * 4, tv2_0)
            buffer.writevectorf32x3(y, yBase + (t * yStride + i + 9) * 4, tv3_0)
            buffer.writevectorf32x3(y, yBase + ((t + 1) * yStride + i + 0) * 4, tv0_1)
            buffer.writevectorf32x3(y, yBase + ((t + 1) * yStride + i + 3) * 4, tv1_1)
            buffer.writevectorf32x3(y, yBase + ((t + 1) * yStride + i + 6) * 4, tv2_1)
            buffer.writevectorf32x3(y, yBase + ((t + 1) * yStride + i + 9) * 4, tv3_1)
        end
    end
end
-- Shared data fill and timing of the bisect modules and the Script copy of the S2 tile (concatenated by gen_bisect.py).
local ROWS, K = 1536, 576
local NG = K // 32
local TILES = ROWS // 12

local function fillAll(T)
    local state = 12345
    local function rnd(n)
        state = (state * 16807) % 2147483647
        return (state // 64) % n
    end
    local w = buffer.create(TILES * NG * 528)
    for at = 0, buffer.len(w) - 4, 4 do
        buffer.writef32(w, at, (rnd(256) + 256 * rnd(256) + 65536 * rnd(256)) / 65536)
    end
    local ws = buffer.create(TILES * NG * 48)
    for at = 0, buffer.len(ws) - 4, 4 do
        buffer.writef32(ws, at, (rnd(1000) + 1) / 100000)
    end
    local x = buffer.create(T * NG * 408)
    for at = 0, buffer.len(x) - 4, 4 do
        buffer.writef32(x, at, (rnd(255) - 127))
    end
    local xs = buffer.create(T * NG * 12)
    for at = 0, buffer.len(xs) - 4, 4 do
        buffer.writef32(xs, at, (rnd(1000) + 1) / 1000)
    end
    return w, ws, x, xs
end

--- Seconds per call of the tile over `window` seconds, after warm up calls.
local function timeTile(tile, T, window)
    local w, ws, x, xs = fillAll(T)
    local y = buffer.create(T * ROWS * 4)
    tile(w, 0, ws, 0, x, xs, y, 0, ROWS, 0, ROWS, K, T)
    tile(w, 0, ws, 0, x, xs, y, 0, ROWS, 0, ROWS, K, T)
    local calls = 0
    local start = os.clock()
    local now = start
    repeat
        tile(w, 0, ws, 0, x, xs, y, 0, ROWS, 0, ROWS, K, T)
        calls += 1
        now = os.clock()
    until now - start >= window
    return (now - start) / calls
end

return { tile = tile, tokens = 2, time = timeTile, rows = ROWS, k = K }
