"""Writer for the `.olelo` container, see docs/container.md for the format."""

import json
import struct

MAGIC = b"OLELO\0\0\0"
VERSION = 1
ALIGN = 64
MAX_PART_BYTES = 1 << 30
Q8_GROUP = 32
Q8_ROW_TILE = 8
Q4_ROW_TILE = 8
V12_ROW_TILE = 12
Q8_ROW_TILES = {"p8": Q8_ROW_TILE, "v12": V12_ROW_TILE}


def quantize_q8(array, rows_padded, kpad):
    """Q8_0 as llama.cpp computes it: per group of 32 a scale d = amax / 127 stored in half precision, and weights
    round(x / d) taken with the unrounded scale. Returns the weights as (rows_padded, kpad) int and the scales as
    (rows_padded, kpad / 32) float32. Padding rows and columns are zero weights with scale 0."""
    import numpy as np

    rows, cols = array.shape
    x = np.zeros((rows_padded, kpad), dtype=np.float32)
    x[:rows, :cols] = array
    groups = x.reshape(rows_padded, kpad // Q8_GROUP, Q8_GROUP)
    amax = np.abs(groups).max(axis=2)
    d = (amax / np.float32(127)).astype(np.float32)
    with np.errstate(divide="ignore"):
        inverse = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    scaled = groups * inverse[:, :, None]
    magnitude = np.abs(scaled)
    base = np.floor(magnitude)
    base += magnitude - base >= 0.5
    q = (np.sign(scaled) * base).astype(np.int16).reshape(rows_padded, kpad)
    return q, d.astype(np.float16).astype(np.float32)


def quantize_q4(array, rows_padded, kpad):
    """Q4_0 as llama.cpp computes it: per group of 32 the scale is the value of largest magnitude divided by -8, stored
    in half precision, and a weight is min(15, trunc(x / d + 8.5)) taken with the unrounded scale. Returns the nibbles as
    bytes of (rows_padded, kpad / 2), element j of a group in the low nibble of byte j and element j + 16 in the high
    nibble, and the scales as (rows_padded, kpad / 32) float32. Padding is the nibble 8 with scale 0."""
    import numpy as np

    rows, cols = array.shape
    x = np.zeros((rows_padded, kpad), dtype=np.float32)
    x[:rows, :cols] = array
    groups = x.reshape(rows_padded, kpad // Q8_GROUP, Q8_GROUP)
    at = np.abs(groups).argmax(axis=2)
    largest = np.take_along_axis(groups, at[:, :, None], axis=2)[:, :, 0]
    d = (largest / np.float32(-8)).astype(np.float32)
    with np.errstate(divide="ignore"):
        inverse = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    scaled = (groups * inverse[:, :, None] + np.float32(8.5)).astype(np.float32)
    nibbles = np.minimum(15, scaled.astype(np.int8)).astype(np.uint8)
    packed = nibbles[:, :, :16] | (nibbles[:, :, 16:] << 4)
    return packed.reshape(rows_padded, kpad // 2), d.astype(np.float16).astype(np.float32)


def unpack_rowmajor(data, rows, kpad):
    """The nibbles of `rows` row major Q4_0 rows as an array of (rows, kpad): byte j of a group holds element j in its low
    nibble and element j + 16 in its high nibble."""
    import numpy as np

    groups = np.frombuffer(data, dtype=np.uint8).reshape(rows, kpad // Q8_GROUP, 16)
    return np.concatenate([groups & 15, groups >> 4], axis=2).reshape(rows, kpad)


def pack_p4(nibbles):
    """The bytes of the `p4` layout for nibbles of (rows, kpad), rows a multiple of 8: per tile of 8 rows and group of 32
    columns, a u32 per column. Pair m of rows holds row 2m in nibble m and row 2m + 1 in nibble m + 4."""
    import numpy as np

    rows, kpad = nibbles.shape
    tiles = rows // Q4_ROW_TILE
    view = nibbles.reshape(tiles, Q4_ROW_TILE, kpad // Q8_GROUP, Q8_GROUP).astype(np.uint32)
    words = np.zeros((tiles, kpad // Q8_GROUP, Q8_GROUP), dtype=np.uint32)
    for row in range(Q4_ROW_TILE):
        position = (row % 2) * 4 + row // 2
        words |= view[:, row] << np.uint32(4 * position)
    return np.ascontiguousarray(words, dtype="<u4").tobytes()


def pack_scales_tiled(scales, tile=Q4_ROW_TILE):
    """The scales of (rows, groups) with the `tile` rows of a tile together per group, as the `p8`, `p4` and `v12`
    layouts keep them."""
    import numpy as np

    rows, groups = scales.shape
    tiles = rows // tile
    return np.ascontiguousarray(scales.reshape(tiles, tile, groups).transpose(0, 2, 1), dtype="<f4").tobytes()


def pack_v12(q):
    """The bytes of the `v12` layout for the Q8 weights q of (rows, kpad) in -128 to 127, rows a multiple of 12. Per tile
    of 12 rows and group of 32 columns there are 11 steps, each 4 row vectors of 3 rows, and a lane holds the offset
    binary bytes a, b, c of the columns j, j + 11, j + 22 as the f32 (a + 256 b + 65536 c) / 65536. Column 32 does not
    exist, its byte is the offset 128 (the weight 0)."""
    import numpy as np

    rows, kpad = q.shape
    tiles = rows // V12_ROW_TILE
    groups = kpad // Q8_GROUP
    byte = (q.astype(np.int32) + 128).reshape(tiles, 4, 3, groups, Q8_GROUP)
    byte = np.concatenate([byte, np.full((tiles, 4, 3, groups, 1), 128, dtype=np.int32)], axis=4)
    a, b, c = byte[..., 0:11], byte[..., 11:22], byte[..., 22:33]
    lanes = (a + 256 * b + 65536 * c).astype(np.float64) / 65536
    # (tiles, rv, l, groups, steps) to (tiles, groups, steps, rv, l)
    lanes = lanes.transpose(0, 3, 4, 1, 2)
    return np.ascontiguousarray(lanes, dtype="<f4").tobytes()


def dequantize(array, recipe):
    """The float32 values a matrix has after storing it in `recipe` (bf16, q8 or q4) and reading it back, for reference
    runs that isolate the rounding of the weights from everything else the runtime does."""
    import numpy as np

    rows, cols = array.shape
    kpad = round_up(cols, Q8_GROUP)
    rows_padded = round_up(rows, 8)
    if recipe == "bf16":
        bits = bf16_bits(array).astype("<u4") << 16
        return bits.view("<f4").astype(np.float32)
    if recipe == "q8":
        q, scales = quantize_q8(array, rows_padded, kpad)
        values = q.reshape(rows_padded, kpad // Q8_GROUP, Q8_GROUP) * scales[:, :, None]
    else:
        packed, scales = quantize_q4(array, rows_padded, kpad)
        groups = packed.reshape(rows_padded, kpad // Q8_GROUP, 16)
        q = np.concatenate([(groups & 15).astype(np.int32) - 8, (groups >> 4).astype(np.int32) - 8], axis=2)
        values = q * scales[:, :, None]
    return values.reshape(rows_padded, kpad)[:rows, :cols].astype(np.float32)


def bf16_bits(array):
    """The BF16 bits of float32 values, rounded to nearest even."""
    import numpy as np

    bits = np.ascontiguousarray(array, dtype="<f4").view("<u4")
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) >> 16).astype("<u2")


def round_up(n, step):
    return (n + step - 1) // step * step


class Writer:
    """Collects tensors and blobs, then writes the file. Offsets in the header are relative to the data area."""

    def __init__(self, header):
        self.header = dict(header)
        self.tensors = []
        self.blobs = {}
        self.chunks = []
        self.size = 0

    def _add(self, data):
        pad = round_up(self.size, ALIGN) - self.size
        if pad:
            self.chunks.append(b"\0" * pad)
            self.size += pad
        offset = self.size
        self.chunks.append(data)
        self.size += len(data)
        return offset

    def add_blob(self, name, data):
        self.blobs[name] = {"offset": self._add(data), "bytes": len(data)}

    def add_matrix(self, name, array, fmt, row_tile, kstep, max_part_bytes=MAX_PART_BYTES):
        """Stores a float32 matrix of shape (rows, cols), padded to row_tile rows and kstep columns, as f32 or bf16
        parts."""
        import numpy as np

        rows, cols = array.shape
        kpad = round_up(cols, kstep)
        rows_padded = round_up(rows, row_tile)
        row_bytes = kpad * (2 if fmt == "bf16" else 4)
        per_part = (max_part_bytes // row_bytes) // row_tile * row_tile
        if per_part < row_tile:
            raise ValueError(f"a tile of rows of {name} does not fit in one buffer")
        padded = np.zeros((rows_padded, kpad), dtype="<f4")
        padded[:rows, :cols] = array
        if fmt == "bf16":
            padded = bf16_bits(padded)
        parts = []
        start = 0
        while start < rows_padded:
            count = min(per_part, rows_padded - start)
            data = padded[start : start + count].tobytes()
            parts.append({"offset": self._add(data), "rowStart": start, "rows": count, "bytes": len(data)})
            start += count
        self.tensors.append(
            {
                "name": name,
                "format": fmt,
                "layout": "rowmajor",
                "rows": rows,
                "cols": cols,
                "kpad": kpad,
                "rowTile": row_tile,
                "parts": parts,
            }
        )

    def add_matrix_q8(self, name, array, max_part_bytes=MAX_PART_BYTES, layout="p8"):
        """Stores a float32 matrix as Q8_0 in the `p8` layout: tiles of 8 rows, one tile column of 8 offset binary
        bytes (weight + 128) per column, and the 8 scales of a tile row together per group of 32 columns. The `v12`
        layout is `pack_v12`."""
        rows, cols = array.shape
        kpad = round_up(cols, Q8_GROUP)
        rows_padded = round_up(rows, Q8_ROW_TILES[layout])
        q, scales = quantize_q8(array, rows_padded, kpad)
        self.add_q8_weights(name, rows, cols, q, scales, layout, max_part_bytes)

    def add_q8_weights(self, name, rows, cols, q, scales, layout, max_part_bytes=MAX_PART_BYTES):
        """Stores quantized Q8 weights q of (rows_padded, kpad) and their scales of (rows_padded, groups) in `layout`."""
        import numpy as np

        tile = Q8_ROW_TILES[layout]
        rows_padded, kpad = q.shape
        assert rows_padded % tile == 0
        row_bytes = kpad if layout == "p8" else kpad // Q8_GROUP * 44
        per_part = (max_part_bytes // row_bytes) // tile * tile
        if per_part < tile:
            raise ValueError(f"a tile of rows of {name} does not fit in one buffer")
        parts = []
        start = 0
        while start < rows_padded:
            count = min(per_part, rows_padded - start)
            block = q[start : start + count]
            if layout == "p8":
                data = np.ascontiguousarray(
                    (block + 128).astype(np.uint8).reshape(count // tile, tile, kpad).transpose(0, 2, 1)
                ).tobytes()
            else:
                data = pack_v12(block)
            scale_data = pack_scales_tiled(scales[start : start + count].astype(np.float32), tile)
            parts.append(
                {
                    "offset": self._add(data),
                    "rowStart": start,
                    "rows": count,
                    "bytes": len(data),
                    "scaleOffset": self._add(scale_data),
                    "scaleBytes": len(scale_data),
                }
            )
            start += count
        self.tensors.append(
            {
                "name": name,
                "format": "q8",
                "layout": layout,
                "rows": rows,
                "cols": cols,
                "kpad": kpad,
                "rowTile": tile,
                "parts": parts,
            }
        )

    def add_matrix_q4(self, name, array, max_part_bytes=MAX_PART_BYTES):
        """Stores a float32 matrix as Q4_0 in the `p4` layout: tiles of 8 rows, a u32 per column holding the 8 rows as
        nibbles, and the 8 scales of a tile row together per group of 32 columns."""
        import numpy as np

        rows, cols = array.shape
        kpad = round_up(cols, Q8_GROUP)
        rows_padded = round_up(rows, Q4_ROW_TILE)
        per_part = (max_part_bytes // (kpad // 2)) // Q4_ROW_TILE * Q4_ROW_TILE
        if per_part < Q4_ROW_TILE:
            raise ValueError(f"a tile of rows of {name} does not fit in one buffer")
        rowmajor, scales = quantize_q4(array, rows_padded, kpad)
        nibbles = unpack_rowmajor(rowmajor, rows_padded, kpad)
        parts = []
        start = 0
        while start < rows_padded:
            count = min(per_part, rows_padded - start)
            data = pack_p4(nibbles[start : start + count])
            scale_data = pack_scales_tiled(scales[start : start + count])
            parts.append(
                {
                    "offset": self._add(data),
                    "rowStart": start,
                    "rows": count,
                    "bytes": len(data),
                    "scaleOffset": self._add(scale_data),
                    "scaleBytes": len(scale_data),
                }
            )
            start += count
        self.tensors.append(
            {
                "name": name,
                "format": "q4",
                "layout": "p4",
                "rows": rows,
                "cols": cols,
                "kpad": kpad,
                "rowTile": Q4_ROW_TILE,
                "parts": parts,
            }
        )

    def add_vector(self, name, array):
        """Stores a float32 vector, unpadded, as one part."""
        import numpy as np

        data = np.ascontiguousarray(array, dtype="<f4").tobytes()
        self.tensors.append(
            {
                "name": name,
                "format": "f32",
                "layout": "vector",
                "rows": 1,
                "cols": len(array),
                "kpad": len(array),
                "rowTile": 1,
                "parts": [{"offset": self._add(data), "rowStart": 0, "rows": 1, "bytes": len(data)}],
            }
        )

    def write(self, path):
        header = dict(self.header)
        header["container"] = VERSION
        header["tensors"] = self.tensors
        header["blobs"] = self.blobs
        text = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        data_start = round_up(16 + len(text), ALIGN)
        with open(path, "wb") as out:
            out.write(MAGIC)
            out.write(struct.pack("<II", VERSION, len(text)))
            out.write(text)
            out.write(b"\0" * (data_start - 16 - len(text)))
            for chunk in self.chunks:
                out.write(chunk)
        return data_start + self.size
