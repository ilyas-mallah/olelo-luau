"""Writes a copy of a container with its Q8 matrices in the vector `v12` layout.

    repack_q8v.py <in.olelo> <out.olelo>

The weights and scales are the same, only their order changes, and the data is 1.375 times as large, so the copy is a
new file. Other tensors and the blobs are copied unchanged. A container with no `p8` Q8 matrix is copied as it is.
"""

import json
import struct
import sys

import numpy as np

from container import ALIGN, MAGIC, Q8_GROUP, VERSION, Writer, round_up


def read_container(f):
    head = f.read(16)
    if head[:8] != MAGIC:
        raise SystemExit("not a container")
    header_len = struct.unpack("<I", head[12:16])[0]
    header = json.loads(f.read(header_len))
    return header, round_up(16 + header_len, ALIGN)


def unpack_p8(data, rows, kpad):
    """The Q8 weights of (rows, kpad) in -128 to 127 from the bytes of the `p8` layout."""
    tiles = rows // 8
    bytes_ = np.frombuffer(data, dtype=np.uint8).reshape(tiles, kpad, 8).transpose(0, 2, 1)
    return bytes_.reshape(rows, kpad).astype(np.int32) - 128


def unpack_scales(data, rows, kpad):
    """The scales of (rows, groups) from the tiled order of the `p8` layout."""
    groups = kpad // Q8_GROUP
    tiles = rows // 8
    return np.frombuffer(data, dtype="<f4").reshape(tiles, groups, 8).transpose(0, 2, 1).reshape(rows, groups)


def repack(src, dst):
    with open(src, "rb") as f:
        header, start = read_container(f)
        writer = Writer({k: v for k, v in header.items() if k not in ("tensors", "blobs", "container")})
        changed = 0
        for t in header["tensors"]:
            if t["format"] == "q8" and t["layout"] == "p8":
                weights, scales = [], []
                for p in t["parts"]:
                    f.seek(start + p["offset"])
                    weights.append(unpack_p8(f.read(p["bytes"]), p["rows"], t["kpad"]))
                    f.seek(start + p["scaleOffset"])
                    scales.append(unpack_scales(f.read(p["scaleBytes"]), p["rows"], t["kpad"]))
                q = np.concatenate(weights)
                sc = np.concatenate(scales)
                rows_padded = round_up(t["rows"], 12)
                q, sc = q[:rows_padded], sc[:rows_padded]
                if rows_padded > q.shape[0]:
                    extra = rows_padded - q.shape[0]
                    q = np.concatenate([q, np.zeros((extra, q.shape[1]), dtype=q.dtype)])
                    sc = np.concatenate([sc, np.zeros((extra, sc.shape[1]), dtype=sc.dtype)])
                writer.add_q8_weights(t["name"], t["rows"], t["cols"], q, sc, "v12")
                changed += 1
            else:
                parts = []
                for p in t["parts"]:
                    entry = {k: v for k, v in p.items() if k not in ("offset", "scaleOffset")}
                    f.seek(start + p["offset"])
                    entry["offset"] = writer._add(f.read(p["bytes"]))
                    if "scaleOffset" in p:
                        f.seek(start + p["scaleOffset"])
                        entry["scaleOffset"] = writer._add(f.read(p["scaleBytes"]))
                    parts.append(entry)
                writer.tensors.append({**t, "parts": parts})
        for name, b in header["blobs"].items():
            f.seek(start + b["offset"])
            writer.add_blob(name, f.read(b["bytes"]))
    size = writer.write(dst)
    print(f"{dst}: {changed} Q8 matrices repacked, {size} bytes")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    repack(sys.argv[1], sys.argv[2])
