"""Rewrites the Q4 matrices of a container from the row major layout to the packed `p4` layout, in place.

    repack_q4.py <container.olelo> [<container.olelo> ...]

The weights and scales of a part keep their size, so the data area does not move. The header changes the layout name
and is padded with spaces to its old length. A container with no row major Q4 matrix is left alone.
"""

import json
import struct
import sys

from container import Q8_GROUP, pack_p4, pack_scales_tiled, round_up, unpack_rowmajor

ALIGN = 64


def repack(path):
    import numpy as np

    with open(path, "r+b") as f:
        head = f.read(16)
        if head[:8] != b"OLELO\0\0\0":
            raise SystemExit(f"{path}: not a container")
        header_len = struct.unpack("<I", head[12:16])[0]
        text = f.read(header_len)
        header = json.loads(text)
        start = round_up(16 + header_len, ALIGN)
        changed = 0
        for t in header["tensors"]:
            if t["format"] != "q4" or t["layout"] != "rowmajor":
                continue
            kpad = t["kpad"]
            for p in t["parts"]:
                f.seek(start + p["offset"])
                weights = f.read(p["bytes"])
                f.seek(start + p["scaleOffset"])
                scales = np.frombuffer(f.read(p["scaleBytes"]), dtype="<f4").reshape(p["rows"], kpad // Q8_GROUP)
                packed = pack_p4(unpack_rowmajor(weights, p["rows"], kpad))
                assert len(packed) == p["bytes"]
                f.seek(start + p["offset"])
                f.write(packed)
                f.seek(start + p["scaleOffset"])
                f.write(pack_scales_tiled(scales))
            t["layout"] = "p4"
            changed += 1
        if changed == 0:
            print(f"{path}: no row major Q4 matrix")
            return
        out = json.dumps(header, separators=(",", ":")).encode()
        if len(out) > header_len:
            raise SystemExit(f"{path}: the new header is longer than the old one")
        f.seek(16)
        f.write(out + b" " * (header_len - len(out)))
        print(f"{path}: repacked {changed} matrices")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for name in sys.argv[1:]:
        repack(name)
