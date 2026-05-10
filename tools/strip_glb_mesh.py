#!/usr/bin/env python3
"""Strip mesh, material, texture, and image data from GLB files.

Keeps only animation curves, skeleton nodes, and skin inverse-bind matrices.
The first argument (--keep) specifies GLBs that should NOT be stripped (base mesh).

Usage:
  python3 strip_glb_mesh.py assets/models/character/ --keep idle.glb
  python3 strip_glb_mesh.py file1.glb file2.glb --keep idle.glb
"""

import argparse
import json
import os
import struct
import sys

GLB_MAGIC = 0x46546C67
GLB_VERSION = 2
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942


def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != GLB_VERSION:
        raise ValueError(f"Not a valid GLB v2: {path}")

    off = 12
    chunks = []
    while off < len(data):
        cl, ct = struct.unpack_from("<II", data, off)
        chunks.append((ct, data[off + 8 : off + 8 + cl]))
        off += 8 + cl
    return chunks


def write_glb(path, json_bytes, bin_bytes):
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_pad
    bin_pad = (4 - len(bin_bytes) % 4) % 4
    bin_bytes += b"\x00" * bin_pad

    total = 12 + 8 + len(json_bytes)
    if bin_bytes:
        total += 8 + len(bin_bytes)

    with open(path, "wb") as f:
        f.write(struct.pack("<III", GLB_MAGIC, GLB_VERSION, total))
        f.write(struct.pack("<II", len(json_bytes), CHUNK_JSON))
        f.write(json_bytes)
        if bin_bytes:
            f.write(struct.pack("<II", len(bin_bytes), CHUNK_BIN))
            f.write(bin_bytes)


def strip_glb(path, dry_run=False):
    chunks = read_glb(path)
    gltf = json.loads(chunks[0][1])
    old_bin = chunks[1][1] if len(chunks) > 1 else b""

    if "animations" not in gltf or not gltf["animations"]:
        return 0, 0

    accessors = gltf.get("accessors", [])
    buffer_views = gltf.get("bufferViews", [])

    needed_accessors = set()
    for anim in gltf.get("animations", []):
        for samp in anim.get("samplers", []):
            needed_accessors.add(samp["input"])
            needed_accessors.add(samp["output"])
    for skin in gltf.get("skins", []):
        if "inverseBindMatrices" in skin:
            needed_accessors.add(skin["inverseBindMatrices"])

    needed_bv = set()
    for ai in needed_accessors:
        if ai < len(accessors) and "bufferView" in accessors[ai]:
            needed_bv.add(accessors[ai]["bufferView"])

    new_bin = bytearray()
    bv_remap = {}
    for old_idx in sorted(needed_bv):
        bv = buffer_views[old_idx]
        bv_start = bv.get("byteOffset", 0)
        bv_len = bv["byteLength"]
        align = (4 - len(new_bin) % 4) % 4
        new_bin += b"\x00" * align
        new_offset = len(new_bin)
        new_bin += old_bin[bv_start : bv_start + bv_len]
        bv_remap[old_idx] = (len(bv_remap), new_offset)

    new_buffer_views = []
    for old_idx in sorted(needed_bv):
        bv = dict(buffer_views[old_idx])
        new_idx, new_offset = bv_remap[old_idx]
        bv["byteOffset"] = new_offset
        bv["buffer"] = 0
        new_buffer_views.append(bv)

    acc_remap = {}
    new_accessors = []
    for old_ai in sorted(needed_accessors):
        acc = dict(accessors[old_ai])
        if "bufferView" in acc and acc["bufferView"] in bv_remap:
            acc["bufferView"] = bv_remap[acc["bufferView"]][0]
        acc_remap[old_ai] = len(new_accessors)
        new_accessors.append(acc)

    for anim in gltf.get("animations", []):
        for samp in anim.get("samplers", []):
            samp["input"] = acc_remap[samp["input"]]
            samp["output"] = acc_remap[samp["output"]]
    for skin in gltf.get("skins", []):
        if "inverseBindMatrices" in skin:
            skin["inverseBindMatrices"] = acc_remap[skin["inverseBindMatrices"]]

    gltf["accessors"] = new_accessors
    gltf["bufferViews"] = new_buffer_views
    gltf["buffers"] = [{"byteLength": len(new_bin)}] if new_bin else []

    for key in ("meshes", "materials", "textures", "images", "samplers"):
        gltf.pop(key, None)

    for node in gltf.get("nodes", []):
        node.pop("mesh", None)

    old_size = os.path.getsize(path)
    if not dry_run:
        json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        write_glb(path, json_bytes, bytes(new_bin))
    new_size = os.path.getsize(path) if not dry_run else 0

    return old_size, new_size


def main():
    parser = argparse.ArgumentParser(description="Strip mesh/texture from animation GLBs")
    parser.add_argument("paths", nargs="+", help="GLB files or directories to process")
    parser.add_argument("--keep", nargs="*", default=["idle.glb"],
                        help="Filenames to skip (base mesh files)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done")
    args = parser.parse_args()

    keep_set = {k.lower() for k in args.keep}
    files = []
    for p in args.paths:
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                if fn.lower().endswith(".glb"):
                    files.append(os.path.join(p, fn))
        elif os.path.isfile(p):
            files.append(p)

    total_saved = 0
    for fp in files:
        base = os.path.basename(fp).lower()
        if base in keep_set:
            print(f"  SKIP (keep) {fp}")
            continue
        try:
            old_sz, new_sz = strip_glb(fp, dry_run=args.dry_run)
            if old_sz == 0:
                print(f"  SKIP (no anims) {fp}")
                continue
            saved = old_sz - new_sz if not args.dry_run else 0
            total_saved += saved
            if args.dry_run:
                print(f"  WOULD STRIP {fp}  ({old_sz:,} bytes)")
            else:
                print(f"  {fp}  {old_sz:,} -> {new_sz:,}  (saved {saved:,})")
        except Exception as e:
            print(f"  ERROR {fp}: {e}", file=sys.stderr)

    if not args.dry_run:
        print(f"\nTotal saved: {total_saved:,} bytes ({total_saved / 1048576:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
