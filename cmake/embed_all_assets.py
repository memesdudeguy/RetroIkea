#!/usr/bin/env python3
"""Write vulkan_embedded_assets.h in one shot (tab-separated lines: SYMBOL<TAB>path)."""
import os
import pathlib
import re
import sys


def emit_array(out, symbol: str, data: bytes) -> None:
    out.write(f"static const unsigned char {symbol}[] = {{\n")
    for i, b in enumerate(data):
        out.write(f"0x{b:02x},")
        if (i % 16) == 15:
            out.write("\n")
    out.write("\n};\n")
    out.write(f"static constexpr std::size_t {symbol}_size = sizeof({symbol});\n\n")


def _validate_header(path: pathlib.Path) -> None:
    text = path.read_text(encoding="ascii")
    for m in re.finditer(
        r"static constexpr std::size_t (\w+)_size = sizeof\(\1\);", text
    ):
        last = m
    try:
        end = last.end()
    except NameError as exc:
        raise SystemExit(f"no size lines in {path}") from exc
    rest = text[end:].strip("\n")
    if rest.strip():
        raise SystemExit(
            f"generated header has trailing garbage after last array: {rest[:120]!r}..."
        )


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: embed_all_assets.py OUT_HEADER MANIFEST_TXT", file=sys.stderr)
        sys.exit(2)
    out_path = pathlib.Path(sys.argv[1])
    manifest = pathlib.Path(sys.argv[2])
    lines = manifest.read_text(encoding="utf-8").splitlines()
    tmp_path = out_path.parent / (out_path.name + ".tmp")
    with tmp_path.open("w", encoding="ascii") as out:
        out.write("#pragma once\n#include <cstddef>\n\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tab = line.find("\t")
            if tab < 0:
                print(f"bad manifest line (no tab): {line!r}", file=sys.stderr)
                sys.exit(1)
            sym = line[:tab].strip()
            path = line[tab + 1 :].strip()
            if not sym or not path:
                print(f"bad manifest line: {line!r}", file=sys.stderr)
                sys.exit(1)
            p = pathlib.Path(path)
            if not p.is_file():
                print(f"missing asset file: {p}", file=sys.stderr)
                sys.exit(1)
            emit_array(out, sym, p.read_bytes())
    _validate_header(tmp_path)
    os.replace(tmp_path, out_path)


if __name__ == "__main__":
    main()
