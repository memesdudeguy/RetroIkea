#!/usr/bin/env bash
# Export a .blend to .glb using Flatpak Blender (no GUI).
# Usage: blender_export_glb.sh INPUT.blend OUTPUT.glb

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 INPUT.blend OUTPUT.glb" >&2
  exit 1
fi

BLEND="$(realpath "$1")"
OUT="$(realpath -m "$2")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/export_blend_to_glb.py"

if [[ ! -f "$BLEND" ]]; then
  echo "error: blend not found: $BLEND" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

_FLATPAK_ENV=()
[[ -n "${RETROIKEA_GLTF_JPEG_QUALITY:-}" ]] &&
  _FLATPAK_ENV+=(--env="RETROIKEA_GLTF_JPEG_QUALITY=${RETROIKEA_GLTF_JPEG_QUALITY}")
[[ -n "${RETROIKEA_GLTF_LOSSLESS:-}" ]] &&
  _FLATPAK_ENV+=(--env="RETROIKEA_GLTF_LOSSLESS=${RETROIKEA_GLTF_LOSSLESS}")
exec flatpak run "${_FLATPAK_ENV[@]}" org.blender.Blender --background "$BLEND" --python "$PY" -- "$OUT"
