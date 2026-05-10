#!/usr/bin/env bash
# Copy Blender batch output into assets/ so the next CMake build uses smaller GLBs in-repo.
# Run after: ./scripts/export_all_project_3d.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP="${EXPORT_DIR:-$REPO/exported_glb_blender}"

if [[ ! -d "$EXP/assets/models" ]]; then
  echo "error: missing $EXP/assets/models — run scripts/export_all_project_3d.sh first" >&2
  exit 1
fi

echo "sync models -> $REPO/assets/models" >&2
# Do not --delete: keep character_base.fbx and any files not present in the export tree.
rsync -a "$EXP/assets/models/" "$REPO/assets/models/"
if [[ -f "$EXP/assets/models/character_base.glb" ]]; then
  install -D -m0644 "$EXP/assets/models/character_base.glb" "$REPO/assets/models/character_base.glb"
fi

MESHY_OUT="$REPO/assets/meshy_reexport"
mkdir -p "$MESHY_OUT"
rm -f "$MESHY_OUT"/__*.glb
shopt -s nullglob
n=0
for f in "$EXP/_external"/*.glb; do
  base="$(basename "$f")"
  [[ "$base" == __* ]] && continue
  cp -f "$f" "$MESHY_OUT/"
  ((++n)) || true
done
if ((n)); then
  echo "copy $n external GLB(s) -> $MESHY_OUT" >&2
elif [[ ! -d "$EXP/_external" ]]; then
  echo "note: no $EXP/_external (optional Meshy downloads)" >&2
else
  echo "note: $EXP/_external has no stable-name *.glb (skip __* leftovers)" >&2
fi

echo "done. Reconfigure CMake (or clear cache) and rebuild." >&2
