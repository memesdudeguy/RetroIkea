#!/usr/bin/env bash
# Re-export Meshy ledge clips (export_apply off) and install as game asset names.
# Shimmy: left = Climb_Left_inplace, right = Climb_Right (JPEG in GLB for smaller files).
#
# Searches MESHY_DIR then MESHY_DIR/Meshy_AI_biped for each file (default MESHY_DIR=~/Downloads).
# Override: MESHY_DIR=/path RETROIKEA_GLTF_JPEG_QUALITY=52 ./install_meshy_ledge_clips_from_downloads.sh
# Sharper: RETROIKEA_GLTF_LOSSLESS=1 (ignores default JPEG quality).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
MESHY_DIR="${MESHY_DIR:-$HOME/Downloads}"

first_existing() {
  local name="$1"
  for d in "$MESHY_DIR" "$MESHY_DIR/Meshy_AI_biped"; do
    if [[ -f "$d/$name" ]]; then
      echo "$d/$name"
      return 0
    fi
  done
  return 1
}

# Try Meshy_AI_biped_* first (current Meshy download naming), then legacy Meshy_AI_Animation_*.
first_existing_any() {
  local name d
  for name in "$@"; do
    for d in "$MESHY_DIR" "$MESHY_DIR/Meshy_AI_biped"; do
      if [[ -f "$d/$name" ]]; then
        echo "$d/$name"
        return 0
      fi
    done
  done
  return 1
}

GRAB="$(first_existing_any \
  "Meshy_AI_biped_Animation_Jump_and_Grab_Wall_withSkin.glb" \
  "Meshy_AI_Animation_Jump_and_Grab_Wall_withSkin.glb")" || {
  echo "missing: Jump_and_Grab_Wall_withSkin.glb (biped or Meshy_AI_Animation) under $MESHY_DIR" >&2
  exit 1
}
LEFT="$(first_existing_any \
  "Meshy_AI_biped_Animation_Climb_Left_with_Both_Limbs_inplace_withSkin.glb" \
  "Meshy_AI_Animation_Climb_Left_with_Both_Limbs_inplace_withSkin.glb")" || {
  echo "missing: Climb_Left_*_inplace_withSkin.glb under $MESHY_DIR" >&2
  exit 1
}
RIGHT="$(first_existing_any \
  "Meshy_AI_biped_Animation_Climb_Right_with_Both_Limbs_withSkin.glb" \
  "Meshy_AI_Animation_Climb_Right_with_Both_Limbs_withSkin.glb")" || {
  echo "missing: Climb_Right_*_withSkin.glb under $MESHY_DIR" >&2
  exit 1
}

# Smaller GLBs by default; lossless skips this.
if [[ -z "${RETROIKEA_GLTF_LOSSLESS:-}" ]]; then
  export RETROIKEA_GLTF_JPEG_QUALITY="${RETROIKEA_GLTF_JPEG_QUALITY:-58}"
fi

OUT="$REPO/exported_meshy_ledge_tmp"
LIST="$REPO/.meshy_ledge_export_list.txt"
rm -rf "$OUT"
mkdir -p "$OUT"
printf '%s\n' "$GRAB" "$LEFT" "$RIGHT" >"$LIST"

_FLATPAK_ENV=()
[[ -n "${RETROIKEA_GLTF_JPEG_QUALITY:-}" ]] &&
  _FLATPAK_ENV+=(--env="RETROIKEA_GLTF_JPEG_QUALITY=${RETROIKEA_GLTF_JPEG_QUALITY}")
[[ -n "${RETROIKEA_GLTF_LOSSLESS:-}" ]] &&
  _FLATPAK_ENV+=(--env="RETROIKEA_GLTF_LOSSLESS=${RETROIKEA_GLTF_LOSSLESS}")
flatpak run "${_FLATPAK_ENV[@]}" org.blender.Blender --background --python "$REPO/scripts/batch_export_3d_to_glb.py" -- "$OUT" "$LIST"

EXT="$OUT/_external"
CHAR="$REPO/assets/models/character"
GRAB_BN=$(basename "$GRAB")
LEFT_BN=$(basename "$LEFT")
RIGHT_BN=$(basename "$RIGHT")
cp -f "$EXT/$GRAB_BN" "$CHAR/ledge_grab_wall.glb"
cp -f "$EXT/$LEFT_BN" "$CHAR/ledge_shimmy_left.glb"
cp -f "$EXT/$RIGHT_BN" "$CHAR/ledge_shimmy_right.glb"
rm -f "$LIST"
echo "installed (JPEG q=${RETROIKEA_GLTF_JPEG_QUALITY:-auto}) -> $CHAR/ledge_{grab_wall,shimmy_left,shimmy_right}.glb" >&2
du -h "$CHAR/ledge_grab_wall.glb" "$CHAR/ledge_shimmy_left.glb" "$CHAR/ledge_shimmy_right.glb" >&2
