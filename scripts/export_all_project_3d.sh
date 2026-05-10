#!/usr/bin/env bash
# Discover RetroIkea 3D assets (under assets/ + common optional CMake paths) and batch-export GLB via Flatpak Blender.
# Output: REPO/exported_glb_blender/  (mirrors paths under repo; external files under _external/)
#
# Usage:
#   ./export_all_project_3d.sh
#   OUT_DIR=/tmp/glb_out ./export_all_project_3d.sh
#
# Smaller GLBs: set RETROIKEA_GLTF_JPEG_QUALITY=55 (etc.) before this script — forwarded into Flatpak below.
# RETROIKEA_GLTF_LOSSLESS=1 keeps textures closer to source (larger files).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="${OUT_DIR:-$REPO/exported_glb_blender}"
# List must live under REPO (or another host path Flatpak can read); Flatpak's /tmp != host /tmp.
LIST="$(mktemp "${REPO}/.export_3d_list.XXXXXX")"
PY="$SCRIPT_DIR/batch_export_3d_to_glb.py"
cleanup() { rm -f "$LIST"; }
trap cleanup EXIT

: >"$LIST"

# Everything under assets (game-shipped tree)
if [[ -d "$REPO/assets" ]]; then
  find "$REPO/assets" -type f \( \
    -iname '*.glb' -o -iname '*.gltf' -o -iname '*.fbx' -o -iname '*.obj' \
  \) 2>/dev/null | sort -u >>"$LIST" || true
fi

# Optional paths from CMakeLists (first existing wins per logical name; still export each file that exists)
add_if_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    realpath "$f" >>"$LIST"
  fi
  return 0
}

# Character clips + fallback (same dir)
CHAR="$REPO/assets/models/character"
for f in \
  idle.glb sprint.glb walk.glb slide_right.glb \
  crouch_walk_left.glb cautious_crouch_forward.glb cautious_crouch_backward.glb \
  crouch_walk_right.glb slide_light.glb step_push.glb crouch_idle_bow.glb \
  ledge_ladder_climb.glb ledge_grab_wall.glb ledge_shimmy_left.glb ledge_shimmy_right.glb \
  jump_regular.glb jump_run.glb punch_combo.glb fall_knockdown.glb stand_up.glb \
  land_hit_reaction.glb hair_shove_reaction.glb proximity_dance.glb character_base.fbx; do
  add_if_file "$CHAR/$f"
done

# External / Downloads-style (Meshy, etc.)
[[ -n "${HOME:-}" ]] || true
if [[ -n "${HOME:-}" ]]; then
  add_if_file "$HOME/Downloads/Meshy_AI_biped/Meshy_AI_biped_Animation_Slow_Ladder_Climb_withSkin.glb"
  add_if_file "$HOME/Downloads/Meshy_AI_Animation_Hip_Hop_Dance_3_withSkin.glb"
  add_if_file "$HOME/Downloads/Meshy_AI_biped/Meshy_AI_biped_Animation_Hip_Hop_Dance_3_withSkin.glb"
fi

add_if_file "$REPO/../Meshy_AI_biped/Meshy_AI_biped_Animation_Slow_Ladder_Climb_withSkin.glb"
add_if_file "$REPO/../Meshy_AI_Animation_Hip_Hop_Dance_3_withSkin.glb"
add_if_file "$REPO/../Downloads/Meshy_AI_biped/Meshy_AI_biped_Animation_Hip_Hop_Dance_3_withSkin.glb"

# Deduplicate paths
sort -u "$LIST" -o "$LIST"

N="$(wc -l <"$LIST" | tr -d ' ')"
if [[ "$N" -eq 0 ]]; then
  echo "No 3D files found. Add meshes under $REPO/assets or place Meshy GLBs in ~/Downloads." >&2
  exit 1
fi

echo "Exporting $N file(s) -> $OUT" >&2
mkdir -p "$OUT"

# No exec: EXIT trap removes LIST after Blender exits.
_FLATPAK_ENV=()
[[ -n "${RETROIKEA_GLTF_JPEG_QUALITY:-}" ]] &&
  _FLATPAK_ENV+=(--env="RETROIKEA_GLTF_JPEG_QUALITY=${RETROIKEA_GLTF_JPEG_QUALITY}")
[[ -n "${RETROIKEA_GLTF_LOSSLESS:-}" ]] &&
  _FLATPAK_ENV+=(--env="RETROIKEA_GLTF_LOSSLESS=${RETROIKEA_GLTF_LOSSLESS}")
flatpak run "${_FLATPAK_ENV[@]}" org.blender.Blender --background --python "$PY" -- "$OUT" "$LIST" "$REPO"
