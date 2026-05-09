#!/usr/bin/env bash
# Build packaging/RetroIkea-Beta-Setup.exe (Wine + Inno Setup 6 ISCC.exe on Linux; or "iscc packaging\windows_setup.iss" on Windows).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISCC_CANDIDATES=(
  "${INNO_ISCC:-}"
  "$HOME/.wine/drive_c/Program Files (x86)/Inno Setup 6/ISCC.exe"
  "$HOME/.wine/drive_c/Program Files/Inno Setup 6/ISCC.exe"
)
ISCC=""
for c in "${ISCC_CANDIDATES[@]}"; do
  [[ -z "$c" ]] && continue
  if [[ -f "$c" ]]; then
    ISCC="$c"
    break
  fi
done
if [[ -z "$ISCC" ]]; then
  echo "ISCC.exe not found. Set INNO_ISCC or install Inno Setup 6 under Wine." >&2
  exit 1
fi
if [[ ! -f "$ROOT/build-win-mingw/RetroIkea.exe" ]]; then
  echo "Missing build-win-mingw/RetroIkea.exe — configure and build the Windows target first." >&2
  exit 1
fi
if [[ ! -d "$ROOT/assets" ]]; then
  echo "Missing $ROOT/assets — installer must package the repo assets tree (textures, audio, models, …)." >&2
  exit 1
fi

TS_DEST="$ROOT/packaging/third_party/tailscale-setup-amd64.exe"
TS_URL="${TAILSCALE_WIN_INSTALLER_URL:-https://pkgs.tailscale.com/stable/tailscale-setup-amd64.exe}"
if [[ ! -f "$TS_DEST" ]]; then
  echo "Fetching bundled Tailscale Windows installer (optional)…"
  mkdir -p "$ROOT/packaging/third_party"
  if curl -fsSL "$TS_URL" -o "$TS_DEST.partial" && mv "$TS_DEST.partial" "$TS_DEST"; then
    ls -lh "$TS_DEST"
  else
    rm -f "$TS_DEST.partial" "$TS_DEST"
    echo "Note: Tailscale installer not bundled (download failed). Inno still builds without it." >&2
  fi
fi

RB="${ROOT//\//\\}"
# Absolute Z:\... path so ISCC resolves {#RetroIkeaRepoRoot}\assets to the real repo tree under Wine
# (same as /home/.../RetroIkea/assets when Z: is the Linux root).
wine "$ISCC" "/DRetroIkeaRepoRoot=Z:${RB}" "Z:${RB}\\packaging\\windows_setup.iss"
ls -lh "$ROOT/packaging/RetroIkea-Beta-Setup.exe"
echo "On Linux, install/run that setup with a 64-bit Wine prefix: wine64 \"$ROOT/packaging/RetroIkea-Beta-Setup.exe\""
