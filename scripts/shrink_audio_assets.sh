#!/usr/bin/env bash
# Re-encode shipped audio under assets/audio/ (smaller files; requires ffmpeg).
# - store_ambient_loop.wav -> store_ambient_loop.mp3 (then removes the WAV)
# - large MP3s: lower constant bitrate; medium: 128k; skips tiny SFX
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO="$(cd "$SCRIPT_DIR/../assets/audio" && pwd)"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "error: ffmpeg not found; install it to shrink audio." >&2
  exit 1
fi

reencode_mp3() {
  local f="$1" br="$2"
  local tmp
  tmp="$(mktemp "${TMPDIR:-/tmp}/retroikea-audio.XXXXXX.mp3")"
  ffmpeg -y -loglevel error -i "$f" -codec:a libmp3lame -b:a "$br" -ac 2 "$tmp"
  mv -f "$tmp" "$f"
}

cd "$AUDIO"

if [[ -f store_ambient_loop.wav ]]; then
  echo "encode store_ambient_loop.wav -> store_ambient_loop.mp3" >&2
  ffmpeg -y -loglevel error -i store_ambient_loop.wav -codec:a libmp3lame -b:a 112k -ac 2 store_ambient_loop.mp3
  rm -f store_ambient_loop.wav
fi

shopt -s nullglob
for f in *.mp3; do
  [[ -f "$f" ]] || continue
  sz=$(wc -c <"$f")
  if ((sz > 5000000)); then
    echo "shrink (96k): $f (${sz}B)" >&2
    reencode_mp3 "$f" 96k
  elif ((sz > 800000)); then
    echo "shrink (112k): $f (${sz}B)" >&2
    reencode_mp3 "$f" 112k
  elif ((sz > 200000)); then
    echo "shrink (128k): $f (${sz}B)" >&2
    reencode_mp3 "$f" 128k
  fi
done

echo "done: $AUDIO" >&2
