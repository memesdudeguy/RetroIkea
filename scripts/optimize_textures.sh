#!/usr/bin/env bash
# Optimize and downgrade textures for smaller size
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="$REPO/backup_textures"
TEXTURES_DIR="$REPO/assets/textures"

# Create backup
echo "Creating texture backup..."
if [ ! -d "$BACKUP_DIR" ]; then
    mkdir -p "$BACKUP_DIR"
    cp -r "$TEXTURES_DIR"/* "$BACKUP_DIR/" || true
fi

# Function to optimize a single image
optimize_image() {
    local input="$1"
    local filename=$(basename "$input")
    local name="${filename%.*}"
    local ext="${filename##*.}"
    
    echo "Optimizing: $filename"
    
    # Convert to JPG for better compression (except for images needing alpha)
    if [[ "$name" == "sign" ]]; then
        # Sign needs transparency, resize and optimize PNG
        convert "$input" -resize "512x512>" -strip -quality 85 "$input.tmp"
        optipng -o7 -quiet "$input.tmp" || true
        mv "$input.tmp" "$input"
    else
        # Convert to JPG with lower quality and smaller size
        local output="${input%.*}.jpg"
        convert "$input" \
            -resize "256x256>" \
            -strip \
            -interlace Plane \
            -gaussian-blur 0.05 \
            -quality 60 \
            -sampling-factor 4:2:0 \
            "$output.tmp"
        
        # Remove original if different format
        if [[ "$input" != "$output" ]]; then
            rm -f "$input"
        fi
        
        # Optimize with jpegoptim
        if command -v jpegoptim >/dev/null 2>&1; then
            jpegoptim -m60 -s "$output.tmp" || true
        fi
        
        mv "$output.tmp" "$output"
    fi
}

# Check for required tools
if ! command -v convert >/dev/null 2>&1; then
    echo "Error: ImageMagick 'convert' command not found"
    echo "Install with: sudo apt-get install imagemagick"
    exit 1
fi

# Process all textures
cd "$TEXTURES_DIR"
for img in *.png *.jpg *.jpeg; do
    [ -f "$img" ] || continue
    optimize_image "$img"
done

# Report results
echo ""
echo "Texture optimization complete!"
echo "Original backup in: $BACKUP_DIR"
echo ""
echo "Size comparison:"
original_size=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
new_size=$(du -sh "$TEXTURES_DIR" 2>/dev/null | cut -f1)
echo "  Original: $original_size"
echo "  Optimized: $new_size"