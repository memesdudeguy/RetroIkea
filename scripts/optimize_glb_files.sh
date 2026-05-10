#!/usr/bin/env bash
# Optimize GLB files using gltf-pipeline with various compression techniques
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO/optimized_glb}"

# Check if required tools are installed
check_tools() {
    local missing=()
    
    # Check for gltf-pipeline
    if ! command -v gltf-pipeline >/dev/null 2>&1; then
        missing+=("gltf-pipeline")
    fi
    
    # Check for gltfpack (optional but recommended)
    if ! command -v gltfpack >/dev/null 2>&1; then
        echo "Warning: gltfpack not found. Install for better compression." >&2
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo "Error: Missing required tools: ${missing[*]}" >&2
        echo "Install with: npm install -g gltf-pipeline" >&2
        exit 1
    fi
}

# Process a single GLB file
process_glb() {
    local input="$1"
    local output="$2"
    local basename=$(basename "$input")
    
    echo "Processing: $basename"
    
    # Create output directory
    mkdir -p "$(dirname "$output")"
    
    # Option 1: Use gltfpack if available (best compression)
    if command -v gltfpack >/dev/null 2>&1; then
        echo "  Using gltfpack for maximum compression..."
        
        # Simplify mesh while preserving animations
        # -si: simplify mesh to target ratio
        # -kn: keep named nodes (for animations)
        # -ke: keep extras
        # -tc: texture compression with basis universal
        gltfpack -i "$input" -o "$output" \
            -si 0.7 \
            -kn \
            -ke \
            -tc \
            -cc \
            || echo "  gltfpack failed, falling back to gltf-pipeline"
    fi
    
    # Option 2: Use gltf-pipeline (fallback or if gltfpack not available)
    if [ ! -f "$output" ]; then
        echo "  Using gltf-pipeline..."
        
        # Basic Draco compression
        gltf-pipeline -i "$input" -o "$output" \
            --draco.compressionLevel=10 \
            --draco.quantizePositionBits=14 \
            --draco.quantizeNormalBits=10 \
            --draco.quantizeTexcoordBits=12 \
            --draco.quantizeColorBits=8 \
            --draco.quantizeGenericBits=12
    fi
    
    # Report size reduction
    if [ -f "$output" ]; then
        local original_size=$(stat -c%s "$input" 2>/dev/null || stat -f%z "$input")
        local new_size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output")
        local reduction=$((100 - (new_size * 100 / original_size)))
        echo "  Reduced from $(numfmt --to=iec $original_size) to $(numfmt --to=iec $new_size) ($reduction% reduction)"
    fi
}

# Main processing
main() {
    check_tools
    
    echo "Starting GLB optimization..."
    echo "Output directory: $OUT_DIR"
    
    # Find all GLB files
    local count=0
    while IFS= read -r -d '' glb_file; do
        # Calculate relative path for output
        local rel_path="${glb_file#$REPO/}"
        local out_path="$OUT_DIR/$rel_path"
        
        process_glb "$glb_file" "$out_path"
        ((count++))
    done < <(find "$REPO/assets" -name "*.glb" -print0)
    
    echo ""
    echo "Processed $count GLB files"
    echo "Optimized files saved to: $OUT_DIR"
    echo ""
    echo "To use optimized files:"
    echo "  1. Review the optimized files"
    echo "  2. Run: rsync -av $OUT_DIR/ $REPO/"
    echo "  3. Rebuild the project"
}

main "$@"