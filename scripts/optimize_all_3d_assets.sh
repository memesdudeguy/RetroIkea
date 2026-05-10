#!/usr/bin/env bash
# Comprehensive 3D asset optimization for RetroIkea
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
BACKUP_DIR="${BACKUP_DIR:-$REPO/backup_original_glb}"
OPTIMIZED_DIR="${OPTIMIZED_DIR:-$REPO/optimized_glb}"
REPORT_FILE="$REPO/optimization_report.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}RetroIkea 3D Asset Optimization${NC}"
echo "================================="

# Step 1: Analyze current assets
echo -e "\n${YELLOW}Step 1: Analyzing current GLB files...${NC}"
if command -v python3 >/dev/null 2>&1 && [ -f "$SCRIPT_DIR/analyze_glb_complexity.py" ]; then
    python3 "$SCRIPT_DIR/analyze_glb_complexity.py" "$REPO/assets" -s -o "$REPORT_FILE" || true
else
    echo "Skipping analysis (Python script not available)"
fi

# Step 2: Backup original files
echo -e "\n${YELLOW}Step 2: Backing up original files...${NC}"
if [ ! -d "$BACKUP_DIR" ]; then
    mkdir -p "$BACKUP_DIR"
    rsync -av "$REPO/assets/" "$BACKUP_DIR/assets/"
    echo -e "${GREEN}Backup created at: $BACKUP_DIR${NC}"
else
    echo "Backup already exists at: $BACKUP_DIR"
fi

# Step 3: Re-export with optimization (if Blender is available)
echo -e "\n${YELLOW}Step 3: Re-exporting with Blender optimization...${NC}"
if command -v blender >/dev/null 2>&1; then
    export RETROIKEA_OPTIMIZE_EXPORT=1
    export RETROIKEA_DECIMATE_RATIO=0.7
    export RETROIKEA_OPTIMIZE_TEXTURES=1
    export RETROIKEA_GLTF_JPEG_QUALITY=75
    
    # Run the export script if it exists
    if [ -f "$SCRIPT_DIR/export_all_project_3d.sh" ]; then
        echo "Running Blender export with optimization..."
        "$SCRIPT_DIR/export_all_project_3d.sh"
    else
        echo "Export script not found, creating file list..."
        find "$REPO/assets" -name "*.fbx" -o -name "*.glb" > "$REPO/3d_files.txt"
        echo "Please run Blender export manually with optimization enabled"
    fi
else
    echo "Blender not found in PATH. Skipping re-export."
fi

# Step 4: Apply additional GLB optimization
echo -e "\n${YELLOW}Step 4: Applying GLB-specific optimizations...${NC}"
if [ -f "$SCRIPT_DIR/optimize_glb_files.sh" ]; then
    "$SCRIPT_DIR/optimize_glb_files.sh"
else
    echo "GLB optimization script not found"
fi

# Step 5: Generate optimization report
echo -e "\n${YELLOW}Step 5: Generating final report...${NC}"
if [ -d "$OPTIMIZED_DIR" ] && command -v python3 >/dev/null 2>&1; then
    python3 "$SCRIPT_DIR/analyze_glb_complexity.py" "$OPTIMIZED_DIR/assets" -s || true
fi

# Summary
echo -e "\n${GREEN}Optimization Complete!${NC}"
echo "====================="
echo ""
echo "Next steps:"
echo "1. Review the optimized files in: $OPTIMIZED_DIR"
echo "2. Test the game with optimized assets"
echo "3. If everything works correctly:"
echo "   rsync -av $OPTIMIZED_DIR/ $REPO/"
echo "4. Rebuild the project"
echo ""
echo "To restore original files if needed:"
echo "   rsync -av $BACKUP_DIR/ $REPO/"
echo ""

# Show size comparison
if [ -d "$OPTIMIZED_DIR" ]; then
    original_size=$(du -sh "$REPO/assets" 2>/dev/null | cut -f1)
    optimized_size=$(du -sh "$OPTIMIZED_DIR/assets" 2>/dev/null | cut -f1 || echo "N/A")
    echo "Size comparison:"
    echo "  Original: $original_size"
    echo "  Optimized: $optimized_size"
fi