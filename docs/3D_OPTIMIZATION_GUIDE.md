# RetroIkea 3D Asset Optimization Guide

## Overview

The RetroIkea game currently uses ~179MB of GLB files. This guide explains how to reduce polygon count and file sizes while maintaining visual quality.

## Current Asset Analysis

- **Total GLB files**: 68 files
- **Total size**: ~179MB
- **Largest files**:
  - Meshy AI pipe model: 8.4MB
  - Character animations: 5.9-6.4MB each
- **Main issues**:
  - No compression (Draco) applied
  - High polygon counts for a retro-style game
  - Unoptimized textures
  - Duplicate files across directories

## Optimization Strategies

### 1. Automated Optimization Pipeline

Run the comprehensive optimization script:

```bash
./scripts/optimize_all_3d_assets.sh
```

This script will:
1. Analyze current assets
2. Create backups
3. Re-export with Blender optimization
4. Apply GLB-specific compression
5. Generate optimization report

### 2. Manual Optimization Steps

#### A. Polygon Reduction (Decimation)

For Blender re-export with decimation:

```bash
export RETROIKEA_DECIMATE_RATIO=0.7  # Reduce to 70% of original polygons
export RETROIKEA_OPTIMIZE_EXPORT=1
./scripts/export_all_project_3d.sh
```

#### B. Texture Optimization

```bash
export RETROIKEA_OPTIMIZE_TEXTURES=1
export RETROIKEA_GLTF_JPEG_QUALITY=75  # Adjust quality (40-100)
```

#### C. GLB Compression with gltf-pipeline

Install required tools:
```bash
npm install -g gltf-pipeline
```

Run compression:
```bash
./scripts/optimize_glb_files.sh
```

#### D. Advanced Compression with gltfpack

For maximum compression, install gltfpack:
```bash
# Download from https://github.com/zeux/meshoptimizer/releases
# Or build from source
git clone https://github.com/zeux/meshoptimizer.git
cd meshoptimizer
cmake . && make
sudo cp gltfpack /usr/local/bin/
```

### 3. Asset-Specific Optimizations

#### Character Animations
- Current: 5.9-6.4MB per animation
- Target: 1-2MB per animation
- Method: Reduce mesh complexity, use shared base mesh

#### Static Props
- Apply aggressive decimation (30-50% reduction)
- Use lower texture resolutions
- Consider using simple materials instead of textures

#### Duplicate Removal
```bash
# Find and remove duplicate GLB files
fdupes -r -d /home/memesdudeguy/Downloads/RetroIkea
```

### 4. Quality vs Size Trade-offs

| Setting | File Size | Visual Impact | Recommendation |
|---------|-----------|---------------|----------------|
| Decimation 0.9 | -10% | Minimal | Safe for all assets |
| Decimation 0.7 | -30% | Slight | Good for most assets |
| Decimation 0.5 | -50% | Noticeable | Background objects only |
| Draco Level 6 | -40-60% | None | Always use |
| Texture 2048→1024 | -75% | Minimal | Good for props |
| JPEG Quality 75 | -20-30% | Slight | Balanced choice |

### 5. Testing Optimized Assets

After optimization:

```bash
# Copy optimized files
rsync -av optimized_glb/ /home/memesdudeguy/Downloads/RetroIkea/

# Rebuild the game
cd /home/memesdudeguy/Downloads/RetroIkea
mkdir -p build && cd build
cmake .. && make

# Test the game
./vulkan_game
```

### 6. Monitoring Performance

Check file sizes:
```bash
# Analyze optimized assets
./scripts/analyze_glb_complexity.py optimized_glb/assets -v
```

### 7. Rollback if Needed

If optimization causes issues:
```bash
# Restore original files
rsync -av backup_original_glb/ /home/memesdudeguy/Downloads/RetroIkea/
```

## Expected Results

With proper optimization:
- **File size reduction**: 50-70% (179MB → 50-90MB)
- **Loading time**: 2-3x faster
- **Memory usage**: Significantly reduced
- **FPS improvement**: Especially on lower-end hardware

## Best Practices

1. **Always backup** before optimization
2. **Test incrementally** - optimize a few files first
3. **Profile performance** - measure FPS and loading times
4. **Visual quality checks** - ensure animations still look correct
5. **Version control** - commit optimized assets separately

## Tools Reference

- **Blender**: Decimation, texture optimization, re-export
- **gltf-pipeline**: Draco compression for GLB files
- **gltfpack**: Advanced mesh optimization
- **Python scripts**: Analysis and automation
- **Bash scripts**: Workflow automation

## Troubleshooting

### Animations broken after optimization
- Don't apply modifiers to animated meshes
- Use `export_apply=False` in Blender export

### Textures look wrong
- Check UV mapping preservation
- Adjust JPEG quality higher

### Game crashes with optimized assets
- Verify GLB files with: `gltf-validator file.glb`
- Check for missing textures or materials
- Ensure Assimp compatibility

## Future Improvements

1. **Level-of-Detail (LOD)** system
2. **Texture atlasing** for multiple small textures
3. **Procedural materials** instead of textures
4. **Streaming** for large environments
5. **GPU instancing** for repeated objects