#!/usr/bin/env python3
"""
Analyze GLB files to report polygon counts, texture sizes, and optimization opportunities.
Requires: pip install pygltflib pillow numpy
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import struct

try:
    from pygltflib import GLTF2
    import numpy as np
    from PIL import Image
    import io
except ImportError:
    print("Please install required packages:")
    print("  pip install pygltflib pillow numpy")
    sys.exit(1)


@dataclass
class MeshStats:
    name: str
    vertex_count: int
    triangle_count: int
    has_normals: bool
    has_texcoords: bool
    has_colors: bool
    has_joints: bool
    has_weights: bool


@dataclass
class TextureStats:
    index: int
    width: int
    height: int
    format: str
    size_bytes: int


@dataclass
class GLBStats:
    file_path: str
    file_size_mb: float
    total_vertices: int
    total_triangles: int
    mesh_count: int
    texture_count: int
    animation_count: int
    meshes: List[MeshStats]
    textures: List[TextureStats]
    has_draco: bool
    recommendations: List[str]


def get_accessor_data(gltf: GLTF2, accessor_index: int) -> Optional[np.ndarray]:
    """Extract data from a glTF accessor."""
    if accessor_index is None:
        return None
    
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[buffer_view.buffer]
    
    # For embedded buffers
    if hasattr(buffer, 'uri') and buffer.uri and buffer.uri.startswith('data:'):
        # Skip data URL prefix
        data = buffer.uri.split(',')[1]
        import base64
        buffer_data = base64.b64decode(data)
    else:
        # Buffer is already loaded by pygltflib
        buffer_data = gltf.binary_blob()
    
    return accessor.count


def analyze_mesh(gltf: GLTF2, mesh_index: int) -> MeshStats:
    """Analyze a single mesh."""
    mesh = gltf.meshes[mesh_index]
    name = mesh.name or f"Mesh_{mesh_index}"
    
    total_vertices = 0
    total_triangles = 0
    has_normals = False
    has_texcoords = False
    has_colors = False
    has_joints = False
    has_weights = False
    
    for primitive in mesh.primitives:
        # Count vertices
        if primitive.attributes.POSITION is not None:
            accessor = gltf.accessors[primitive.attributes.POSITION]
            total_vertices += accessor.count
        
        # Count triangles
        if primitive.indices is not None:
            accessor = gltf.accessors[primitive.indices]
            if primitive.mode in [None, 4]:  # TRIANGLES mode
                total_triangles += accessor.count // 3
        
        # Check attributes
        if primitive.attributes.NORMAL is not None:
            has_normals = True
        if primitive.attributes.TEXCOORD_0 is not None:
            has_texcoords = True
        if primitive.attributes.COLOR_0 is not None:
            has_colors = True
        if primitive.attributes.JOINTS_0 is not None:
            has_joints = True
        if primitive.attributes.WEIGHTS_0 is not None:
            has_weights = True
    
    return MeshStats(
        name=name,
        vertex_count=total_vertices,
        triangle_count=total_triangles,
        has_normals=has_normals,
        has_texcoords=has_texcoords,
        has_colors=has_colors,
        has_joints=has_joints,
        has_weights=has_weights
    )


def analyze_texture(gltf: GLTF2, texture_index: int) -> Optional[TextureStats]:
    """Analyze a texture."""
    texture = gltf.textures[texture_index]
    if texture.source is None:
        return None
    
    image = gltf.images[texture.source]
    
    # Default values
    width, height = 0, 0
    format = "unknown"
    size_bytes = 0
    
    if image.bufferView is not None:
        # Image is embedded
        buffer_view = gltf.bufferViews[image.bufferView]
        size_bytes = buffer_view.byteLength
        
        # Try to decode image to get dimensions
        try:
            if hasattr(gltf, '_glb_data'):
                buffer_data = gltf._glb_data[buffer_view.byteOffset:buffer_view.byteOffset + buffer_view.byteLength]
                img = Image.open(io.BytesIO(buffer_data))
                width, height = img.size
                format = img.format
        except:
            pass
    elif image.uri:
        # External image reference
        format = Path(image.uri).suffix.lower().strip('.')
    
    return TextureStats(
        index=texture_index,
        width=width,
        height=height,
        format=format,
        size_bytes=size_bytes
    )


def analyze_glb(file_path: str) -> GLBStats:
    """Analyze a GLB file and return statistics."""
    file_size = os.path.getsize(file_path)
    
    # Load GLB
    gltf = GLTF2.load(file_path)
    
    # Analyze meshes
    meshes = []
    total_vertices = 0
    total_triangles = 0
    
    for i in range(len(gltf.meshes)):
        mesh_stats = analyze_mesh(gltf, i)
        meshes.append(mesh_stats)
        total_vertices += mesh_stats.vertex_count
        total_triangles += mesh_stats.triangle_count
    
    # Analyze textures
    textures = []
    for i in range(len(gltf.textures)):
        texture_stats = analyze_texture(gltf, i)
        if texture_stats:
            textures.append(texture_stats)
    
    # Check for Draco compression
    has_draco = any(
        hasattr(ext, 'name') and ext.name == 'KHR_draco_mesh_compression'
        for ext in (gltf.extensionsUsed or [])
    )
    
    # Generate recommendations
    recommendations = []
    
    if total_triangles > 50000:
        recommendations.append(f"High polygon count ({total_triangles:,} triangles). Consider decimation.")
    
    if not has_draco:
        recommendations.append("No Draco compression detected. Enable for 50-90% size reduction.")
    
    for texture in textures:
        if texture.width > 2048 or texture.height > 2048:
            recommendations.append(f"Texture {texture.index} is {texture.width}x{texture.height}. Consider reducing to 2048x2048.")
    
    if len(textures) > 5:
        recommendations.append(f"Many textures ({len(textures)}). Consider texture atlasing.")
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        recommendations.append(f"Large file size ({file_size / 1024 / 1024:.1f}MB). Apply optimizations.")
    
    return GLBStats(
        file_path=file_path,
        file_size_mb=file_size / 1024 / 1024,
        total_vertices=total_vertices,
        total_triangles=total_triangles,
        mesh_count=len(meshes),
        texture_count=len(textures),
        animation_count=len(gltf.animations) if gltf.animations else 0,
        meshes=meshes,
        textures=textures,
        has_draco=has_draco,
        recommendations=recommendations
    )


def print_stats(stats: GLBStats, verbose: bool = False):
    """Print statistics for a GLB file."""
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(stats.file_path)}")
    print(f"Path: {stats.file_path}")
    print(f"Size: {stats.file_size_mb:.2f} MB")
    print(f"\nGeometry:")
    print(f"  Total Vertices: {stats.total_vertices:,}")
    print(f"  Total Triangles: {stats.total_triangles:,}")
    print(f"  Meshes: {stats.mesh_count}")
    print(f"  Animations: {stats.animation_count}")
    print(f"\nTextures: {stats.texture_count}")
    
    if verbose and stats.textures:
        for tex in stats.textures:
            print(f"  - Texture {tex.index}: {tex.width}x{tex.height} {tex.format} ({tex.size_bytes / 1024:.1f} KB)")
    
    print(f"\nCompression:")
    print(f"  Draco: {'Yes' if stats.has_draco else 'No'}")
    
    if stats.recommendations:
        print(f"\nRecommendations:")
        for rec in stats.recommendations:
            print(f"  - {rec}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze GLB files for optimization opportunities")
    parser.add_argument("paths", nargs="+", help="GLB files or directories to analyze")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed information")
    parser.add_argument("-s", "--summary", action="store_true", help="Show summary only")
    parser.add_argument("-o", "--output", help="Save report to JSON file")
    
    args = parser.parse_args()
    
    # Collect all GLB files
    glb_files = []
    for path in args.paths:
        if os.path.isfile(path) and path.endswith('.glb'):
            glb_files.append(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.glb'):
                        glb_files.append(os.path.join(root, file))
    
    if not glb_files:
        print("No GLB files found.")
        return
    
    # Analyze files
    all_stats = []
    total_size = 0
    total_vertices = 0
    total_triangles = 0
    
    for glb_file in sorted(glb_files):
        try:
            stats = analyze_glb(glb_file)
            all_stats.append(stats)
            total_size += stats.file_size_mb
            total_vertices += stats.total_vertices
            total_triangles += stats.total_triangles
            
            if not args.summary:
                print_stats(stats, args.verbose)
        except Exception as e:
            print(f"Error analyzing {glb_file}: {e}")
    
    # Print summary
    if args.summary or len(glb_files) > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total Files: {len(all_stats)}")
        print(f"Total Size: {total_size:.2f} MB")
        print(f"Total Vertices: {total_vertices:,}")
        print(f"Total Triangles: {total_triangles:,}")
        print(f"Average File Size: {total_size / len(all_stats):.2f} MB")
        print(f"Average Triangles per File: {total_triangles // len(all_stats):,}")
        
        # Top 5 largest files
        print(f"\nLargest Files:")
        for stats in sorted(all_stats, key=lambda x: x.file_size_mb, reverse=True)[:5]:
            print(f"  - {os.path.basename(stats.file_path)}: {stats.file_size_mb:.2f} MB ({stats.total_triangles:,} triangles)")
    
    # Save report
    if args.output:
        report = {
            "summary": {
                "total_files": len(all_stats),
                "total_size_mb": total_size,
                "total_vertices": total_vertices,
                "total_triangles": total_triangles,
            },
            "files": [
                {
                    "path": stats.file_path,
                    "size_mb": stats.file_size_mb,
                    "vertices": stats.total_vertices,
                    "triangles": stats.total_triangles,
                    "has_draco": stats.has_draco,
                    "recommendations": stats.recommendations,
                }
                for stats in all_stats
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()