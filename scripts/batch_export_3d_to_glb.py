# Batch import 3D files and export each as GLB (Blender 4.x / 5.x, headless).
# Usage: blender --background --python this_script.py -- OUT_ROOT PATH_LIST.txt [REPO_ROOT]
#
# Smaller files: we still output .glb (vulkan_game uses Assimp; no Draco/meshopt in bundled Assimp).
# Embedded images are re-encoded — default JPEG (tunable). Optional env:
#   RETROIKEA_GLTF_LOSSLESS=1   — prefer AUTO (near-lossless) then high-Q JPEG
#   RETROIKEA_GLTF_JPEG_QUALITY=70  — JPEG quality 40–100 (default 70)
import bpy
import os
import sys


def _argv_after_dd():
    a = sys.argv
    if "--" not in a:
        return []
    return a[a.index("--") + 1 :]


def clear_scene():
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_one(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=path)
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=path)
        elif ext == ".obj":
            if hasattr(bpy.ops.wm, "obj_import"):
                bpy.ops.wm.obj_import(filepath=path)
            else:
                bpy.ops.import_scene.obj(filepath=path)
        else:
            print(f"skip unsupported ext {ext}: {path}", flush=True)
            return False
    except Exception as ex:
        print(f"import failed {path}: {ex}", flush=True)
        return False
    return True


def _glb_texture_export_attempts():
    """Order of bpy.ops.export_scene.gltf kwargs for embedded images (inside .glb)."""
    try:
        q = int(os.environ.get("RETROIKEA_GLTF_JPEG_QUALITY", "70"))
    except ValueError:
        q = 70
    q = max(40, min(100, q))
    lossless = os.environ.get("RETROIKEA_GLTF_LOSSLESS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if lossless:
        return (
            dict(export_image_format="AUTO"),
            dict(export_image_format="JPEG", export_jpeg_quality=92),
            dict(
                export_image_format="WEBP",
                export_image_quality=90,
                export_image_webp_fallback=True,
            ),
        )
    return (
        dict(export_image_format="JPEG", export_jpeg_quality=q),
        dict(
            export_image_format="WEBP",
            export_image_quality=max(55, min(90, q + 5)),
            export_image_webp_fallback=True,
        ),
        dict(export_image_format="AUTO"),
    )


def decimate_meshes(target_ratio=0.5):
    """Apply decimation to all mesh objects to reduce polygon count."""
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            # Select and make active
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            
            # Add decimate modifier
            decimate = obj.modifiers.new(name="Decimate", type='DECIMATE')
            decimate.ratio = target_ratio
            decimate.use_collapse_triangulate = True
            
            # Apply modifier (optional - can export with modifiers unapplied)
            # bpy.ops.object.modifier_apply(modifier=decimate.name)
            
            obj.select_set(False)


def optimize_textures():
    """Downscale textures that are too large."""
    for img in bpy.data.images:
        if img.size[0] > 2048 or img.size[1] > 2048:
            # Scale down to max 2048
            scale = min(2048.0 / img.size[0], 2048.0 / img.size[1])
            new_width = int(img.size[0] * scale)
            new_height = int(img.size[1] * scale)
            img.scale(new_width, new_height)


def export_glb(out_path: str, optimize: bool = True) -> None:
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    
    # Optional optimization
    if optimize:
        # Check environment variable for decimation ratio
        decimate_ratio = float(os.environ.get("RETROIKEA_DECIMATE_RATIO", "0.7"))
        if 0 < decimate_ratio < 1:
            decimate_meshes(decimate_ratio)
        
        # Optimize textures if requested
        if os.environ.get("RETROIKEA_OPTIMIZE_TEXTURES", "").lower() in ("1", "true", "yes"):
            optimize_textures()
    
    # export_apply=False: True breaks skinned GLB (inverse bind / weights vs Assimp).
    base = dict(
        filepath=out_path,
        export_format="GLB",
        use_selection=False,
        export_apply=False,
        export_yup=True,
        export_unused_images=False,
        # Enable Draco compression if available
        export_draco_mesh_compression_enable=True,
        export_draco_mesh_compression_level=6,
        export_draco_position_quantization=14,
        export_draco_normal_quantization=10,
        export_draco_texcoord_quantization=12,
        export_draco_color_quantization=8,
        export_draco_generic_quantization=12,
    )
    attempts = _glb_texture_export_attempts()
    last_err = None
    for extra in attempts:
        try:
            bpy.ops.export_scene.gltf(**base, **extra)
            return
        except Exception as ex:
            last_err = ex
            continue
    if last_err:
        raise last_err
    bpy.ops.export_scene.gltf(**base)


def main():
    args = _argv_after_dd()
    if len(args) < 2:
        print(
            "usage: blender --background --python batch_export_3d_to_glb.py -- OUT_ROOT PATH_LIST.txt [REPO_ROOT]",
            file=sys.stderr,
        )
        sys.exit(1)
    out_root = os.path.abspath(args[0])
    list_path = os.path.abspath(args[1])
    repo_root = os.path.abspath(args[2]) if len(args) > 2 else ""

    if not os.path.isfile(list_path):
        print(f"list file not found: {list_path}", file=sys.stderr)
        sys.exit(1)

    with open(list_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]

    os.makedirs(out_root, exist_ok=True)
    ok = 0
    for inp in lines:
        inp = os.path.abspath(os.path.expanduser(inp))
        if not os.path.isfile(inp):
            print(f"missing: {inp}", flush=True)
            continue

        if repo_root and inp.startswith(repo_root + os.sep):
            rel = os.path.relpath(inp, repo_root)
            sub = os.path.join(out_root, rel)
            out_path = os.path.splitext(sub)[0] + ".glb"
        else:
            # Stable names for CMake (basename); avoids huge __path__ prefixes.
            base_nm = os.path.basename(inp)
            out_path = os.path.join(out_root, "_external", os.path.splitext(base_nm)[0] + ".glb")

        clear_scene()
        if not import_one(inp):
            continue
        if not bpy.context.scene.objects:
            print(f"no objects after import: {inp}", flush=True)
            continue
        try:
            # Check if optimization is requested via environment variable
            optimize = os.environ.get("RETROIKEA_OPTIMIZE_EXPORT", "1").lower() in ("1", "true", "yes")
            export_glb(out_path, optimize=optimize)
            print(f"ok: {inp} -> {out_path}", flush=True)
            ok += 1
        except Exception as ex:
            print(f"export failed {inp}: {ex}", flush=True)

    print(f"done, exported {ok} / {len(lines)}", flush=True)


main()
