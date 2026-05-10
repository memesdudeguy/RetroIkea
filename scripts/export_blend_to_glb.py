# Run inside Blender: bpy CLI API (Blender 4.x / 5.x glTF exporter).
# Same texture policy as batch_export_3d_to_glb.py — see RETROIKEA_GLTF_* env vars there.
import bpy
import os
import sys


def _glb_texture_export_attempts():
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

argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1 :]
else:
    argv = []

if len(argv) < 1:
    print("usage: blender --background FILE.blend --python ... -- OUTPUT.glb", file=sys.stderr)
    sys.exit(1)

out_path = argv[0]

base = dict(
    filepath=out_path,
    export_format="GLB",
    use_selection=False,
    export_apply=False,
    export_yup=True,
    export_unused_images=False,
)
for extra in _glb_texture_export_attempts():
    try:
        bpy.ops.export_scene.gltf(**base, **extra)
        break
    except Exception:
        continue
else:
    bpy.ops.export_scene.gltf(**base)

print(f"exported: {out_path}")
