import os
import sys
import numpy as np
from PIL import Image

import config
from core.infer_da3_image import engine
from core.inference import process_image
from core.process_depth import apply_bas_relief
from utils.curves import apply_curve_correction
from utils.mesh import create_mesh, export_quad_obj

def main():
    print("-" * 60)
    print("DA3 BASED BAS RELIEF MESH GENERATOR")
    print("-" * 60)

    # Automatically construct numeric mapping (1-indexed) directly from config.CURVE_PRESETS
    preset_names = list(config.CURVE_PRESETS.keys())
    preset_mapping = {str(i + 1): name for i, name in enumerate(preset_names)}

    print("\nAvailable Curve Presets:")
    for num, name in preset_mapping.items():
        print(f"  [{num}] {name}")

    macro_choice = input(
        f"\nChoose Macro Curve Preset (1-{len(preset_mapping)}) [Default: 1 (linear)]: "
    ).strip()
    macro_preset_name = preset_mapping.get(macro_choice, "linear")

    macro_strength_input = input(
        "Enter Macro Curve Strength (0.0 - 1.0) [Default: 0.5]: "
    ).strip()
    try:
        curve_strength = (
            float(macro_strength_input) if macro_strength_input else 0.5
        )
    except ValueError:
        curve_strength = 0.5

    detail_choice = input(
        f"Choose Detail Curve Preset (1-{len(preset_mapping)}) [Default: 1 (linear)]: "
    ).strip()
    detail_preset_name = preset_mapping.get(detail_choice, "linear")

    detail_strength_input = input(
        "Enter Detail Curve Strength (0.0 - 1.0) [Default: 0.5]: "
    ).strip()
    try:
        detail_curve_strength = (
            float(detail_strength_input) if detail_strength_input else 0.5
        )
    except ValueError:
        detail_curve_strength = 0.5

    s_curve = config.CURVE_PRESETS.get(
        macro_preset_name, config.CURVE_PRESETS["linear"]
    )
    detail_s_curve = config.CURVE_PRESETS.get(
        detail_preset_name, config.CURVE_PRESETS["linear"]
    )

    print("\nSelect Depth-Anything Tiling Strategy:")
    print("  [1] 1 Tile   (Native, passes PROCESS_RES directly)")
    print("  [2] 9 Tiles  (2x2 Grid, runs model at PROCESS_RES // 2)")
    print("  [3] 27 Tiles  (3x3 Grid, runs model at PROCESS_RES // 3)")
    print("  [4] 49 Tiles (4x4 Grid, runs model at PROCESS_RES // 4)")
    print("  [5] 81 Tiles (5x5 Grid, runs model at PROCESS_RES // 5)")

    tile_choice = input("Enter selection (1-5) [Default: 2]: ").strip()
    tile_grid_mode = (
        int(tile_choice) if tile_choice in ["1", "2", "3", "4", "5"] else 2
    )

    bg_tolerance = input(
        "Enter background detection tolerance (0 - 1000) [Default: 0]: "
    ).strip()
    background_threshold = (
        float(bg_tolerance) / 1000 if bg_tolerance else config.BACKGROUND_THRESHOLD
    )

    flatten_background = background_threshold > 0

    detail_percent_input = input(
        "Enter detail percent (0 - 100) [Default: 5]: "
    ).strip()
    detail_percent = (
        float(int(detail_percent_input)) / 100
        if detail_percent_input
        else config.DETAIL_PERCENT
    )

    top_image_path = (
        input("\nEnter absolute path to top image: ")
        .strip()
        .replace('"', "")
        .replace("'", "")
    )
    if not os.path.exists(top_image_path):
        print(f"\n[ERROR] File path not found: {top_image_path}")
        sys.exit(1)

    bottom_image_path = (
        input(
            "Enter absolute path to bottom image (Leave BLANK for single-sided): "
        )
        .strip()
        .replace('"', "")
        .replace("'", "")
    )
    has_bottom = len(bottom_image_path) > 0

    if has_bottom and not os.path.exists(bottom_image_path):
        print(f"\n[ERROR] File path not found: {bottom_image_path}")
        sys.exit(1)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    base_name_top = os.path.splitext(os.path.basename(top_image_path))[0]

    if has_bottom:
        base_name_bottom = os.path.splitext(os.path.basename(bottom_image_path))[0]
        mesh_out_path = os.path.join(
            config.OUTPUT_DIR, f"{base_name_top}_{base_name_bottom}_hybrid_pad.stl"
        )
    else:
        mesh_out_path = os.path.join(
            config.OUTPUT_DIR, f"{base_name_top}_single_sided_hybrid_pad.obj"
        )

    # Process Top View Image
    top = process_image(top_image_path, tile_grid_mode)
    relief_top = apply_bas_relief(
        top[0], base_name_top, flatten_background, background_threshold
    )
    # relief_top = top[0]

    img_top = Image.open(top[1]).convert("L")
    if img_top.size != (top[0].shape[1], top[0].shape[0]):
        img_top = img_top.resize(
            (top[0].shape[1], top[0].shape[0]), Image.Resampling.LANCZOS
        )
    detail_top = np.array(img_top, dtype=np.float32)
    detail_top /= np.max(detail_top) if np.max(detail_top) > 0 else 1.0
    detail_top = 0.5 + 0.5 * detail_top

    # Process Bottom View Image (if single-sided, fill zeros/ones)
    if has_bottom:
        bottom = process_image(bottom_image_path, tile_grid_mode)
        relief_bottom = apply_bas_relief(
            bottom[0], base_name_bottom, flatten_background, background_threshold
        )

        img_bottom = Image.open(bottom[1]).convert("L")
        if img_bottom.size != (bottom[0].shape[1], bottom[0].shape[0]):
            img_bottom = img_bottom.resize(
                (bottom[0].shape[1], bottom[0].shape[0]), Image.Resampling.LANCZOS
            )
        detail_bottom = np.array(img_bottom, dtype=np.float32)
        detail_bottom /= np.max(detail_bottom) if np.max(detail_bottom) > 0 else 1.0
        detail_bottom = 1.0 - detail_bottom
    else:
        relief_bottom = np.zeros_like(relief_top)
        detail_bottom = np.ones_like(detail_top)

    print("\nApplying curve parameter adjustments to bas-relief and detail maps...")
    relief_top = apply_curve_correction(relief_top, s_curve, strength=curve_strength)
    detail_top = apply_curve_correction(
        detail_top, detail_s_curve, strength=detail_curve_strength
    )

    if has_bottom:
        relief_bottom = apply_curve_correction(
            relief_bottom, s_curve, strength=curve_strength
        )
        detail_bottom = apply_curve_correction(
            detail_bottom, detail_s_curve, strength=detail_curve_strength
        )

    print("Generating 3D mesh triangles...")
    vertices, faces = create_mesh(
        relief_top,
        detail_top,
        relief_bottom if has_bottom else None,
        detail_bottom if has_bottom else None,
        detail_weight=detail_percent,
        closed=has_bottom,
    )

    export_quad_obj(mesh_out_path, vertices, faces)
    engine.unload()

    print("-" * 60)
    print(f"PROCESS SUCCESSFUL! Mesh saved to: {mesh_out_path}")
    print("-" * 60)


if __name__ == "__main__":
    main()