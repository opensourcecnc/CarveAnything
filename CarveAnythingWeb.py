import os
import gradio as gr
import numpy as np
from PIL import Image

import config
from core.infer_da3_image import engine
from core.inference import process_image
from core.process_depth import apply_bas_relief
from utils.curves import apply_curve_correction
from utils.mesh import create_mesh, export_quad_obj


def get_preset_names():
    curve_dict = getattr(config, "CURVE_PRESETS", {})
    return list(curve_dict.keys()) if curve_dict else ["linear"]

def triangulate_and_export_obj(filepath, vertices, faces):
    with open(filepath, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            if len(face) == 4:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                f.write(f"f {face[0]+1} {face[2]+1} {face[3]+1}\n")
            elif len(face) == 3:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def generate_lowres_preview(relief_top, detail_top, relief_bottom, detail_bottom, detail_weight, closed, output_path, target_res=256):
    h, w = relief_top.shape
    max_dim = max(h, w)
    if max_dim > target_res:
        stride = int(np.ceil(max_dim / target_res))
        relief_top_low = relief_top[::stride, ::stride]
        detail_top_low = detail_top[::stride, ::stride]
        relief_bot_low = relief_bottom[::stride, ::stride] if relief_bottom is not None else None
        detail_bot_low = detail_bottom[::stride, ::stride] if detail_bottom is not None else None
    else:
        relief_top_low = relief_top
        detail_top_low = detail_top
        relief_bot_low = relief_bottom
        detail_bot_low = detail_bottom
    v_low, f_low = create_mesh(relief_top_low, detail_top_low, relief_bot_low, detail_bot_low, detail_weight=detail_weight, closed=closed)
    triangulate_and_export_obj(output_path, v_low, f_low)


def run_pipeline(top_image_path, bottom_image_path, macro_preset_name, curve_strength, detail_preset_name, detail_curve_strength, tile_grid_mode_str, bg_tolerance, detail_percent):
    try:
        if not top_image_path or not os.path.exists(top_image_path):
            return None, "Error: Please upload or drag and drop a Top Image."
        has_bottom = bool(bottom_image_path and os.path.exists(bottom_image_path))
        tile_grid_mode = int(tile_grid_mode_str.split()[0])
        background_threshold = (float(bg_tolerance) / 1000.0 if bg_tolerance > 0 else getattr(config, "BACKGROUND_THRESHOLD", 0.0))
        flatten_background = background_threshold > 0
        detail_percent_val = float(detail_percent) / 100.0
        curve_presets = getattr(config, "CURVE_PRESETS", {})
        s_curve = curve_presets.get(macro_preset_name)
        detail_s_curve = curve_presets.get(detail_preset_name)
        out_dir = getattr(config, "OUTPUT_DIR", "./outputs")
        os.makedirs(out_dir, exist_ok=True)
        base_name_top = os.path.splitext(os.path.basename(top_image_path))[0]
        if has_bottom:
            base_name_bottom = os.path.splitext(os.path.basename(bottom_image_path))[0]
            mesh_out_path = os.path.join(out_dir, f"{base_name_top}_{base_name_bottom}_hybrid_pad.obj")
            preview_out_path = os.path.join(out_dir, f"{base_name_top}_{base_name_bottom}_hybrid_pad_preview.obj")
        else:
            mesh_out_path = os.path.join(out_dir, f"{base_name_top}_single_sided_hybrid_pad.obj")
            preview_out_path = os.path.join(out_dir, f"{base_name_top}_single_sided_hybrid_pad_preview.obj")
        top = process_image(top_image_path, tile_grid_mode)
        relief_top = apply_bas_relief(
            top[0], base_name_top, flatten_background, background_threshold
        )
        img_top = Image.open(top[1]).convert("L")
        if img_top.size != (top[0].shape[1], top[0].shape[0]):
            img_top = img_top.resize(
                (top[0].shape[1], top[0].shape[0]), Image.Resampling.LANCZOS
            )
        detail_top = np.array(img_top, dtype=np.float32)
        detail_top /= np.max(detail_top) if np.max(detail_top) > 0 else 1.0
        detail_top = 0.5 + 0.5 * detail_top
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
        if s_curve is not None:
            relief_top = apply_curve_correction(relief_top, s_curve, strength=curve_strength)
        if detail_s_curve is not None:
            detail_top = apply_curve_correction(detail_top, detail_s_curve, strength=detail_curve_strength)
        if has_bottom:
            if s_curve is not None:
                relief_bottom = apply_curve_correction(relief_bottom, s_curve, strength=curve_strength)
            if detail_s_curve is not None:
                detail_bottom = apply_curve_correction(detail_bottom, detail_s_curve, strength=detail_curve_strength)
        vertices, faces = create_mesh(relief_top, detail_top, relief_bottom if has_bottom else None, detail_bottom if has_bottom else None, detail_weight=detail_percent_val, closed=has_bottom)
        export_quad_obj(mesh_out_path, vertices, faces)
        generate_lowres_preview(relief_top, detail_top, relief_bottom if has_bottom else None, detail_bottom if has_bottom else None, detail_weight=detail_percent_val, closed=has_bottom, output_path=preview_out_path, target_res=256)
        if hasattr(engine, "unload"):
            engine.unload()
        log_msg = f"SUCCESS!\nFull Mesh Saved: {os.path.abspath(mesh_out_path)}\nPreview Mesh Loaded: {os.path.abspath(preview_out_path)}"
        return preview_out_path, log_msg
    except Exception as e:
        import traceback
        return None, f"Execution Error: {str(e)}\n\n{traceback.format_exc()}"

def build_app():
    preset_options = get_preset_names()

    with gr.Blocks(title="DA3 Bas Relief Generator") as app:
        gr.Markdown("# DA3 Based Bas Relief Mesh Generator")

        with gr.Row():
            # Left Control Panel
            with gr.Column(scale=1):
                gr.Markdown("### 1. Select / Drop Images")
                top_path = gr.Image(
                    label="Top Image (Click to Browse or Drag & Drop)",
                    type="filepath",
                    sources=["upload"],
                )
                bottom_path = gr.Image(
                    label="Bottom Image (Optional)",
                    type="filepath",
                    sources=["upload"],
                )

                gr.Markdown("### 2. Curve Presets")
                macro_preset = gr.Dropdown(
                    label="Macro Curve Preset",
                    choices=preset_options,
                    value=preset_options[0] if preset_options else "linear",
                )
                macro_strength = gr.Slider(
                    label="Macro Strength", minimum=0.0, maximum=1.0, value=0.5, step=0.05
                )

                detail_preset = gr.Dropdown(
                    label="Detail Curve Preset",
                    choices=preset_options,
                    value=preset_options[0] if preset_options else "linear",
                )
                detail_strength = gr.Slider(
                    label="Detail Strength", minimum=0.0, maximum=1.0, value=0.5, step=0.05
                )

                gr.Markdown("### 3. Execution Settings")
                tile_mode = gr.Radio(
                    label="Depth-Anything Tiling Strategy",
                    choices=[
                        "1 Tile (Native)",
                        "2 (9 Tiles - 2x2 Grid)",
                        "3 (27 Tiles - 3x3 Grid)",
                        "4 (49 Tiles - 4x4 Grid)",
                        "5 (81 Tiles - 5x5 Grid)",
                    ],
                    value="2 (9 Tiles - 2x2 Grid)",
                )

                bg_tol = gr.Slider(
                    label="Background Detection Tolerance (0 - 1000)",
                    minimum=0,
                    maximum=1000,
                    value=0,
                    step=1,
                )
                detail_pct = gr.Slider(
                    label="Detail Percent (0 - 100)",
                    minimum=0,
                    maximum=100,
                    value=5,
                    step=1,
                )

                btn_run = gr.Button("Generate 3D Mesh", variant="primary")

            # Right Viewport & Output Panel
            with gr.Column(scale=2):
                gr.Markdown("### 3D Mesh Preview")
                mesh_3d = gr.Model3D(label="Interactive 3D Viewport", height=500)
                status_box = gr.Textbox(label="System Console Log", interactive=False, lines=6)

        # Wire Button Event
        btn_run.click(
            fn=run_pipeline,
            inputs=[
                top_path,
                bottom_path,
                macro_preset,
                macro_strength,
                detail_preset,
                detail_strength,
                tile_mode,
                bg_tol,
                detail_pct,
            ],
            outputs=[mesh_3d, status_box],
        )

    return app


if __name__ == "__main__":
    out_dir = os.path.abspath(getattr(config, "OUTPUT_DIR", "./outputs"))
    os.makedirs(out_dir, exist_ok=True)
    
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        allowed_paths=[out_dir],
    )