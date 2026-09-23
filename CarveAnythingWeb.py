import os, cv2
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

def generate_preview_obj_with_normals(filepath, vertices, faces):
    vertices = np.array(vertices, dtype=np.float32)
    triangles = []
    for f in faces:
        if len(f) == 4:
            triangles.append([f[0], f[1], f[2]])
            triangles.append([f[0], f[2], f[3]])
        elif len(f) == 3:
            triangles.append([f[0], f[1], f[2]])
    triangles = np.array(triangles, dtype=np.int32)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(vertices, dtype=np.float32)
    np.add.at(normals, triangles[:, 0], face_normals)
    np.add.at(normals, triangles[:, 1], face_normals)
    np.add.at(normals, triangles[:, 2], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals /= norms
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    preview_vertices = vertices - center
    max_extent = np.abs(preview_vertices).max()
    if max_extent > 0:
        preview_vertices /= max_extent
    preview_vertices[:, 0] *= -1.0  # Flip X
    preview_vertices[:, 2] *= -1.0  # Flip Z
    color = " 0.900000 0.900000 0.900000"
    with open(filepath, "w") as f:
        for v in preview_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}{color}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for tri in triangles:
            v1, v2, v3 = tri[0] + 1, tri[1] + 1, tri[2] + 1
            f.write(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}\n")

def generate_lowres_preview(relief_top, detail_top, relief_bottom, detail_bottom, detail_weight, closed, output_path, size, target_thickness, target_res=300):
    h, w = relief_top.shape
    scale = min(target_res / float(h), target_res / float(w))
    if scale < 1.0:
        new_w = max(2, int(w * scale))
        new_h = max(2, int(h * scale))
        relief_top_low = cv2.resize(relief_top, (new_w, new_h), interpolation=cv2.INTER_AREA)
        detail_top_low = cv2.resize(detail_top, (new_w, new_h), interpolation=cv2.INTER_AREA)
        relief_bot_low = (cv2.resize(relief_bottom, (new_w, new_h), interpolation=cv2.INTER_AREA) if relief_bottom is not None else None)
        detail_bot_low = (cv2.resize(detail_bottom, (new_w, new_h), interpolation=cv2.INTER_AREA) if detail_bottom is not None else None)
    else:
        relief_top_low = relief_top
        detail_top_low = detail_top
        relief_bot_low = relief_bottom
        detail_bot_low = detail_bottom
    v_low, f_low = create_mesh(relief_top_low, detail_top_low, relief_bot_low, detail_bot_low, size=size, target_thickness=target_thickness, detail_weight=detail_weight, closed=closed)
    generate_preview_obj_with_normals(output_path, v_low, f_low)


def run_pipeline(top_image_path, bottom_image_path, macro_preset_name, curve_strength, detail_preset_name, detail_curve_strength, tile_mode, bg_tolerance, detail_percent, force_depth, save_raw_depth, process_res, mesh_resolution_scale, mesh_long_side, mesh_thickness):
    try:
        if not top_image_path or not os.path.exists(top_image_path):
            return None, "Error: Please upload or drag and drop a Top Image."
        has_bottom = bool(bottom_image_path and os.path.exists(bottom_image_path))
        tile_grid_mode = tile_mode
        background_threshold = (float(bg_tolerance) / 100.0 if bg_tolerance > 0 else getattr(config, "BACKGROUND_THRESHOLD", 0.0))
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
            da3_raw_mesh_out_path = os.path.join(out_dir, f"{base_name_top}_{base_name_bottom}_da3_pad.obj")
            preview_out_path = os.path.join(out_dir, f"{base_name_top}_{base_name_bottom}_hybrid_pad_preview.obj")
        else:
            mesh_out_path = os.path.join(out_dir, f"{base_name_top}_single_sided_hybrid_pad.obj")
            da3_raw_mesh_out_path = os.path.join(out_dir, f"{base_name_top}_single_sided_da3_pad.obj")
            preview_out_path = os.path.join(out_dir, f"{base_name_top}_single_sided_hybrid_pad_preview.obj")
        top = process_image(top_image_path, tile_grid_mode, force_depth, process_res)
        if not save_raw_depth:
            relief_top = apply_bas_relief(top[0], base_name_top, flatten_background, background_threshold)
        else:
            relief_top = top[0]
        img_top = Image.open(top[1]).convert("L")
        if img_top.size != (top[0].shape[1], top[0].shape[0]):
            img_top = img_top.resize(
                (top[0].shape[1], top[0].shape[0]), Image.Resampling.LANCZOS
            )
        detail_top = np.array(img_top, dtype=np.float32)
        detail_top /= np.max(detail_top) if np.max(detail_top) > 0 else 1.0
        if not save_raw_depth:
            detail_top = 0.5 + 0.5 * detail_top
        else:
            detail_top = 0*detail_top
        if has_bottom:
            bottom = process_image(bottom_image_path, tile_grid_mode, force_depth, process_res)
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
        vertices, faces = create_mesh(relief_top, detail_top, relief_bottom if has_bottom else None, detail_bottom if has_bottom else None, mesh_long_side, mesh_thickness, detail_percent_val, mesh_resolution_scale, has_bottom)
        export_quad_obj(mesh_out_path if not save_raw_depth else da3_raw_mesh_out_path, vertices, faces)
        generate_lowres_preview(relief_top=relief_top, detail_top=detail_top, relief_bottom=relief_bottom if has_bottom else None, detail_bottom=detail_bottom if has_bottom else None, detail_weight=detail_percent_val, closed=has_bottom, output_path=preview_out_path, target_res=600,size=mesh_long_side, target_thickness=mesh_thickness)
        if hasattr(engine, "unload"):
            engine.unload()
        log_msg = f"SUCCESS!\nFull Mesh Saved: {os.path.abspath(mesh_out_path)}\nPreview Mesh Loaded: {os.path.abspath(preview_out_path)}"
        return preview_out_path, log_msg, False
    except Exception as e:
        import traceback
        return None, f"Execution Error: {str(e)}\n\n{traceback.format_exc()}"
custom_css = """
.toggle-switch input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 40px !important;
    height: 20px !important;
    background: #4a4a4a;
    border-radius: 20px;
    position: relative;
    cursor: pointer;
    outline: none;
    transition: background 0.3s;
    
    /* FIX: Force vertical centering of the pseudo-element inside the track */
    display: inline-flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
}

.toggle-switch input[type="checkbox"]:checked {
    background: #2563eb;
}

.toggle-switch input[type="checkbox"]::before {
    content: '';
    position: absolute;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    left: 3px;
    background: white;
    transition: transform 0.3s;
    
    /* FIX: Clear top positioning so align-items handles vertical alignment */
    top: auto !important; 
}

.toggle-switch input[type="checkbox"]:checked::before {
    transform: translateX(20px);
}
"""

def build_app():
    preset_options = get_preset_names()

    with gr.Blocks(title="CarveAnything", css=custom_css) as app:
        gr.Markdown("# CarveAnything")

        with gr.Row():
            # Left Control Panel
            with gr.Column(scale=1):
                gr.Markdown("### 1. Select / Drop Images")
                with gr.Tabs("Select Source Image(s):"):
                    with gr.Tab("Top"):
                        top_path = gr.Image(
                            label="Top Image (Click to Browse or Drag & Drop)",
                            type="filepath",
                            sources=["upload"],
                        )
                    with gr.Tab("Bottom (optional)"):
                        bottom_path = gr.Image(
                            label="Bottom Image (Optional)",
                            type="filepath",
                            sources=["upload"],
                        )
                gr.Markdown("### 2. Mesh Parameters")
                with gr.Tabs("Correction curves:"):
                    with gr.Tab("Volume curve"):
                        macro_preset = gr.Dropdown(
                            label="Volume Curve:",
                            choices=preset_options,
                            value=preset_options[0] if preset_options else "linear",
                        )
                        macro_strength = gr.Slider(
                            label="Volume Curve Strength", minimum=0.0, maximum=1.0, value=0.75, step=0.05
                        )
                    with gr.Tab("Details curve"):
                        detail_preset = gr.Dropdown(
                            label="Details Curve:",
                            choices=preset_options,
                            value=preset_options[0] if preset_options else "linear",
                        )
                        detail_strength = gr.Slider(
                            label="Details Curve Strength", minimum=0.0, maximum=1.0, value=0.25, step=0.05
                        )
                        detail_pct = gr.Slider(
                            label="Detail Percent (0 - 100)",
                            minimum=0,
                            maximum=100,
                            value=10,
                            step=1,
                        )
                    with gr.Tab("Mesh parameters"):
                        mesh_resolution_scale = gr.Slider(
                            label="Face Count Scaling",
                            minimum=0.01,
                            maximum=1,
                            value=1,
                            step=0.01,
                        )
                        mesh_long_side = gr.Slider(
                            label="Longest Side Size (mm)",
                            minimum=0,
                            maximum=1000,
                            value=120,
                            step=1,
                        )
                        mesh_thickness = gr.Slider(
                            label="Thickness (mm)",
                            minimum=0,
                            maximum=100,
                            value=15,
                            step=1,
                        )
                        bg_tol = gr.Slider(
                            label="Background Flattening Tolerance (0 - disabled)",
                            minimum=0,
                            maximum=10,
                            value=0,
                            step=0.01,
                        )


                gr.Markdown("### 3. Execution Settings")
                process_res = gr.Slider(
                    label="Output Resolution", minimum=1000, maximum=6000, value=config.PROCESS_RES, step=10
                )
                tile_mode_choices = [
                    ("1 Tile (Native)", 1),
                    ("2x2 Grid (9 tiles in total)", 2),
                    ("3x3 Grid (27 tiles in total)", 3),
                    ("4x4 Grid (49 Tiles in total)", 4),
                    ("5x5 Grid (81 Tiles in total)", 5),
                ]
                tile_mode = gr.Dropdown(
                    label="Depth-Anything Tiling Strategy",
                    choices=tile_mode_choices,
                    value=4,
                )
                force_depth = gr.Checkbox(
                            label="Force DepthAnything 3 Inference",
                            value=False,
                            info="Bypass cached depth maps",
                            elem_classes=["toggle-switch"],
                        )
                save_raw_depth = gr.Checkbox(
                            label="Output Raw DepthAnything 3",
                            value=False,
                            info="Save mesh based on the raw Depth Anything 3 output",
                            elem_classes=["toggle-switch"],
                        )
                btn_run = gr.Button("Generate 3D Mesh", variant="primary")

            # Right Viewport & Output Panel
            with gr.Column(scale=2):
                gr.Markdown("### 3D Mesh Preview")
                mesh_3d = gr.Model3D(label="Interactive 3D Viewport", height=500)
                status_box = gr.Textbox(label="Log", interactive=False, lines=6)

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
                force_depth,
                save_raw_depth,
                process_res,
                mesh_resolution_scale,
                mesh_long_side,
                mesh_thickness,
            ],
            outputs=[mesh_3d, status_box, force_depth],
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
        inbrowser=True,
    )