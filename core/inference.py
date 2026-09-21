import os
import cv2
import numpy as np
from PIL import Image

import config
from core.infer_da3_image import engine

def brighten_dark_areas(image: Image.Image, gamma: float = 0.5) -> Image.Image:
    """Brightens underexposed regions of an image using gamma correction."""
    img_np = np.array(image, dtype=np.float32) / 255.0
    adjusted = np.power(img_np, gamma)
    brightened_np = np.clip(adjusted * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(brightened_np)


def generate_tile(tile_h: int, tile_w: int, x0: int, y0: int, src_cv: np.ndarray, model_res: int) -> np.ndarray:
    """Slices an in-memory image array, predicts depth via engine, and resizes using cubic interpolation."""
    y1 = y0 + tile_h
    x1 = x0 + tile_w
    tile = src_cv[y0:y1, x0:x1]

    # Directly pass in-memory NumPy array slice to the engine
    prediction = engine.run_inference(
        images=[tile],
        export_dir=None,
        process_res=model_res,
        export_format="npz",
        conf_thresh_percentile=60.0,
    )

    tile_depth = prediction.depth.squeeze().astype(np.float32)
    tile_depth = cv2.resize(tile_depth, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    return (tile_depth - tile_depth.min()) / (tile_depth.max() - tile_depth.min() + 1e-6)


def process_image(image_path: str, tile_grid_mode: int) -> list:
    """Executes global depth guidance, multi-tile grid inference, overlap blending, and 16-bit depth export."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    source_out_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_source.png")
    depth_out_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_depth.png")

    # Skip processing if cached 16-bit depth file already exists on disk
    if os.path.exists(depth_out_path):
        print(f"\n[SKIP] Existing depth map found: {depth_out_path}")
        depth = cv2.imread(depth_out_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Could not read existing depth map: {depth_out_path}")
        depth = depth.astype(np.float32) / 65535.0
        return [depth, source_out_path, depth_out_path]

    # 1. Image preprocessing and scaling
    input_img = Image.open(image_path)
    enhanced = brighten_dark_areas(input_img, config.GAMMA_BRIGHTEN)
    enhanced.save(source_out_path)

    src_cv = cv2.imread(image_path)
    src_h, src_w = src_cv.shape[:2]
    scale = config.PROCESS_RES / max(src_h, src_w)
    w = int(round(src_w * scale))
    h = int(round(src_h * scale))

    if scale > 1.0:
        src_cv = cv2.GaussianBlur(src_cv, (0, 0), 0.5)
    interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    src_cv = cv2.resize(src_cv, (w, h), interpolation=interp)

    print("\n[1/5] Generating global guide & tiled high-res maps...")

    # 2. Low-resolution global guide pass
    low_res_depth = generate_tile(h, w, 0, 0, src_cv, 900)
    low_res_norm = (low_res_depth - low_res_depth.min()) / (low_res_depth.max() - low_res_depth.min() + 1e-6)

    # 3. Build cosine blending tile filter matrices
    tile_sizes = [[1, 1]] if tile_grid_mode == 1 else [[tile_grid_mode, tile_grid_mode]]
    filters = []

    for tile_size in tile_sizes:
        num_x, num_y = tile_size
        M = h // num_x
        N = w // num_y
        filter_dict = {
            "right_filter": np.zeros((M, N), dtype=np.float32),
            "left_filter": np.zeros((M, N), dtype=np.float32),
            "top_filter": np.zeros((M, N), dtype=np.float32),
            "bottom_filter": np.zeros((M, N), dtype=np.float32),
            "top_right_filter": np.zeros((M, N), dtype=np.float32),
            "top_left_filter": np.zeros((M, N), dtype=np.float32),
            "bottom_right_filter": np.zeros((M, N), dtype=np.float32),
            "bottom_left_filter": np.zeros((M, N), dtype=np.float32),
            "filter": np.zeros((M, N), dtype=np.float32),
        }

        for i in range(M):
            for j in range(N):
                x_val = np.cos((abs(M / 2 - i) / M) * np.pi) ** 2
                y_val = np.cos((abs(N / 2 - j) / N) * np.pi) ** 2

                filter_dict["right_filter"][i, j] = x_val if j > N / 2 else x_val * y_val
                filter_dict["left_filter"][i, j] = x_val if j < N / 2 else x_val * y_val
                filter_dict["top_filter"][i, j] = y_val if i < M / 2 else x_val * y_val
                filter_dict["bottom_filter"][i, j] = y_val if i > M / 2 else x_val * y_val

                if j > N / 2 and i < M / 2:
                    filter_dict["top_right_filter"][i, j] = 1.0
                elif j > N / 2:
                    filter_dict["top_right_filter"][i, j] = x_val
                elif i < M / 2:
                    filter_dict["top_right_filter"][i, j] = y_val
                else:
                    filter_dict["top_right_filter"][i, j] = x_val * y_val

                if j < N / 2 and i < M / 2:
                    filter_dict["top_left_filter"][i, j] = 1.0
                elif j < N / 2:
                    filter_dict["top_left_filter"][i, j] = x_val
                elif i < M / 2:
                    filter_dict["top_left_filter"][i, j] = y_val
                else:
                    filter_dict["top_left_filter"][i, j] = x_val * y_val

                if j > N / 2 and i > M / 2:
                    filter_dict["bottom_right_filter"][i, j] = 1.0
                elif j > N / 2:
                    filter_dict["bottom_right_filter"][i, j] = x_val
                elif i > M / 2:
                    filter_dict["bottom_right_filter"][i, j] = y_val
                else:
                    filter_dict["bottom_right_filter"][i, j] = x_val * y_val

                if j < N / 2 and i > M / 2:
                    filter_dict["bottom_left_filter"][i, j] = 1.0
                elif j < N / 2:
                    filter_dict["bottom_left_filter"][i, j] = x_val
                elif i > M / 2:
                    filter_dict["bottom_left_filter"][i, j] = y_val
                else:
                    filter_dict["bottom_left_filter"][i, j] = x_val * y_val

                filter_dict["filter"][i, j] = x_val * y_val

        filters.append(filter_dict)

    # 4. Tiled High-Resolution Inference Loop
    compiled_tiles_list = []
    for pass_index, tile_size in enumerate(tile_sizes):
        num_x, num_y = tile_size
        M = h // num_x
        N = w // num_y
        compiled_tiles = np.zeros((h, w), dtype=np.float32)

        x_coords = list(range(0, h, h // num_x))[:num_x]
        y_coords = list(range(0, w, w // num_y))[:num_y]
        x_coords_between = list(range((h // num_x) // 2, h, h // num_x))[: max(0, num_x - 1)]
        y_coords_between = list(range((w // num_y) // 2, w, w // num_y))[: max(0, num_y - 1)]

        x_coords_all = x_coords + x_coords_between
        y_coords_all = y_coords + y_coords_between

        print(f"\nMatrix size: {len(x_coords_all)} by {len(y_coords_all)}")
        model_res = max(1, config.PROCESS_RES // tile_grid_mode) if pass_index == 0 else max(1, config.PROCESS_RES // (tile_grid_mode * 2))

        count = 0
        for x in x_coords_all:
            for y in y_coords_all:
                tile_depth = generate_tile(M, N, y, x, src_cv, model_res)
                count += 1
                print(f"Processing tile {count}/{len(x_coords_all) * len(y_coords_all)}...")

                guide = low_res_norm[x : x + M, y : y + N]

                t_min = np.percentile(tile_depth, config.STITCHING_BOTTOM_TOLERANCE)
                t_max = np.percentile(tile_depth, 100 - config.STITCHING_TOLERANCE)
                g_min = np.percentile(guide, config.STITCHING_BOTTOM_TOLERANCE)
                g_max = np.percentile(guide, 100 - config.STITCHING_TOLERANCE)

                scale_val = (g_max - g_min) / (t_max - t_min + 1e-6)
                offset_val = g_min - scale_val * t_min

                adjusted = tile_depth * scale_val + offset_val
                residual = adjusted - guide
                low_freq = cv2.GaussianBlur(residual, (0, 0), sigmaX=M // 4)
                adjusted -= low_freq

                # Filter selection based on boundary conditions
                if y == min(y_coords_all) and x == min(x_coords_all):
                    selected_filter = filters[pass_index]["top_left_filter"]
                elif y == min(y_coords_all) and x == max(x_coords_all):
                    selected_filter = filters[pass_index]["bottom_left_filter"]
                elif y == max(y_coords_all) and x == min(x_coords_all):
                    selected_filter = filters[pass_index]["top_right_filter"]
                elif y == max(y_coords_all) and x == max(x_coords_all):
                    selected_filter = filters[pass_index]["bottom_right_filter"]
                elif y == min(y_coords_all):
                    selected_filter = filters[pass_index]["left_filter"]
                elif y == max(y_coords_all):
                    selected_filter = filters[pass_index]["right_filter"]
                elif x == min(x_coords_all):
                    selected_filter = filters[pass_index]["top_filter"]
                elif x == max(x_coords_all):
                    selected_filter = filters[pass_index]["bottom_filter"]
                else:
                    selected_filter = filters[pass_index]["filter"]

                compiled_tiles[x : x + M, y : y + N] += selected_filter * adjusted

        compiled_tiles[compiled_tiles < 0] = 0
        compiled_tiles_list.append(compiled_tiles)

    # 5. Final normalization & 16-bit PNG save
    final_depth = compiled_tiles_list[0]
    final_depth -= final_depth.min()
    final_depth /= final_depth.max() + 1e-6

    save_h, save_w = final_depth.shape
    rescale = config.PROCESS_RES / max(save_h, save_w)
    if rescale != 1.0:
        new_w = int(round(save_w * rescale))
        new_h = int(round(save_h * rescale))
        final_depth = cv2.resize(final_depth, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(depth_out_path, (final_depth * 65535).astype(np.uint16))
    return [final_depth, source_out_path, depth_out_path]