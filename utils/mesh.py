import numpy as np
import config


def create_mesh(
    top_depth,
    top_detail,
    bot_depth=None,
    bot_detail=None,
    width=config.TARGET_WIDTH_MM,
    amplitude=config.RELIEF_AMPLITUDE_MM,
    base_thickness=config.TOTAL_TARGET_THICKNESS_MM,
    detail_weight=config.DETAIL_PERCENT,
    min_thickness=1.0,
    downsample=config.DOWNSAMPLE,
    closed=True,
):
    top_d = np.max(top_depth) - top_depth[::downsample, ::downsample]
    top_dt = top_detail[::downsample, ::downsample]
    H, W = top_d.shape
    height = width * (H / W)

    def compute_z(d, dt, detail_weight):
        return ((1.0 - detail_weight) * d + detail_weight * dt) * amplitude

    raw_top = np.flipud(compute_z(top_d, top_dt, detail_weight))
    Z_top = base_thickness + raw_top
    X, Y = np.meshgrid(np.linspace(0, width, W), np.linspace(0, height, H))

    if not closed:
        vertices = np.stack([X, Y, Z_top], axis=-1).reshape(-1, 3)
        idx = np.arange(H * W).reshape(H, W)
        v0 = idx[:-1, :-1].ravel()
        v1 = idx[:-1, 1:].ravel()
        v2 = idx[1:, 1:].ravel()
        v3 = idx[1:, :-1].ravel()
        faces = np.column_stack([v0, v1, v2, v3])
        return vertices.astype(np.float32), faces

    bot_d = np.max(bot_depth) - bot_depth[::downsample, ::downsample]
    bot_dt = (
        np.max(np.fliplr(bot_detail)) - bot_detail[::downsample, ::downsample]
    )
    raw_bot = np.fliplr(np.flipud(compute_z(bot_d, bot_dt, detail_weight)))
    Z_bottom = -raw_bot
    diff = Z_top - Z_bottom
    mask = diff < min_thickness

    if np.any(mask):
        adjustment = (min_thickness - diff[mask]) / 2.0
        Z_top[mask] += adjustment
        Z_bottom[mask] -= adjustment

    vertices = np.vstack(
        [
            np.stack([X, Y, Z_top], axis=-1).reshape(-1, 3),
            np.stack([X, Y, Z_bottom], axis=-1).reshape(-1, 3),
        ]
    )
    idx = np.arange(H * W).reshape(H, W)
    offset = H * W
    v0 = idx[:-1, :-1].ravel()
    v1 = idx[:-1, 1:].ravel()
    v2 = idx[1:, 1:].ravel()
    v3 = idx[1:, :-1].ravel()
    top_faces = np.column_stack([v0, v1, v2, v3])
    bot_faces = np.column_stack(
        [v3 + offset, v2 + offset, v1 + offset, v0 + offset]
    )
    l_edge = np.column_stack(
        [idx[:-1, 0], idx[1:, 0], idx[1:, 0] + offset, idx[:-1, 0] + offset]
    )
    r_edge = np.column_stack(
        [
            idx[:-1, -1],
            idx[:-1, -1] + offset,
            idx[1:, -1] + offset,
            idx[1:, -1],
        ]
    )
    t_edge = np.column_stack(
        [idx[0, :-1], idx[0, 1:], idx[0, 1:] + offset, idx[0, :-1] + offset]
    )
    b_edge = np.column_stack(
        [
            idx[-1, :-1],
            idx[-1, :-1] + offset,
            idx[-1, 1:] + offset,
            idx[-1, 1:],
        ]
    )
    faces = np.vstack([top_faces, bot_faces, l_edge, r_edge, t_edge, b_edge])

    return vertices.astype(np.float32), faces


def export_quad_obj(filename, vertices, faces, batch_size=10000):
    with open(filename, "w") as f:
        f.write("# Quad mesh\n")
        v_lines = []
        for x, y, z in vertices:
            v_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            if len(v_lines) >= batch_size:
                f.write("".join(v_lines))
                v_lines.clear()
        if v_lines:
            f.write("".join(v_lines))

        f_lines = []
        for face in faces:
            if len(face) == 4:
                f_lines.append(
                    f"f {face[0]+1} {face[1]+1} {face[2]+1} {face[3]+1}\n"
                )
            else:
                f_lines.append(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

            if len(f_lines) >= batch_size:
                f.write("".join(f_lines))
                f_lines.clear()
        if f_lines:
            f.write("".join(f_lines))