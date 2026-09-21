import numpy as np
from scipy.interpolate import PchipInterpolator
import torch

import config

device = config.DEVICE

def apply_curve_correction(depth_map, curve_points, strength=1.0):
    is_numpy = isinstance(depth_map, np.ndarray)
    if is_numpy:
        depth_tensor = torch.tensor(depth_map, dtype=torch.float32, device=device)
    else:
        depth_tensor = depth_map.to(device).float()
    curve_x, curve_y = zip(*sorted(curve_points))
    pchip = PchipInterpolator(curve_x, curve_y)
    lut_x = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
    lut_y = pchip(lut_x).astype(np.float32)
    lut_y = np.clip(lut_y, 0.0, 1.0)
    lut_tensor = torch.tensor(lut_y, device=device)
    scaled_depth = (1.0 - depth_tensor) * 65535.0
    indices_floor = torch.floor(scaled_depth).long().clamp(0, 65535)
    indices_ceil = torch.ceil(scaled_depth).long().clamp(0, 65535)
    weight = scaled_depth - indices_floor.float()
    curved_map = torch.lerp(lut_tensor[indices_floor], lut_tensor[indices_ceil], weight)
    curved_map = 1.0 - curved_map
    result = (1.0 - strength) * depth_tensor + strength * curved_map
    if is_numpy:
        return result.cpu().numpy()
    return result

def get_preset_curve(preset_name: str) -> list[tuple[float, float]]:
    return config.CURVE_PRESETS.get(preset_name.lower(), config.CURVE_PRESETS["linear"])