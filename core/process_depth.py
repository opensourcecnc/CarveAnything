import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import config

def bilateral_filter_gpu(img, d=5, sigma_r=0.02, sigma_s=6.0):
    H, W = img.shape
    pad = d // 2
    padded = F.pad(img.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='replicate').squeeze()
    y, x = torch.meshgrid(torch.arange(-pad, pad+1, device=config.DEVICE), torch.arange(-pad, pad+1, device=config.DEVICE), indexing='ij')
    w_spatial = torch.exp(-(x**2 + y**2) / (2 * sigma_s**2))
    unfolded = F.unfold(padded.unsqueeze(0).unsqueeze(0), kernel_size=d).squeeze(0)
    center = img.view(1, H*W)
    diff = unfolded - center
    w_range = torch.exp(-(diff**2) / (2 * sigma_r**2))
    weights = w_range * w_spatial.view(-1, 1)
    weights_sum = weights.sum(dim=0, keepdim=True) + 1e-6
    output = (unfolded * weights).sum(dim=0, keepdim=True) / weights_sum
    return output.view(H, W)

def poisson_reconstruct_fft(gx, gy):
    h, w = gx.shape
    f = np.zeros((h, w), dtype=np.float32)
    f[:, :-1] += gx[:, :-1]
    f[:, 1:]  -= gx[:, :-1]
    f[:-1, :] += gy[:-1, :]
    f[1:, :]  -= gy[:-1, :]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    denom = (2 * np.cos(2 * np.pi * xx / w) + 2 * np.cos(2 * np.pi * yy / h)- 4)
    denom[0, 0] = 1.0
    u_fft = np.fft.fft2(f) / denom
    u_fft[0, 0] = 0
    return np.real(np.fft.ifft2(u_fft)).astype(np.float32)

def process_matrix_offsets_gpu(matrix1_tensor, matrix2_tensor, tolerance=1e-6):
    relief = matrix2_tensor.clone()
    h, w = relief.shape
    border_values = torch.cat([matrix1_tensor[0, :], matrix1_tensor[-1, :], matrix1_tensor[:, 0], matrix1_tensor[:, -1]])
    flat = border_values.mean()
    mask = torch.zeros((h, w), dtype=torch.bool, device=config.DEVICE)
    mask[0, :] = True; mask[-1, :] = True; mask[:, 0] = True; mask[:, -1] = True
    surface = torch.zeros_like(relief)
    surface[mask] = matrix1_tensor[mask] - flat
    kernel = torch.tensor([[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]], dtype=torch.float32, device=config.DEVICE).view(1, 1, 3, 3)
    internal_mask = (~mask).float().view(1, 1, h, w)
    surface_pad = surface.view(1, 1, h, w)
    for _ in range(200000):
        prev_surface = surface_pad.clone()
        smoothed = F.conv2d(surface_pad, kernel, padding=1)
        surface_pad = torch.where(internal_mask == 1.0, smoothed, surface_pad)
        if torch.abs(surface_pad - prev_surface).max() < tolerance:
            print(f"Required degree of flatness reached!")
            break
    surface = surface_pad.view(h, w)
    return surface, relief - surface

def background_flattening (depth_in, threshold):
    depth = depth_in.copy()
    h, w = depth.shape
    seed_x = 5
    seed_y = h - 6
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_temp = depth.copy()
    cv2.floodFill(image=flood_temp, mask=flood_mask, seedPoint=(seed_x, seed_y), newVal=1.0, loDiff=threshold, upDiff=threshold, flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
    bg_mask = (flood_mask[1:-1, 1:-1] == 255)
    if np.any(bg_mask):
        bg_median = np.median(depth[bg_mask])
        current_mask = bg_mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(20):
            dilated = cv2.dilate(current_mask, kernel)
            border = (dilated == 1) & (current_mask == 0)
            if not np.any(border):
                break
            inner_edge = (current_mask == 1) & (cv2.erode(current_mask, kernel) == 0)
            edge_brightness_cutoff = np.mean(depth[inner_edge]) if np.any(inner_edge) else bg_median
            valid_halo_pixels = border & (depth >= edge_brightness_cutoff)
            if not np.any(valid_halo_pixels):
                break
            current_mask[valid_halo_pixels] = 1
        final_bg_mask = (current_mask == 1)
    else:
        final_bg_mask = bg_mask
    if np.any(final_bg_mask):
        depth[final_bg_mask] = bg_median
    return depth


def flatten_surface (hybrid_surface):
    hybrid_surface = torch.from_numpy(hybrid_surface).to(config.DEVICE)
    matrix1_tensor = torch.zeros_like(hybrid_surface)
    matrix1_tensor[0, :] = hybrid_surface[0, :]
    matrix1_tensor[-1, :] = hybrid_surface[-1, :]
    matrix1_tensor[:, 0] = hybrid_surface[:, 0]
    matrix1_tensor[:, -1] = hybrid_surface[:, -1]
    LOWRES_FACTOR = 4
    h, w = hybrid_surface.shape
    lr_size = (w // LOWRES_FACTOR, h // LOWRES_FACTOR)
    surface_lr = F.interpolate(hybrid_surface[None,None], size=(lr_size[1], lr_size[0]), mode="bilinear", align_corners=False).squeeze()
    matrix1_lr = torch.zeros_like(surface_lr)
    matrix1_lr[0,:]  = surface_lr[0,:]
    matrix1_lr[-1,:] = surface_lr[-1,:]
    matrix1_lr[:,0]  = surface_lr[:,0]
    matrix1_lr[:,-1] = surface_lr[:,-1]
    surface_lr, _ = process_matrix_offsets_gpu(matrix1_lr, surface_lr, tolerance=config.FLATNESS_TOLERANCE)
    surface = F.interpolate(surface_lr[None,None], size=(h,w), mode="bicubic", align_corners=False).squeeze()
    result_flattened_tensor = hybrid_surface - surface
    result_flattened_tensor -= result_flattened_tensor.min()
    result_flattened_tensor /= (result_flattened_tensor.max() + 1e-6)
    return result_flattened_tensor.cpu().numpy()

def apply_bas_relief(depth, image_path, flatten_background, background_threshold = config.BACKGROUND_THRESHOLD):
    basrelief_out_path = os.path.join(config.OUTPUT_DIR, f"{image_path}_basrelief.png")
    print("\nExecuting Bas-Relief Engine on Depth Map...")
    if flatten_background:
        depth = background_flattening(depth, background_threshold)
    depth_tensor = torch.tensor(depth, dtype=torch.float32, device=config.DEVICE)
    depth_tensor -= depth_tensor.min()
    depth_tensor /= (depth_tensor.max() + 1e-6)
    depth_tensor = torch.pow(depth_tensor, config.BAS_RELIEF_GAMMA)
    smooth_depth = bilateral_filter_gpu(depth_tensor, d=5, sigma_r=config.BILATERAL_SIGMA_COLOR, sigma_s=config.BILATERAL_SIGMA_SPACE)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=config.DEVICE).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=config.DEVICE).view(1, 1, 3, 3)
    padded_pillow = F.pad(smooth_depth.view(1, 1, *smooth_depth.shape), (1, 1, 1, 1), mode='circular')
    gx_pillow = F.conv2d(padded_pillow, sobel_x).squeeze()
    gy_pillow = F.conv2d(padded_pillow, sobel_y).squeeze()
    mag = torch.sqrt(gx_pillow**2 + gy_pillow**2)
    scale = (torch.tanh(mag / config.GRADIENT_ATTENUATION) * config.GRADIENT_ATTENUATION) / (mag + 1e-6)
    scale *= config.DETAIL_BOOST 
    gx_pillow *= scale
    gy_pillow *= scale
    gx_np = gx_pillow.cpu().numpy()
    gy_np = gy_pillow.cpu().numpy()
    hybrid_surface = poisson_reconstruct_fft(gx_np, gy_np)
    print(f"Flattening the bas relief ...")
    relief = flatten_surface(hybrid_surface)
    cv2.imwrite(basrelief_out_path, (relief * 65535).astype(np.uint16))
    print(f"Saved Smooth bas relief map: {basrelief_out_path}")
    return relief