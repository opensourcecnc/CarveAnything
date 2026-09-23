import torch

# ============================================================
# CONFIGURATION PARAMETERS (EXACT MATCH FROM ORIGINAL SCRIPT)
# ============================================================

# Directory Paths
MODEL_DIR = "./da-models"
OUTPUT_DIR = "./output"

# Tiling & Stitching Settings
TILE_GRID_MODE = 2  # Will be overwritten at runtime (1, 2, 3, or 4)
STITCHING_TOLERANCE = 5
STITCHING_BOTTOM_TOLERANCE = 5

# Background Processing
BACKGROUND_THRESHOLD = 0       # Threshold (0.0 to 1.0) below which pixels are considered background
FLATTEN_BACKGROUND = True         # Toggle to enable/disable background flattening

# Inference & Image Enhancement Constants
GAMMA_BRIGHTEN = 0.3          # Gamma applied to the source image
PROCESS_RES = 3000            # Global Inference & Processing resolution
BAS_RELIEF_GAMMA = 0.7        # Gamma curve scaling for bas-relief structure
FLATNESS_TOLERANCE = 1e-5  # Number of flattening iterations

# Gradient & Filtering Parameters
GRADIENT_ATTENUATION = 0.07
DETAIL_BOOST = 1.0
BILATERAL_SIGMA_COLOR = 0.02
BILATERAL_SIGMA_SPACE = 10

# Blending Parameters
PILLOW_BLEND_WEIGHT = 0.85 # HYBRID BLENDING BALANCE (0.0 = Pure Script 1 Flat, 1.0 = Pure Script 2 Pillow)

# 3D Mesh Generation Parameters
TARGET_WIDTH_MM = 120.0           # Set physical width in mm
TOTAL_TARGET_THICKNESS_MM = 16.0   # Absolute maximum height of final mesh
RELIEF_AMPLITUDE_MM = 18.0         # Total budget for 3D displacement shapes
DETAIL_PERCENT = 0.05             # Budget % given to fine image texture (0.0 = pure depth map)
DOWNSAMPLE = 1                    # Mesh downsampling step factor (1 = full res)

# Curve Control Point Presets (Exact Tuned Values)
CURVE_PRESETS = {
    "linear": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
    "default_macro": [(0.0, 0.0), (0.2418, 0.7685), (1.0, 1.0)],
    "deepen_foreground": [(0.0, 0.0), (0.40, 0.20), (0.60, 0.55), (1.0, 1.0)],
    "balanced": [(0.0, 0.0), (0.15, 0.25), (0.60, 0.55), (1.0, 1.0)],
    "deepen_midtones": [(0.0, 0.0), (0.20, 0.10), (0.50, 0.50), (0.80, 0.90), (1.0, 1.0)],
    "deepen_background": [(0.0, 0.0), (0.30, 0.45), (0.60, 0.75), (1.0, 1.0)]
}

# Hardware Acceleration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"