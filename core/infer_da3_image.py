import gc
import torch
import typer
import numpy as np
from PIL import Image
from typing import Any, List, Optional, Union

import config
from depth_anything_3.api import DepthAnything3

class DepthInferenceEngine:

    def __init__(self, model_dir: str = config.MODEL_DIR, device: str = config.DEVICE):
        self.model_dir = model_dir
        self.device = device
        self.model = None

    def load_model(self):
        if self.model is None:
            typer.echo(f"Loading Depth-Anything-3 model from {self.model_dir}...")
            self.model = DepthAnything3.from_pretrained(self.model_dir).to(self.device)
        return self.model

    """Runs local inference directly on paths, numpy arrays, or PIL images."""
    def run_inference(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        export_dir: Optional[str] = None,
        export_format: str = "npz",
        process_res: int = 504,
        conf_thresh_percentile: float = 60.0,
        **kwargs
    ) -> Any:
        model = self.load_model()
        inference_kwargs = {
            "image": images,
            "export_dir": export_dir,
            "export_format": export_format,
            "process_res": process_res,
            "conf_thresh_percentile": conf_thresh_percentile,
            **kwargs
        }
        return model.inference(**inference_kwargs)

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

engine = DepthInferenceEngine()