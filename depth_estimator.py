"""
Drishtimarga — Monocular Depth Estimation
Uses Depth Anything V2 METRIC models that output actual meters.
Falls back to bbox-based estimation if model fails to load.
"""

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from config import DepthConfig

logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    Monocular depth estimation using Depth Anything V2 Metric models.
    These models output depth directly in meters (no normalization needed).
    Falls back to bbox-size heuristic if model fails to load.
    """

    def __init__(self, config: DepthConfig):
        self.config = config
        self.model = None
        self.transform = None
        self.device = self._resolve_device()
        self._last_depth_map: Optional[np.ndarray] = None
        self._frame_counter = 0
        self._last_inference_ms = 0.0
        self._model_loaded = False

        if config.enabled:
            self._load_model()

    def _resolve_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _load_model(self):
        """Load Depth Anything V2 Metric model."""
        try:
            logger.info(f"Loading Depth Anything V2 Metric ({self.config.model_name})...")

            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            # Metric models output depth in METERS directly
            # Indoor: trained on Hypersim, range 0-20m
            # Outdoor: trained on Virtual KITTI, range 0-80m
            model_map = {
                "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                "base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
                "large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                "outdoor-small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
                "outdoor-base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
                "outdoor-large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
            }
            model_id = model_map.get(self.config.model_name, self.config.model_name)

            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()

            # Warm up
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self._run_model(dummy)

            self._model_loaded = True
            logger.info(f"Depth Metric model loaded on {self.device}")

        except Exception as e:
            logger.warning(f"Failed to load depth model: {e}")
            logger.warning("Falling back to bbox-based depth estimation")
            self._model_loaded = False

    @torch.no_grad()
    def _run_model(self, frame: np.ndarray) -> np.ndarray:
        """Run the depth model on a frame. Returns depth in meters."""
        from PIL import Image

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth

        # Interpolate to original frame size
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(frame.shape[0], frame.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return depth  # values are in METERS

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Get depth map for the frame.
        Returns depth map in METERS (Metric model output) or None.
        Runs model only every N frames for performance; returns cached map otherwise.
        """
        self._frame_counter += 1

        if not self._model_loaded:
            return None

        # Skip frames for speed
        if (self._frame_counter % self.config.run_every_n_frames != 0
                and self._last_depth_map is not None):
            return self._last_depth_map

        t_start = time.perf_counter()
        depth_map = self._run_model(frame)
        self._last_inference_ms = (time.perf_counter() - t_start) * 1000

        # Metric model outputs depth in meters directly.
        # Clip to valid range (0.3m - 20m for indoor model)
        depth_map = np.clip(depth_map, 0.3, 20.0)

        self._last_depth_map = depth_map
        return depth_map

    def get_depth_at_point(self, depth_map: Optional[np.ndarray],
                           x: int, y: int) -> float:
        """Get depth in meters at a pixel coordinate."""
        if depth_map is None:
            return 5.0  # default mid-range in meters
        h, w = depth_map.shape[:2]
        x = np.clip(int(x), 0, w - 1)
        y = np.clip(int(y), 0, h - 1)
        # Sample a small patch for robustness
        patch_size = 5
        y1 = max(0, y - patch_size)
        y2 = min(h, y + patch_size)
        x1 = max(0, x - patch_size)
        x2 = min(w, x + patch_size)
        return float(np.median(depth_map[y1:y2, x1:x2]))

    def get_depth_visualization(self, depth_map: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Convert depth map (in meters) to color visualization."""
        if depth_map is None:
            return None
        # Normalize meters to 0-255 for visualization (0m=bright, 20m=dark)
        depth_normalized = np.clip(depth_map / 20.0, 0.0, 1.0)
        depth_vis = (depth_normalized * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    @property
    def inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def is_available(self) -> bool:
        return self._model_loaded
