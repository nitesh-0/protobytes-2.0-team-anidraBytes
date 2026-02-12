"""
Drishtimarga — Monocular Depth Estimation
Uses Depth Anything V2 RELATIVE model for reliable depth ordering.
Outputs normalized 0-1 depth map (0 = close, 1 = far).
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
    Monocular depth estimation using Depth Anything V2 (relative).
    Outputs normalized depth map where 0 = closest, 1 = farthest.
    Actual metric distances are computed downstream by blending with
    bbox-height heuristics in SpatialEngine.
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
        """Load Depth Anything V2 relative model."""
        try:
            logger.info(f"Loading Depth Anything V2 ({self.config.model_name})...")

            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            # Relative models — reliable depth ordering (NOT metric)
            model_map = {
                "small": "depth-anything/Depth-Anything-V2-Small-hf",
                "base": "depth-anything/Depth-Anything-V2-Base-hf",
                "large": "depth-anything/Depth-Anything-V2-Large-hf",
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
            logger.info(f"Depth model loaded on {self.device}")

        except Exception as e:
            logger.warning(f"Failed to load depth model: {e}")
            logger.warning("Falling back to bbox-based depth estimation")
            self._model_loaded = False

    @torch.no_grad()
    def _run_model(self, frame: np.ndarray) -> np.ndarray:
        """Run the depth model on a frame. Returns raw disparity output."""
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

        return depth

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Get depth map for the frame.
        Returns normalized depth map (0 = close, 1 = far) or None.
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

        # Normalize to 0-1 range
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-6:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_map)

        # Depth Anything V2 outputs DISPARITY (higher = closer).
        # Invert so that 0 = close, 1 = far for intuitive downstream use.
        depth_map = 1.0 - depth_map

        self._last_depth_map = depth_map
        return depth_map

    def get_depth_at_point(self, depth_map: Optional[np.ndarray],
                           x: int, y: int) -> float:
        """Get relative depth value (0=close, 1=far) at a pixel coordinate."""
        if depth_map is None:
            return 0.5  # default mid-range
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
        """Convert depth map to color visualization."""
        if depth_map is None:
            return None
        depth_vis = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    @property
    def inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def is_available(self) -> bool:
        return self._model_loaded
