"""
Drishtimarga v2 — Metric Depth Estimation

WHY THE OLD DEPTH WAS WRONG:
════════════════════════════
The v1 code used the *relative* Depth Anything V2 model, which outputs
arbitrary per-frame values (NOT meters).  It then normalized every frame
independently to 0-1:

    depth = (depth - d_min) / (d_max - d_min)   # WRONG for distances
    meters = 0.3 + depth * 19.7                  # meaningless mapping

This means "0.5" could mean 2m in one frame and 12m in the next, because
normalization destroys absolute scale.  A chair 2m away would report
different distances depending on what ELSE is in the frame.

THE FIX:
════════
Use the **Metric** variant of Depth Anything V2, which is fine-tuned to
output depth values directly in **meters**.  No normalization, no mapping.
The model output IS the distance.

Model variants (all output meters):
  metric-outdoor-small  → fast,  good for street/outdoor scenes
  metric-outdoor-base   → balanced (DEFAULT) — best for Drishtimarga
  metric-outdoor-large  → most accurate, heavier
  metric-indoor-small   → optimized for indoor scenes
  metric-indoor-base    → indoor balanced

Fallback chain:
  1. Metric depth model (real meters) — preferred
  2. Relative depth model + bbox-size fusion — acceptable
  3. Bbox-height heuristic alone — last resort
"""

import logging
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from config import DepthConfig

logger = logging.getLogger(__name__)

# ── HuggingFace model ID mapping ──
_MODEL_MAP: Dict[str, str] = {
    # Metric models (output meters directly)
    "metric-outdoor-small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "metric-outdoor-base":  "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "metric-outdoor-large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "metric-indoor-small":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "metric-indoor-base":   "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "metric-indoor-large":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    # Relative models (arbitrary scale — NOT recommended)
    "relative-small":       "depth-anything/Depth-Anything-V2-Small-hf",
    "relative-base":        "depth-anything/Depth-Anything-V2-Base-hf",
}


class DepthEstimator:
    """
    Monocular depth estimation with metric output.

    Primary: Depth Anything V2 Metric (outputs real meters)
    Fallback: bbox-height heuristic if model fails to load.
    """

    def __init__(self, config: DepthConfig):
        self.config = config
        self.device = self._resolve_device()
        self.model = None
        self.processor = None
        self._is_metric = False
        self._model_loaded = False
        self._last_depth_map: Optional[np.ndarray] = None
        self._prev_depth_map: Optional[np.ndarray] = None
        self._frame_counter = 0
        self._last_inference_ms = 0.0

        if config.enabled:
            self._load_model()

    def _resolve_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _load_model(self):
        """Load the depth model with fallback chain."""
        model_key = self.config.model_name
        model_id = _MODEL_MAP.get(model_key, model_key)
        self._is_metric = "metric" in model_key.lower() or "metric" in model_id.lower()

        try:
            logger.info(f"Loading depth model: {model_key}")
            logger.info(f"  HuggingFace ID: {model_id}")
            logger.info(f"  Metric output: {self._is_metric}")

            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()

            if self.device == "cuda" and torch.cuda.is_available():
                # Use FP16 for speed on GPU
                self.model = self.model.half()
                logger.info("  Using FP16 on GPU")

            # Warm up
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self._run_inference(dummy)

            self._model_loaded = True
            mode = "METRIC (meters)" if self._is_metric else "RELATIVE (needs calibration)"
            logger.info(f"Depth model ready — output mode: {mode}")

        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            logger.warning("Will use bbox-height heuristic for distance estimation")
            self._model_loaded = False

    @torch.no_grad()
    def _run_inference(self, frame: np.ndarray) -> np.ndarray:
        """Run depth model on a single frame."""
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Match model precision
        if self.device == "cuda" and next(self.model.parameters()).dtype == torch.float16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}

        outputs = self.model(**inputs)
        depth = outputs.predicted_depth  # (1, H, W)

        # Resize to original frame dimensions
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1).float(),
            size=(frame.shape[0], frame.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return depth

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Get depth map for the current frame.

        For METRIC models: returned values are depth in meters.
        For RELATIVE models: returned values are relative (higher = farther).

        Uses frame-skipping: only runs inference every N frames, reuses
        the last depth map otherwise.
        """
        self._frame_counter += 1

        if not self._model_loaded:
            return None

        # Frame skipping for performance
        if (self._frame_counter % self.config.run_every_n_frames != 0
                and self._last_depth_map is not None):
            return self._last_depth_map

        t0 = time.perf_counter()
        depth_map = self._run_inference(frame)
        self._last_inference_ms = (time.perf_counter() - t0) * 1000

        if self._is_metric:
            # Metric model: values ARE meters already.
            # Clamp to reasonable range (0.1m – 80m)
            depth_map = np.clip(depth_map, 0.1, 80.0)

            # Temporal EMA smoothing across frames for stability
            if (self._prev_depth_map is not None
                    and self._prev_depth_map.shape == depth_map.shape):
                alpha = self.config.temporal_alpha
                depth_map = alpha * depth_map + (1.0 - alpha) * self._prev_depth_map
            self._prev_depth_map = depth_map.copy()
        else:
            # Relative model: normalize to 0-1 for visualization only.
            # Distance estimation will need bbox heuristic.
            d_min, d_max = depth_map.min(), depth_map.max()
            if d_max - d_min > 1e-6:
                depth_map = (depth_map - d_min) / (d_max - d_min)

        self._last_depth_map = depth_map
        return depth_map

    def get_distance_at_bbox(self, depth_map: Optional[np.ndarray],
                              cx: float, cy: float,
                              bbox: Optional[np.ndarray] = None,
                              bbox_h: float = 0,
                              class_name: str = "") -> float:
        """
        Get the metric distance (meters) at a bounding box.

        For metric models: samples the inner 50% of the bounding box
        with trimmed percentile for noise robustness.
        For relative/unavailable: falls back to bbox-height heuristic.
        """
        if depth_map is not None and self._is_metric:
            # ── Metric depth: robust sampling ──
            h, w = depth_map.shape[:2]

            if bbox is not None:
                # Use inner 50% of bounding box for robust sampling
                x1b, y1b, x2b, y2b = bbox
                bw, bh = x2b - x1b, y2b - y1b
                # Shrink to inner 50%
                margin_x = bw * 0.25
                margin_y = bh * 0.25
                ix1 = int(np.clip(x1b + margin_x, 0, w - 1))
                iy1 = int(np.clip(y1b + margin_y, 0, h - 1))
                ix2 = int(np.clip(x2b - margin_x, 1, w))
                iy2 = int(np.clip(y2b - margin_y, 1, h))
                patch = depth_map[iy1:iy2, ix1:ix2]
            else:
                # Fallback: patch around center
                p = self.config.sample_patch
                ix = int(np.clip(cx, 0, w - 1))
                iy = int(np.clip(cy, 0, h - 1))
                y1 = max(0, iy - p)
                y2 = min(h, iy + p)
                x1 = max(0, ix - p)
                x2 = min(w, ix + p)
                patch = depth_map[y1:y2, x1:x2]

            if patch.size > 0:
                # Trimmed percentile: remove top/bottom 15% outliers
                flat = patch.flatten()
                if len(flat) > 10:
                    low = np.percentile(flat, 15)
                    high = np.percentile(flat, 85)
                    trimmed = flat[(flat >= low) & (flat <= high)]
                    if len(trimmed) > 0:
                        distance = float(np.median(trimmed))
                    else:
                        distance = float(np.median(flat))
                else:
                    distance = float(np.median(flat))
                return float(np.clip(distance, 0.2, 80.0))

        # ── Fallback: bbox-height heuristic ──
        return self._bbox_distance(bbox_h, class_name)

    def _bbox_distance(self, bbox_h: float, class_name: str) -> float:
        """Estimate distance from bounding box height and known object sizes."""
        known_h = self.config.known_heights.get(class_name, 1.0)
        if bbox_h > 10:
            distance = (known_h * self.config.focal_length_px) / bbox_h
        else:
            distance = 20.0
        return float(np.clip(distance, 0.3, 50.0))

    def get_visualization(self, depth_map: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Convert depth map to colorized visualization."""
        if depth_map is None:
            return None

        if self._is_metric:
            # Metric: normalize for display (0m=bright, 20m=dark)
            vis = np.clip(depth_map / 20.0, 0, 1)
            vis = (vis * 255).astype(np.uint8)
        else:
            vis = (depth_map * 255).astype(np.uint8)

        return cv2.applyColorMap(vis, cv2.COLORMAP_MAGMA)

    @property
    def inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def is_available(self) -> bool:
        return self._model_loaded

    @property
    def is_metric(self) -> bool:
        return self._model_loaded and self._is_metric
