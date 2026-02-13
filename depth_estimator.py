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
from PIL import Image

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
        self.processor = None
        self.transform = None
        self.model_type = "midas" # "midas" or "transformers"
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
        """Load depth model based on config."""
        try:
            logger.info(f"Loading Depth Model: {self.config.model_name}")

            if "DepthAnything" in self.config.model_name:
                self._load_depth_anything_v2()
            else:
                self._load_midas()

            self._model_loaded = True
            logger.info(f"Depth model loaded on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            import traceback
            traceback.print_exc()
            self._model_loaded = False

    def _load_depth_anything_v2(self):
        """Load Depth Anything V2 via Transformers."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        
        self.model_type = "transformers"
        model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        
        logger.info(f"Loading {model_id}...")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device).eval()
        
        # Warmup
        dummy = np.zeros((self.config.input_size, self.config.input_size, 3), dtype=np.uint8)
        self._run_transformers(dummy)

    def _load_midas(self):
        """Load MiDaS depth model via Torch Hub."""
        self.model_type = "midas"
        
        # Use torch.hub for MiDaS
        self.model = torch.hub.load("intel-isl/MiDaS", self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if "small" in self.config.model_name.lower():
            self.transform = midas_transforms.small_transform
        else:
            self.transform = midas_transforms.dpt_transform

        # Warm up
        dummy = np.zeros((self.config.input_size, self.config.input_size, 3), dtype=np.uint8)
        self._run_midas(dummy)

    @torch.no_grad()
    def _run_midas(self, frame: np.ndarray) -> np.ndarray:
        """Run MiDaS model on a frame."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        prediction = self.model(input_batch)
        
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        return prediction.cpu().numpy()

    @torch.no_grad()
    def _run_transformers(self, frame: np.ndarray) -> np.ndarray:
        """Run Depth Anything V2 via Transformers."""
        # Convert BGR to RGB PIL Image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        return prediction.cpu().numpy()

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Get depth map for the frame.
        Returns normalized depth map (0 = close, 1 = far) or None.
        """
        self._frame_counter += 1

        if not self._model_loaded:
            return None

        # Run model every N frames
        if self._frame_counter % self.config.run_every_n_frames != 0:
            return self._last_depth_map

        t_start = time.perf_counter()
        
        if self.model_type == "transformers":
            raw_depth = self._run_transformers(frame)
        else:
            raw_depth = self._run_midas(frame)
            
        self._last_inference_ms = (time.perf_counter() - t_start) * 1000

        # Normalize to 0-1
        d_min, d_max = raw_depth.min(), raw_depth.max()
        if d_max - d_min > 1e-6:
            depth_map = (raw_depth - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(raw_depth)

        # Invert so 0 = close, 1 = far
        # Ensure we check the model output type. 
        # MiDaS is inverse depth (disp), so higher = closer.
        # Depth Anything is relative depth, usually higher = closer too (disparity-like).
        # We want 0 = close, 1 = far.
        
        # Both models generally output "disparity" or "inverse depth".
        # So Max Value = Closest.
        # We normalized it to 0..1 where 1 is Max (Closest).
        # So we invert it: 1 - depth_map makes 0 = Closest (Max).
        
        # Wait, if 1.0 was Close, then 1-1.0 = 0.0 = Close. Correct.
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
