"""
Drishtimarga v2 — Modal GPU Server

Runs YOLO detection + Depth Anything V2 on a cloud GPU.
Exposes a FastAPI endpoint that accepts JPEG frames and returns
detection/depth/threat JSON results.

Deploy:
    modal deploy modal_server.py

Test locally:
    modal serve modal_server.py

The local client (local_client.py) captures frames from ESP32-CAM
and POSTs them here.
"""

import io
import logging
import time
from typing import Dict, List, Optional

import modal

try:
    import torch
except ImportError:
    torch = None  # Available inside the Modal container

logger = logging.getLogger("drishtimarga-modal")

# ── Modal App & Image ──

app = modal.App("drishtimarga-v2")

# Build the container image with all ML dependencies
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install(
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
        "Pillow>=10.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.3.0",
        "transformers>=4.36.0",
        "fastapi[standard]",
    )
)

# ── Model weights volume (cached across cold starts) ──
model_volume = modal.Volume.from_name("drishtimarga-models", create_if_missing=True)

# ── HuggingFace model IDs ──
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
YOLO_MODEL = "yolo11x.pt"


@app.cls(
    image=gpu_image,
    gpu="H100",                       # T4 is cheapest; upgrade to A10G/A100 if needed
    timeout=300,
    scaledown_window=120,           # keep warm for 2 min between requests
    volumes={"/models": model_volume},
)
@modal.concurrent(max_inputs=4)
class DrishtimargaInference:
    """
    Stateful Modal class that loads models once on container start,
    then serves inference requests via a FastAPI web endpoint.
    """

    @modal.enter()
    def load_models(self):
        """Called once when the container starts — load YOLO + Depth models."""
        import numpy as np
        import torch
        from ultralytics import YOLO
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")

        # ── YOLO ──
        logger.info(f"Loading YOLO: {YOLO_MODEL}")
        self.yolo = YOLO(YOLO_MODEL)
        # Warm up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.yolo.predict(dummy, verbose=False, device=self.device)
        logger.info("YOLO ready")

        # ── Depth Anything V2 Metric ──
        logger.info(f"Loading depth: {DEPTH_MODEL_ID}")
        self.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
        self.depth_model.to(self.device).eval()
        if self.device == "cuda":
            self.depth_model = self.depth_model.half()
            logger.info("Depth using FP16")
        # Temporal EMA state for depth smoothing
        self._prev_depth_map = None
        self._temporal_alpha = 0.6

        # Warm up
        self._run_depth(dummy)
        logger.info("Depth ready")

        # Tracking state (ByteTrack needs persist=True across frames)
        self._track_counter = 0

    @torch.no_grad()
    def _run_depth(self, frame_bgr):
        """Run depth inference, return metric depth map in meters."""
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        inputs = self.depth_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}

        outputs = self.depth_model(**inputs)
        depth = outputs.predicted_depth

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1).float(),
            size=(frame_bgr.shape[0], frame_bgr.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        import numpy as np
        depth = np.clip(depth, 0.1, 80.0)

        # Temporal EMA smoothing
        if (self._prev_depth_map is not None
                and self._prev_depth_map.shape == depth.shape):
            alpha = self._temporal_alpha
            depth = alpha * depth + (1.0 - alpha) * self._prev_depth_map
        self._prev_depth_map = depth.copy()

        return depth

    def _get_distance_at_bbox(self, depth_map, bbox, cx, cy, bbox_h, class_name):
        """Sample metric distance from depth map for a bounding box."""
        import numpy as np

        KNOWN_HEIGHTS = {
            "person": 1.7, "car": 1.5, "bus": 2.8, "truck": 3.0,
            "motorcycle": 1.1, "bicycle": 1.0, "dog": 0.5, "cat": 0.3,
            "chair": 0.8, "bottle": 0.25, "cup": 0.12, "laptop": 0.25,
            "tv": 0.5, "cell phone": 0.14, "backpack": 0.5,
        }

        if depth_map is not None:
            h, w = depth_map.shape[:2]
            x1b, y1b, x2b, y2b = bbox
            bw, bh = x2b - x1b, y2b - y1b
            margin_x = bw * 0.25
            margin_y = bh * 0.25
            ix1 = int(np.clip(x1b + margin_x, 0, w - 1))
            iy1 = int(np.clip(y1b + margin_y, 0, h - 1))
            ix2 = int(np.clip(x2b - margin_x, 1, w))
            iy2 = int(np.clip(y2b - margin_y, 1, h))
            patch = depth_map[iy1:iy2, ix1:ix2]

            if patch.size > 0:
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

        # Fallback: bbox-height heuristic
        known_h = KNOWN_HEIGHTS.get(class_name, 1.0)
        if bbox_h > 10:
            distance = (500.0 * known_h) / bbox_h
        else:
            distance = 20.0
        return float(np.clip(distance, 0.3, 50.0))

    def _process_frame(self, frame_bgr):
        """
        Full inference pipeline on one frame.
        Returns list of detection dicts with distances.
        """
        import numpy as np
        import time as _time

        t0 = _time.perf_counter()
        h_frame, w_frame = frame_bgr.shape[:2]

        # ── YOLO detect + track ──
        self._track_counter += 1
        results = self.yolo.track(
            frame_bgr,
            persist=True,
            conf=0.40,
            iou=0.50,
            imgsz=640,
            tracker="bytetrack.yaml",
            half=(self.device == "cuda"),
            device=self.device,
            verbose=False,
        )
        det_ms = (_time.perf_counter() - t0) * 1000

        # ── Depth ──
        t1 = _time.perf_counter()
        depth_map = self._run_depth(frame_bgr)
        depth_ms = (_time.perf_counter() - t1) * 1000

        # ── Build results ──
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                tid = int(box.id[0].item())
                cid = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                conf = float(box.conf[0].item())
                class_name = results[0].names[cid]

                distance = self._get_distance_at_bbox(
                    depth_map, [x1, y1, x2, y2], cx, cy, bbox_h, class_name)

                rel_x = cx / w_frame if w_frame > 0 else 0.5
                if rel_x < 0.33:
                    position = "left"
                elif rel_x > 0.67:
                    position = "right"
                else:
                    position = "center"

                detections.append({
                    "track_id": tid,
                    "class_name": class_name,
                    "class_id": cid,
                    "confidence": round(conf, 3),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "center": [round(cx, 1), round(cy, 1)],
                    "bbox_width": round(bbox_w, 1),
                    "bbox_height": round(bbox_h, 1),
                    "distance": round(distance, 2),
                    "position": position,
                    "rel_x": round(rel_x, 3),
                })

        total_ms = (_time.perf_counter() - t0) * 1000
        return {
            "detections": detections,
            "frame_shape": [h_frame, w_frame],
            "detection_ms": round(det_ms, 1),
            "depth_ms": round(depth_ms, 1),
            "total_ms": round(total_ms, 1),
            "depth_metric": True,
        }

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def infer(self, request: dict):
        """
        Accept a base64-encoded JPEG frame, run full pipeline, return JSON.

        Request body:
            {"frame_b64": "<base64 JPEG>"}

        Response:
            {
                "detections": [...],
                "frame_shape": [H, W],
                "detection_ms": ...,
                "depth_ms": ...,
                "total_ms": ...,
                "depth_metric": true
            }
        """
        import base64
        import cv2
        import numpy as np

        frame_b64 = request.get("frame_b64", "")
        if not frame_b64:
            return {"error": "No frame_b64 in request body"}

        # Decode JPEG
        jpg_bytes = base64.b64decode(frame_b64)
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Failed to decode JPEG frame"}

        return self._process_frame(frame)

    @modal.fastapi_endpoint(method="GET", docs=True)
    async def health(self):
        """Health check endpoint."""
        return {"status": "ok", "model": "drishtimarga-v2", "gpu": self.device}
