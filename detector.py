"""
Drishtimarga v2 — Object Detection & Tracking
Supports YOLOv11x (best accuracy) down to YOLOv8n (fastest).
"""

import logging
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from ultralytics import YOLO

from config import DetectorConfig

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detected object with tracking info."""
    track_id: int
    class_name: str
    class_id: int
    confidence: float
    bbox: np.ndarray          # [x1, y1, x2, y2]
    center: np.ndarray        # [cx, cy]
    bbox_width: float
    bbox_height: float
    timestamp: float

    @property
    def area(self) -> float:
        return self.bbox_width * self.bbox_height


class ObjectDetector:
    """
    YOLO detector with integrated ByteTrack tracking.

    Model ladder (accuracy vs speed):
        yolo11x.pt  → 56.1 mAP, ~5ms TRT   ← default (best accuracy)
        yolo11l.pt  → 53.4 mAP, ~3ms TRT
        yolo11m.pt  → 51.5 mAP, ~2ms TRT
        yolo11s.pt  → 47.0 mAP, ~1.5ms TRT
        yolo11n.pt  → 39.5 mAP, ~1ms TRT   ← fastest
        yolov8x.pt  → 53.9 mAP
        yolov8n.pt  → 37.3 mAP
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._resolve_device()
        self._load_model()
        self._frame_count = 0
        self._last_inference_ms = 0.0

    def _resolve_device(self):
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_mem / 1e9
                logger.info(f"CUDA: {gpu_name} ({vram:.1f} GB)")
            else:
                self.device = "cpu"
                logger.info("No CUDA — running on CPU")
        else:
            self.device = self.config.device

    def _load_model(self):
        model_name = self.config.model_name
        logger.info(f"Loading detection model: {model_name}")
        logger.info(f"  Input size: {self.config.input_size}, "
                     f"Conf: {self.config.confidence}, "
                     f"Device: {self.device}")

        self.model = YOLO(model_name)

        # Warm up with dummy frame
        dummy = np.zeros(
            (self.config.input_size, self.config.input_size, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False, device=self.device)
        logger.info(f"Detection model ready ({model_name})")

    def detect_and_track(self, frame: np.ndarray) -> List[Detection]:
        """Run detection + tracking. Returns detections with persistent IDs."""
        self._frame_count += 1
        t0 = time.perf_counter()

        results = self.model.track(
            frame,
            persist=True,
            conf=self.config.confidence,
            iou=self.config.iou_threshold,
            imgsz=self.config.input_size,
            tracker=self.config.tracker,
            half=self.config.half_precision and self.device == "cuda",
            device=self.device,
            verbose=False,
        )

        self._last_inference_ms = (time.perf_counter() - t0) * 1000
        now = time.time()

        detections: List[Detection] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue

                tid = int(box.id[0].item())
                cid = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                detections.append(Detection(
                    track_id=tid,
                    class_name=results[0].names[cid],
                    class_id=cid,
                    confidence=float(box.conf[0].item()),
                    bbox=np.array([x1, y1, x2, y2]),
                    center=np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
                    bbox_width=x2 - x1,
                    bbox_height=y2 - y1,
                    timestamp=now,
                ))

        return detections

    @property
    def inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def frame_count(self) -> int:
        return self._frame_count
