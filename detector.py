"""
Drishtimarga — Object Detection & Tracking
YOLO11 detection with ByteTrack persistent tracking.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

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
    """YOLO11-based detector with integrated ByteTrack tracking."""

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
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.info("No CUDA — running on CPU (expect slower performance)")
        else:
            self.device = self.config.device

    def _load_model(self):
        logger.info(f"Loading YOLO model: {self.config.model_name}")
        self.model = YOLO(self.config.model_name)
        # Warm up the model
        dummy = np.zeros((self.config.input_size, self.config.input_size, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False, device=self.device)
        logger.info("YOLO model loaded and warmed up")

    def detect_and_track(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection + tracking on a single frame.
        Returns list of Detection objects with persistent track IDs.
        """
        self._frame_count += 1
        t_start = time.perf_counter()

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

        self._last_inference_ms = (time.perf_counter() - t_start) * 1000
        timestamp = time.time()

        detections: List[Detection] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                if box.id is None:
                    continue

                track_id = int(box.id[0].item())
                class_id = int(box.cls[0].item())
                class_name = results[0].names[class_id]
                confidence = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1

                detections.append(Detection(
                    track_id=track_id,
                    class_name=class_name,
                    class_id=class_id,
                    confidence=confidence,
                    bbox=xyxy,
                    center=np.array([cx, cy]),
                    bbox_width=w,
                    bbox_height=h,
                    timestamp=timestamp,
                ))

        return detections

    @property
    def inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def class_names(self) -> dict:
        return self.model.names
