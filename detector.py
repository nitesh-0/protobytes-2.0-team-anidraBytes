"""
Drishtimarga — Object Detection & Tracking (FIXED VERSION)
YOLO detection with ByteTrack persistent tracking.
Added class filtering and track persistence to reduce false positives.
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
    """
    YOLO-based detector with integrated ByteTrack tracking.
    
    FIXES:
    1. Class filtering to remove irrelevant objects
    2. Track persistence check to reduce hallucinations
    3. Confidence threshold raised to 0.65
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._resolve_device()
        self._load_model()
        self._frame_count = 0
        self._last_inference_ms = 0.0
        
        # NEW: Track persistence monitoring
        self._track_first_seen = {}  # track_id -> first_frame_number
        self._track_frame_count = {}  # track_id -> number of frames seen

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
        
        # Log class filtering
        logger.info(f"Filtering to {len(self.config.relevant_classes)} relevant classes")
        logger.info(f"Requiring {self.config.min_track_frames} frames of persistence")

    def detect_and_track(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection + tracking on a single frame.
        Returns list of Detection objects with persistent track IDs.
        
        IMPROVED: Now filters by class relevance and track persistence.
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
        current_track_ids = set()
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                if box.id is None:
                    continue

                track_id = int(box.id[0].item())
                class_id = int(box.cls[0].item())
                class_name = results[0].names[class_id]
                confidence = float(box.conf[0].item())
                
                # FIX 1: Class filtering
                if class_name not in self.config.relevant_classes:
                    continue  # Skip irrelevant objects
                
                # Track persistence monitoring
                current_track_ids.add(track_id)
                if track_id not in self._track_first_seen:
                    self._track_first_seen[track_id] = self._frame_count
                    self._track_frame_count[track_id] = 1
                    # Log new track
                    logger.debug(f"New track {track_id}: {class_name} (conf={confidence:.2f})")
                else:
                    self._track_frame_count[track_id] += 1
                
                # FIX 2: Track persistence check
                frames_seen = self._track_frame_count[track_id]
                if frames_seen < self.config.min_track_frames:
                    # Don't include in detections until it's persistent
                    # This drastically reduces hallucinations
                    continue
                
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

        # Cleanup stale tracks
        stale_tracks = set(self._track_first_seen.keys()) - current_track_ids
        for track_id in stale_tracks:
            del self._track_first_seen[track_id]
            del self._track_frame_count[track_id]

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
    
    def get_track_info(self, track_id: int) -> dict:
        """Get tracking statistics for a specific track ID."""
        return {
            'first_frame': self._track_first_seen.get(track_id, -1),
            'frame_count': self._track_frame_count.get(track_id, 0),
            'age_frames': self._frame_count - self._track_first_seen.get(track_id, self._frame_count)
        }