"""
Drishtimarga v3 — Configuration
All tunable parameters in one place.
"""

from dataclasses import dataclass, field
from typing import Dict, Union


@dataclass
class CameraConfig:
    source: Union[int, str] = 0         # 0 = webcam, or ESP32-CAM URL string
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1


@dataclass
class DetectorConfig:
    # yolo11x.pt = best accuracy (56.1 mAP)
    model_name: str = "yolo11x.pt"
    confidence: float = 0.40
    iou_threshold: float = 0.50
    input_size: int = 640
    device: str = "auto"                # "auto", "cuda", "cpu"
    tracker: str = "bytetrack.yaml"
    half_precision: bool = True


@dataclass
class DepthConfig:
    # Metric model outputs real meters — no calibration needed
    # Using the LARGE model for best accuracy
    model_name: str = "metric-outdoor-large"
    enabled: bool = True
    run_every_n_frames: int = 2
    device: str = "auto"
    sample_patch: int = 16
    temporal_alpha: float = 0.6
    focal_length_px: float = 500.0
    known_heights: Dict[str, float] = field(default_factory=lambda: {
        "person": 1.7, "car": 1.5, "bus": 2.8, "truck": 3.0,
        "motorcycle": 1.1, "bicycle": 1.0, "dog": 0.5, "cat": 0.3,
        "chair": 0.8, "bottle": 0.25, "cup": 0.12, "laptop": 0.25,
        "tv": 0.5, "cell phone": 0.14, "backpack": 0.5,
    })


@dataclass
class SpatialConfig:
    grid_size: int = 20
    cell_size: float = 0.5
    track_history_length: int = 20
    stale_timeout: float = 2.5
    distance_smoothing_alpha: float = 0.3


@dataclass
class ThreatConfig:
    danger_weights: Dict[str, float] = field(default_factory=lambda: {
        "bus": 5.0, "truck": 5.0, "car": 4.0, "motorcycle": 3.5,
        "bicycle": 2.5, "person": 1.5, "dog": 2.0, "cat": 1.0,
        "skateboard": 2.0, "train": 6.0,
        "chair": 1.0, "bench": 1.0, "fire hydrant": 1.5,
        "stop sign": 0.5, "traffic light": 0.3,
    })
    default_danger: float = 1.0
    critical_ttc: float = 3.0
    close_distance: float = 1.5
    approach_speed_threshold: float = 0.3
    center_multiplier: float = 1.5


@dataclass
class DisplayConfig:
    show_video: bool = True
    show_depth: bool = False
    annotation_thickness: int = 2
    font_scale: float = 0.5


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    threat: ThreatConfig = field(default_factory=ThreatConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
