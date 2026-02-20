"""
Drishtimarga v2 — Configuration
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
    # yolo11x.pt = best accuracy (56.1 mAP); swap to yolo11n.pt for speed
    model_name: str = "yolo11x.pt"
    confidence: float = 0.40
    iou_threshold: float = 0.50
    input_size: int = 640               # 640 for best accuracy with x model
    device: str = "auto"                # "auto", "cuda", "cpu"
    tracker: str = "bytetrack.yaml"
    half_precision: bool = True


@dataclass
class DepthConfig:
    # Metric model outputs real meters — no calibration needed
    # Options:  "metric-outdoor-small", "metric-outdoor-base",
    #           "metric-outdoor-large", "metric-indoor-small", etc.
    #           "relative-small" for old non-metric fallback
    model_name: str = "metric-outdoor-large"
    enabled: bool = True
    run_every_n_frames: int = 2         # run depth every Nth frame; reuse last otherwise
    device: str = "auto"
    # Depth sampling: patch size (px) around bbox center for median sampling
    sample_patch: int = 16
    # Temporal EMA alpha for smoothing depth maps across frames (0-1, higher = more current frame)
    temporal_alpha: float = 0.6
    # Fallback bbox-based depth (used when depth model is unavailable)
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
    stale_timeout: float = 2.5          # seconds before track considered gone
    # Per-object distance EMA smoothing factor (0-1, lower = more smoothing)
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
class AnnouncementConfig:
    """Controls the smart announcement memory to prevent repetition."""
    # State transition thresholds
    stable_after_seconds: float = 6.0       # object must be unchanged for this long → STABLE
    depart_after_seconds: float = 2.5       # not seen for this long → DEPARTING
    gone_after_seconds: float = 5.0         # not seen for this long → removed from memory

    # Re-announcement triggers (object must change by this much to re-announce)
    distance_change_near: float = 0.8       # meters (for objects < 3m)
    distance_change_far: float = 2.0        # meters (for objects >= 3m)
    position_change: bool = True            # re-announce when left↔center↔right changes
    velocity_change_threshold: float = 0.8  # m/s change to re-announce

    # Adaptive cooldowns
    cooldown_new: float = 0.0               # no cooldown for first announcement
    cooldown_active: float = 7.0            # seconds between updates for active objects
    cooldown_stable: float = 60.0           # very long cooldown — object hasn't changed
    cooldown_critical: float = 2.5          # short cooldown for critical threats

    # Limits
    max_announcements_per_cycle: int = 2    # max spoken per processing cycle
    max_concurrent_objects: int = 6         # only track the N most important objects for audio

    # Departure
    announce_departures: bool = True
    departure_min_lifetime: float = 5.0     # only announce departure if object was seen > 5s

    # Scene summaries
    scene_summary_interval: float = 20.0
    scene_summary_only_on_change: bool = True


@dataclass
class AudioConfig:
    rate: int = 185
    volume: float = 1.0
    urgent_prefix: str = "Warning!"
    critical_prefix: str = "DANGER!"


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
    announcement: AnnouncementConfig = field(default_factory=AnnouncementConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
