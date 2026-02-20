"""
Drishtimarga — Configuration
All tunable parameters in one place.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CameraConfig:
    source: int = 0                     # 0 = default webcam, or URL string for ESP32
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1                # keep only latest frame


@dataclass
class DetectorConfig:
    # YOLO11m for best accuracy/speed trade-off
    model_name: str = "yolo11m.pt"
    confidence: float = 0.35
    iou_threshold: float = 0.5
    input_size: int = 640               # 640 for best accuracy
    device: str = "auto"                # "auto", "cuda", "cpu"
    tracker: str = "bytetrack.yaml"     # or "botsort.yaml"
    half_precision: bool = True         # FP16 on GPU



@dataclass
class DepthConfig:
    model_name: str = "base"            # "small", "base", or "large" (Depth Anything V2)
    enabled: bool = True
    # skip frames for speed (1 = every frame)
    run_every_n_frames: int = 3
    input_size: int = 518               # Depth Anything V2 native size
    device: str = "auto"


@dataclass
class SpatialConfig:
    grid_size: int = 20                 # 20x20 occupancy grid
    cell_size: float = 0.5             # meters per cell → 10m x 10m coverage
    track_history_length: int = 15      # frames of history per tracked object
    stale_timeout: float = 3.0          # seconds before object is considered gone
    # Distance estimation (focal length calibration)
    focal_length_px: float = 500.0      # approximate for 640x480 webcam
    known_heights: Dict[str, float] = field(default_factory=lambda: {
        "person": 1.7, "car": 1.5, "bus": 2.8, "truck": 3.0,
        "motorcycle": 1.1, "bicycle": 1.0, "dog": 0.5, "cat": 0.3,
        "chair": 0.8, "bottle": 0.25, "cup": 0.12, "laptop": 0.25,
        "tv": 0.5, "cell phone": 0.14, "backpack": 0.5,
    })


@dataclass
class ThreatConfig:
    # Danger multipliers by object class
    danger_weights: Dict[str, float] = field(default_factory=lambda: {
        "bus": 5.0, "truck": 5.0, "car": 4.0, "motorcycle": 3.5,
        "bicycle": 2.5, "person": 1.5, "dog": 2.0, "cat": 1.0,
        "skateboard": 2.0, "train": 6.0,
        # Static obstacles
        "chair": 1.0, "bench": 1.0, "fire hydrant": 1.5,
        "stop sign": 0.5, "traffic light": 0.3,
    })
    default_danger: float = 1.0
    # Thresholds
    critical_ttc: float = 3.0           # seconds — immediate alert
    close_distance: float = 1.5         # meters — "very close" warning
    approach_speed_threshold: float = 0.3  # m/s to count as "approaching"
    center_multiplier: float = 1.5      # objects in center of path are more dangerous


@dataclass
class NavigationConfig:
    """Parameters controlling spoken navigation guidance."""
    # Distance thresholds (meters)
    immediate_zone: float = 1.0         # < 1m  → STOP / urgent dodge
    close_zone: float = 2.5            # < 2.5m → active avoidance advice
    medium_zone: float = 5.0           # < 5m  → heads-up with direction
    far_zone: float = 10.0             # < 10m → mention existence
    # Path-clear announcements
    path_clear_delay: float = 2.0      # seconds of no detections before "path clear"
    path_clear_repeat: float = 10.0    # seconds between repeated "path clear" messages
    # Guidance verbosity
    guidance_interval: float = 12.0    # seconds between full navigation updates
    announce_departures: bool = True   # tell user when objects leave the scene


@dataclass
class AudioConfig:
    rate: int = 190                     # speech rate (words per minute)
    volume: float = 1.0                 # 0.0 to 1.0
    cooldown: float = 8.0              # seconds between re-announcing same object
    class_cooldown: float = 6.0        # seconds between re-announcing same class
    distance_change_threshold: float = 0.8  # meters change to re-announce
    max_announcements_per_cycle: int = 2
    scene_summary_interval: float = 20.0  # seconds between scene overviews
    min_speak_gap: float = 3.0         # minimum seconds between any two announcements
    urgent_prefix: str = "Warning!"
    critical_prefix: str = "DANGER!"


@dataclass
class DisplayConfig:
    show_video: bool = True             # show annotated video window
    show_depth: bool = False            # show depth map window
    show_grid: bool = False             # show occupancy grid
    annotation_thickness: int = 2
    font_scale: float = 0.5


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    threat: ThreatConfig = field(default_factory=ThreatConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
