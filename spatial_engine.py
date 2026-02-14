"""
Drishtimarga v2 — Spatial Understanding & Threat Scoring
Tracks objects in space, computes velocities, and scores threats.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import SpatialConfig, ThreatConfig
from detector import Detection
from depth_estimator import DepthEstimator

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    SKIP = 4


class Position:
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass
class TrackedObject:
    """Persistent state for a single tracked object."""
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    # History: [(distance_m, rel_x, rel_y, timestamp)]
    history: list = field(default_factory=list)

    @property
    def distance(self) -> float:
        return self.history[-1][0] if self.history else float("inf")

    @property
    def rel_x(self) -> float:
        return self.history[-1][1] if self.history else 0.5

    @property
    def position(self) -> str:
        rx = self.rel_x
        if rx < 0.33:
            return Position.LEFT
        elif rx > 0.67:
            return Position.RIGHT
        return Position.CENTER

    @property
    def approach_velocity(self) -> float:
        """
        Approach velocity (m/s). Positive = getting closer.
        Uses linear regression over recent history for smooth estimate.
        """
        if len(self.history) < 3:
            return 0.0
        recent = self.history[-min(10, len(self.history)):]
        distances = [h[0] for h in recent]
        times = [h[3] for h in recent]
        n = len(distances)
        st = sum(times)
        sd = sum(distances)
        std = sum(t * d for t, d in zip(times, distances))
        stt = sum(t * t for t in times)
        denom = n * stt - st * st
        if abs(denom) < 1e-10:
            return 0.0
        slope = (n * std - st * sd) / denom
        return -slope  # negative slope means distance decreasing = approaching

    @property
    def time_to_collision(self) -> float:
        vel = self.approach_velocity
        if vel <= 0.05:
            return float("inf")
        d = self.distance
        return d / vel if d > 0 else 0.0

    @property
    def is_moving(self) -> bool:
        return abs(self.approach_velocity) > 0.2

    @property
    def age(self) -> float:
        return self.last_seen - self.first_seen

    def is_stale(self, timeout: float) -> bool:
        return (time.time() - self.last_seen) > timeout


@dataclass
class ThreatAssessment:
    """Threat analysis result for a tracked object."""
    track_id: int
    class_name: str
    distance: float
    position: str
    approach_velocity: float
    time_to_collision: float
    threat_score: float
    priority: Priority
    is_new: bool
    message: str


class SpatialEngine:
    """
    Spatial understanding engine.
    Merges detection + metric depth → distance, velocity, threat scores.
    """

    def __init__(self, spatial_cfg: SpatialConfig, threat_cfg: ThreatConfig):
        self.s_cfg = spatial_cfg
        self.t_cfg = threat_cfg
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.occupancy_grid = np.zeros(
            (spatial_cfg.grid_size, spatial_cfg.grid_size), dtype=np.float32)
        self._prev_track_ids: set = set()

    def update(self, detections: List[Detection],
               depth_map: Optional[np.ndarray],
               frame_shape: Tuple[int, int],
               depth_estimator: DepthEstimator) -> List[ThreatAssessment]:
        """
        Update spatial state with new detections.
        Now takes the depth_estimator directly so it can call get_distance_at_bbox.
        """
        now = time.time()
        h_frame, w_frame = frame_shape[:2]
        assessments: List[ThreatAssessment] = []

        # Decay occupancy grid
        self.occupancy_grid *= 0.93

        active_ids = set()

        for det in detections:
            tid = det.track_id
            active_ids.add(tid)

            # ── Get metric distance ──
            distance = depth_estimator.get_distance_at_bbox(
                depth_map,
                cx=det.center[0], cy=det.center[1],
                bbox=det.bbox,
                bbox_h=det.bbox_height,
                class_name=det.class_name,
            )

            # Relative position in frame (0=left, 1=right)
            rel_x = det.center[0] / w_frame if w_frame > 0 else 0.5
            rel_y = det.center[1] / h_frame if h_frame > 0 else 0.5

            # Create or update tracked object
            is_new = tid not in self.tracked_objects
            if is_new:
                self.tracked_objects[tid] = TrackedObject(
                    track_id=tid,
                    class_name=det.class_name,
                    first_seen=now,
                    last_seen=now,
                )

            obj = self.tracked_objects[tid]
            obj.last_seen = now
            obj.class_name = det.class_name

            # Per-object distance smoothing (EMA)
            # Prevents noisy monocular depth from jittering the announced distance
            if obj.history:
                prev_dist = obj.history[-1][0]
                alpha = self.s_cfg.distance_smoothing_alpha
                distance = alpha * distance + (1.0 - alpha) * prev_dist

            obj.history.append((distance, rel_x, rel_y, now))
            if len(obj.history) > self.s_cfg.track_history_length:
                obj.history = obj.history[-self.s_cfg.track_history_length:]

            # Update occupancy grid
            self._update_grid(distance, rel_x)

            # Score threat
            assessment = self._assess_threat(obj, is_new)
            assessments.append(assessment)

        # Track which IDs just departed
        self._prev_track_ids = active_ids

        # Clean stale tracks
        stale = [tid for tid, o in self.tracked_objects.items()
                 if o.is_stale(self.s_cfg.stale_timeout) and tid not in active_ids]
        for tid in stale:
            del self.tracked_objects[tid]

        assessments.sort(key=lambda a: a.threat_score, reverse=True)
        return assessments

    def _update_grid(self, distance: float, rel_x: float):
        gs = self.s_cfg.grid_size
        cs = self.s_cfg.cell_size
        gz = int(np.clip(distance / cs, 0, gs - 1))
        gx = int(np.clip(rel_x * gs, 0, gs - 1))
        self.occupancy_grid[gz, gx] = 1.0

    def _assess_threat(self, obj: TrackedObject, is_new: bool) -> ThreatAssessment:
        distance = obj.distance
        velocity = obj.approach_velocity
        ttc = obj.time_to_collision
        position = obj.position

        # ── Threat score ──
        score = 0.0
        if distance < 15:
            score += max(0, (15 - distance)) * 5
        if velocity > self.t_cfg.approach_speed_threshold:
            score += velocity * 25
        danger_w = self.t_cfg.danger_weights.get(
            obj.class_name, self.t_cfg.default_danger)
        score *= danger_w
        if position == Position.CENTER:
            score *= self.t_cfg.center_multiplier
        if ttc < self.t_cfg.critical_ttc:
            score *= 2.0

        # ── Priority ──
        if ttc < self.t_cfg.critical_ttc and velocity > 0.5:
            priority = Priority.CRITICAL
        elif (is_new and distance < 5.0) or velocity > self.t_cfg.approach_speed_threshold:
            priority = Priority.HIGH
        elif is_new or distance < self.t_cfg.close_distance:
            priority = Priority.NORMAL
        else:
            priority = Priority.LOW

        # ── Message ──
        message = self._build_message(obj, distance, position, velocity, ttc, priority)

        return ThreatAssessment(
            track_id=obj.track_id,
            class_name=obj.class_name,
            distance=distance,
            position=position,
            approach_velocity=velocity,
            time_to_collision=ttc,
            threat_score=score,
            priority=priority,
            is_new=is_new,
            message=message,
        )

    def _quantize_distance_str(self, distance: float) -> str:
        """Quantize distance for announcement to avoid micro-change chatter."""
        if distance < 1.0:
            return "very close"
        elif distance < 2.0:
            q = round(distance * 2) / 2  # 0.5m buckets
            return f"{q:.1f} meters"
        elif distance < 5.0:
            q = round(distance)  # 1m buckets
            return f"{int(q)} meters"
        elif distance < 10.0:
            q = round(distance)  # 1m buckets
            return f"about {int(q)} meters"
        else:
            q = round(distance / 2) * 2  # 2m buckets
            return f"about {int(q)} meters"

    def _build_message(self, obj, distance, position, velocity, ttc, priority):
        cls = obj.class_name
        d = self._quantize_distance_str(distance)
        if priority == Priority.CRITICAL:
            return f"DANGER! {cls} approaching fast! {d}, {position}!"
        if velocity > self.t_cfg.approach_speed_threshold:
            return f"Warning, {cls} approaching, {d}, {position}"
        if distance < self.t_cfg.close_distance:
            return f"{cls}, very close, {position}"
        return f"{cls}, {d}, {position}"

    def get_path_status(self) -> str:
        gs = self.s_cfg.grid_size
        mid = gs // 2
        ahead = self.occupancy_grid[:5, mid - 2:mid + 2]
        if np.sum(ahead) < 0.5:
            return "Path ahead is clear"
        left_d = np.sum(self.occupancy_grid[:5, :mid])
        right_d = np.sum(self.occupancy_grid[:5, mid:])
        if left_d < right_d:
            return "Path blocked ahead, left side is more open"
        return "Path blocked ahead, right side is more open"

    def get_scene_summary(self) -> str:
        active = [o for o in self.tracked_objects.values()
                  if not o.is_stale(self.s_cfg.stale_timeout)]
        if not active:
            return "Area appears clear"
        left = [o for o in active if o.position == Position.LEFT]
        center = [o for o in active if o.position == Position.CENTER]
        right = [o for o in active if o.position == Position.RIGHT]
        parts = []
        if center:
            names = ", ".join(sorted(set(o.class_name for o in center[:2])))
            parts.append(f"{names} ahead")
        if left:
            names = ", ".join(sorted(set(o.class_name for o in left[:2])))
            parts.append(f"{names} on left")
        if right:
            names = ", ".join(sorted(set(o.class_name for o in right[:2])))
            parts.append(f"{names} on right")
        path = self.get_path_status()
        return ". ".join(parts) + ". " + path

    def get_recently_departed(self, timeout: float = 2.5,
                               min_lifetime: float = 3.0) -> List[TrackedObject]:
        """Return objects that recently stopped being tracked."""
        now = time.time()
        departed = []
        for obj in list(self.tracked_objects.values()):
            since_seen = now - obj.last_seen
            if timeout < since_seen < timeout + 2.0 and obj.age > min_lifetime:
                departed.append(obj)
        return departed
