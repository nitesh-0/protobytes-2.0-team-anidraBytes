"""
Drishtimarga — Spatial Understanding & Threat Scoring
Tracks objects in 3D space, computes velocities, and scores threats.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import SpatialConfig, ThreatConfig, NavigationConfig
from detector import Detection

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    CRITICAL = 0    # TTC < 3s, immediate interrupt
    HIGH = 1        # new close object, approaching
    NORMAL = 2      # new object in scene
    LOW = 3         # background, already announced
    SKIP = 4        # suppressed


class Position(str):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass
class TrackedObject:
    """Persistent state for a tracked object across frames."""
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    # Position history: [(distance_m, rel_x, rel_y, timestamp), ...]
    history: list = field(default_factory=list)
    # Last announced state
    last_announced_time: float = 0.0
    last_announced_distance: float = float("inf")
    announcement_count: int = 0

    @property
    def distance(self) -> float:
        """Current estimated distance in meters."""
        if not self.history:
            return float("inf")
        return self.history[-1][0]

    @property
    def position(self) -> str:
        """Current left/center/right position."""
        if not self.history:
            return Position.CENTER
        rel_x = self.history[-1][1]
        if rel_x < 0.33:
            return Position.LEFT
        elif rel_x > 0.67:
            return Position.RIGHT
        return Position.CENTER

    @property
    def approach_velocity(self) -> float:
        """
        Approach velocity in m/s.
        Positive = approaching, Negative = moving away.
        Uses linear regression for smooth estimate.
        """
        if len(self.history) < 3:
            return 0.0

        recent = self.history[-min(8, len(self.history)):]
        distances = [h[0] for h in recent]
        times = [h[3] for h in recent]

        n = len(distances)
        if n < 2:
            return 0.0

        sum_t = sum(times)
        sum_d = sum(distances)
        sum_td = sum(t * d for t, d in zip(times, distances))
        sum_tt = sum(t * t for t in times)

        denom = n * sum_tt - sum_t ** 2
        if abs(denom) < 1e-10:
            return 0.0

        slope = (n * sum_td - sum_t * sum_d) / denom
        # negative slope in distance = approaching (positive velocity)
        return -slope

    @property
    def time_to_collision(self) -> float:
        """Estimated seconds until collision. inf if moving away."""
        vel = self.approach_velocity
        if vel <= 0.05:
            return float("inf")
        current_dist = self.distance
        if current_dist <= 0:
            return 0.0
        return current_dist / vel

    @property
    def lateral_velocity(self) -> float:
        """Lateral movement speed (left/right)."""
        if len(self.history) < 3:
            return 0.0
        recent = self.history[-5:]
        xs = [h[1] for h in recent]
        times = [h[3] for h in recent]
        if len(xs) < 2:
            return 0.0
        dx = xs[-1] - xs[0]
        dt = times[-1] - times[0]
        return dx / dt if dt > 0 else 0.0

    @property
    def is_moving(self) -> bool:
        return abs(self.approach_velocity) > 0.2 or abs(self.lateral_velocity) > 0.05

    @property
    def age(self) -> float:
        """Seconds since first seen."""
        return self.last_seen - self.first_seen

    def is_stale(self, timeout: float) -> bool:
        return (time.time() - self.last_seen) > timeout


@dataclass
class ThreatAssessment:
    """Result of threat analysis for a tracked object."""
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
    Manages 3D spatial understanding, distance estimation,
    velocity tracking, threat prioritization, and navigation guidance.
    """

    def __init__(self, spatial_cfg: SpatialConfig, threat_cfg: ThreatConfig,
                 nav_cfg: NavigationConfig = None):
        self.s_cfg = spatial_cfg
        self.t_cfg = threat_cfg
        self.n_cfg = nav_cfg or NavigationConfig()
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.occupancy_grid = np.zeros(
            (spatial_cfg.grid_size, spatial_cfg.grid_size), dtype=np.float32
        )
        self._known_tracks: set = set()
        # Navigation state
        self._last_detect_time: float = time.time()
        self._last_path_clear_time: float = 0.0
        self._last_guidance_time: float = 0.0
        self._recently_departed: List[str] = []   # class names that just left
        self._departure_announce_time: float = 0.0

    def estimate_distance(self, detection: Detection,
                          depth_map: Optional[np.ndarray],
                          frame_shape: Tuple[int, int]) -> float:
        """
        Estimate metric distance to a detected object.
        Uses depth model if available, falls back to bbox-height heuristic.
        """
        h_frame, w_frame = frame_shape[:2]

        # Method 1: Depth model (if available)
        if depth_map is not None:
            cx, cy = int(detection.center[0]), int(detection.center[1])
            # Sample depth in a small region around center
            patch = 8
            y1 = max(0, cy - patch)
            y2 = min(h_frame, cy + patch)
            x1 = max(0, cx - patch)
            x2 = min(w_frame, cx + patch)
            rel_depth = float(np.median(depth_map[y1:y2, x1:x2]))

            # Method 1: Depth model (Depth Anything V2 outputs inverse depth / disparity)
            # We must invert it to get meters.
            # Calibration: relative disparity 0..1 usually maps to far..close
            # Actually, most monocular models output Disparity ~ 1/Distance
            # So Distance = Scale / (Disparity + Epsilon)
            
            # Simple inversion logic:
            # Let's assume the model output 'rel_depth' is proportional to 1/distance.
            # We need to calibrate the scale factor based on real-world tests (or config).
            # For now, we use a heuristic:
            epsilon = 0.01  # avoid div by zero
            # Invert: High value (close) -> Low distance
            # Low value (far) -> High distance
            
            # rel_depth is 0..1 from our normalization in depth_estimator.py
            # But wait, depth_estimator.py normalized it!
            # If we assume the raw output was disparity, then:
            # 0.0 (min disparity) = Farthest
            # 1.0 (max disparity) = Closest
            
            # So: distance = Constant / (rel_depth + epsilon)
            # Tuning Constant:
            # If rel_depth = 1.0 (closest), we want ~0.5m -> Constant = 0.5
            # If rel_depth = 0.0 (farthest), we want ~20m -> Constant / epsilon = 20 -> Constant = 0.2
            
            # Let's try a blend:
            scale_factor = 2.0  # Tunable parameter
            distance_m = scale_factor / (rel_depth + 0.1) 
            
            # Clamp to reasonable range for indoor/walking
            return np.clip(distance_m, 0.4, 20.0)

        # Method 2: Bbox height heuristic (fallback)
        class_name = detection.class_name
        known_h = self.s_cfg.known_heights.get(class_name, 1.0)
        if detection.bbox_height > 5:
            distance_m = (known_h * self.s_cfg.focal_length_px) / \
                detection.bbox_height
        else:
            distance_m = 20.0  # very far

        return np.clip(distance_m, 0.3, 50.0)

    def update(self, detections: List[Detection],
               depth_map: Optional[np.ndarray],
               frame_shape: Tuple[int, int]) -> List[ThreatAssessment]:
        """
        Update spatial state with new detections and return threat assessments.
        """
        now = time.time()
        h_frame, w_frame = frame_shape[:2]
        assessments: List[ThreatAssessment] = []

        # Decay occupancy grid
        self.occupancy_grid *= 0.95

        active_ids = set()
        for det in detections:
            tid = det.track_id
            active_ids.add(tid)

            # Estimate distance
            distance = self.estimate_distance(det, depth_map, frame_shape)

            # Relative position (0-1, left to right)
            rel_x = det.center[0] / w_frame if w_frame > 0 else 0.5
            rel_y = det.center[1] / h_frame if h_frame > 0 else 0.5

            # Create or update tracked object
            is_first_frame = tid not in self.tracked_objects
            if is_first_frame:
                self.tracked_objects[tid] = TrackedObject(
                    track_id=tid,
                    class_name=det.class_name,
                    first_seen=now,
                    last_seen=now,
                )

            obj = self.tracked_objects[tid]
            obj.last_seen = now
            obj.class_name = det.class_name  # update in case of misclass correction
            obj.history.append((distance, rel_x, rel_y, now))
            # Trim history
            if len(obj.history) > self.s_cfg.track_history_length:
                obj.history = obj.history[-self.s_cfg.track_history_length:]

            # Object counts as "new" for the first 1.5 seconds after appearing.
            # This is crucial: it gives the HIGH-priority first-detection message
            # multiple chances to be queued in case TTS is busy with another message.
            is_new = (now - obj.first_seen) < 1.5

            # Update occupancy grid
            self._update_grid(distance, rel_x)

            # Compute threat assessment
            assessment = self._assess_threat(obj, is_new)
            assessments.append(assessment)

        # Track when we last saw any detection
        if detections:
            self._last_detect_time = now

        # Clean stale tracks and record departures
        stale_ids = [
            tid for tid, obj in self.tracked_objects.items()
            if obj.is_stale(self.s_cfg.stale_timeout) and tid not in active_ids
        ]
        for tid in stale_ids:
            departed_obj = self.tracked_objects[tid]
            # Only note departure for objects that were close enough to matter
            if departed_obj.distance < self.n_cfg.medium_zone:
                self._recently_departed.append(departed_obj.class_name)
                self._departure_announce_time = now
            del self.tracked_objects[tid]

        # Sort by threat score (highest first)
        assessments.sort(key=lambda a: a.threat_score, reverse=True)
        return assessments

    def _update_grid(self, distance: float, rel_x: float):
        """Update the 2D occupancy grid."""
        gs = self.s_cfg.grid_size
        cs = self.s_cfg.cell_size
        # Map distance to grid row (0 = close, gs-1 = far)
        gz = int(distance / cs)
        # Map rel_x to grid col
        gx = int(rel_x * gs)
        gz = np.clip(gz, 0, gs - 1)
        gx = np.clip(gx, 0, gs - 1)
        self.occupancy_grid[gz, gx] = 1.0

    def _assess_threat(self, obj: TrackedObject, is_new: bool) -> ThreatAssessment:
        """Compute threat score and priority for an object."""
        distance = obj.distance
        velocity = obj.approach_velocity
        ttc = obj.time_to_collision
        position = obj.position

        # --- Threat Score Calculation ---
        score = 0.0

        # Distance factor (closer = more dangerous)
        if distance < 10:
            score += max(0, (10 - distance)) * 8

        # Approach velocity factor
        if velocity > self.t_cfg.approach_speed_threshold:
            score += velocity * 25

        # Class danger weight
        danger_w = self.t_cfg.danger_weights.get(
            obj.class_name, self.t_cfg.default_danger
        )
        score *= danger_w

        # Center of path multiplier
        if position == Position.CENTER:
            score *= self.t_cfg.center_multiplier

        # TTC urgency boost
        if ttc < self.t_cfg.critical_ttc:
            score *= 2.0
        elif ttc < self.t_cfg.critical_ttc * 2:
            score *= 1.3

        # --- Priority Assignment ---
        if ttc < self.t_cfg.critical_ttc and velocity > 0.5:
            priority = Priority.CRITICAL
        elif (is_new and distance < 4.0) or velocity > self.t_cfg.approach_speed_threshold:
            priority = Priority.HIGH
        elif is_new or distance < self.t_cfg.close_distance:
            priority = Priority.NORMAL
        else:
            priority = Priority.LOW

        # --- Audio Suppression Logic is handled in AudioEngine ---
        # We just set priorities based on immediate threat.
        
        # --- Generate Message ---
        message = self._build_message(
            obj, distance, position, velocity, ttc, priority)

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

    def _build_message(self, obj: TrackedObject, distance: float,
                       position: str, velocity: float,
                       ttc: float, priority: Priority) -> str:
        """
        Build an actionable spoken navigation message.
        Messages tell the user WHAT is there, WHERE it is, HOW FAR,
        and WHAT TO DO about it.
        """
        cls = obj.class_name
        dist_str = f"{distance:.1f} meters"
        pos_phrase = self._position_phrase(position)
        dodge_advice = self._dodge_direction(position)

        # ── CRITICAL: Imminent collision ──
        if priority == Priority.CRITICAL:
            if distance < self.n_cfg.immediate_zone:
                return f"DANGER! {cls} right in front of you! Stop and {dodge_advice}!"
            return (f"DANGER! {cls} approaching fast, {dist_str} {pos_phrase}! "
                    f"{dodge_advice.capitalize()}!")

        # ── IMMEDIATE ZONE (< 1m): Very close, must act now ──
        if distance < self.n_cfg.immediate_zone:
            return f"Careful! {cls} very close {pos_phrase}. {dodge_advice.capitalize()}."

        # ── CLOSE ZONE (< 2.5m): Active avoidance ──
        if distance < self.n_cfg.close_zone:
            if velocity > self.t_cfg.approach_speed_threshold:
                return (f"Warning, {cls} approaching {pos_phrase}, {dist_str}. "
                        f"{dodge_advice.capitalize()}.")
            return f"{cls} nearby {pos_phrase}, {dist_str}. {dodge_advice.capitalize()}."

        # ── MEDIUM ZONE (< 5m): Heads-up ──
        if distance < self.n_cfg.medium_zone:
            if velocity > self.t_cfg.approach_speed_threshold:
                return f"{cls} coming toward you {pos_phrase}, {dist_str}."
            return f"{cls} {pos_phrase}, {dist_str}."

        # ── FAR ZONE (< 10m): Awareness ──
        if distance < self.n_cfg.far_zone:
            return f"{cls} ahead {pos_phrase}, about {distance:.0f} meters."

        # ── Beyond 10m: only mention if moving or notable ──
        if velocity > self.t_cfg.approach_speed_threshold:
            return f"{cls} approaching from {distance:.0f} meters {pos_phrase}."
        return f"{cls} in the distance {pos_phrase}."

    def _position_phrase(self, position: str) -> str:
        """Convert left/center/right to natural speech."""
        if position == Position.LEFT:
            return "on your left"
        elif position == Position.RIGHT:
            return "on your right"
        return "ahead"

    def _dodge_direction(self, position: str) -> str:
        """Suggest which way to move to avoid an obstacle."""
        if position == Position.LEFT:
            return "move to your right"
        elif position == Position.RIGHT:
            return "move to your left"
        # Object is center — pick the more open side from occupancy grid
        return self._suggest_open_side()

    def _suggest_open_side(self) -> str:
        """Use the occupancy grid to suggest the more open direction."""
        gs = self.s_cfg.grid_size
        left_density = float(np.sum(self.occupancy_grid[:5, :gs // 2]))
        right_density = float(np.sum(self.occupancy_grid[:5, gs // 2:]))
        if left_density < right_density:
            return "move to your left"
        return "move to your right"

    def get_path_status(self) -> str:
        """Analyze if the path ahead is clear and give actionable advice."""
        gs = self.s_cfg.grid_size
        # Check center columns, first 5 rows (close range)
        center_start = gs // 2 - 2
        center_end = gs // 2 + 2
        ahead = self.occupancy_grid[:5, center_start:center_end]

        if np.sum(ahead) < 0.5:
            return "Path ahead is clear, safe to walk forward."

        # Check which side is more open
        left_density = float(np.sum(self.occupancy_grid[:5, :gs // 2]))
        right_density = float(np.sum(self.occupancy_grid[:5, gs // 2:]))

        if left_density < right_density * 0.5:
            return "Path blocked ahead. Move to your left, it's more open."
        elif right_density < left_density * 0.5:
            return "Path blocked ahead. Move to your right, it's more open."
        elif left_density < right_density:
            return "Path blocked ahead. Try moving slightly left."
        else:
            return "Path blocked ahead. Try moving slightly right."

    def get_navigation_guidance(self) -> Optional[str]:
        """
        Produce a navigation guidance message if enough time has passed.
        Called every frame — returns None when nothing needs to be said.
        """
        now = time.time()

        # ── Departure announcements ──
        if (self.n_cfg.announce_departures
                and self._recently_departed
                and now - self._departure_announce_time < 1.5):
            names = ", ".join(set(self._recently_departed))
            self._recently_departed.clear()
            return f"{names} is no longer nearby."

        # ── Path-clear announcement ──
        active = [
            obj for obj in self.tracked_objects.values()
            if not obj.is_stale(self.s_cfg.stale_timeout)
            and obj.distance < self.n_cfg.medium_zone
        ]
        if not active:
            time_since_last_detect = now - self._last_detect_time
            time_since_last_clear = now - self._last_path_clear_time
            if (time_since_last_detect >= self.n_cfg.path_clear_delay
                    and time_since_last_clear >= self.n_cfg.path_clear_repeat):
                self._last_path_clear_time = now
                return "Path is clear. Safe to move forward."

        # ── Periodic navigation update (with objects present) ──
        if active and now - self._last_guidance_time >= self.n_cfg.guidance_interval:
            self._last_guidance_time = now
            return self._build_navigation_update(active)

        return None

    def _build_navigation_update(self, active_objects: List[TrackedObject]) -> str:
        """
        Build a concise navigation update summarizing what's around
        the user and what they should do.
        """
        # Sort by distance
        sorted_objs = sorted(active_objects, key=lambda o: o.distance)

        # Take the closest 3
        closest = sorted_objs[:3]
        parts = []
        for obj in closest:
            pos_phrase = self._position_phrase(obj.position)
            parts.append(
                f"{obj.class_name} at {obj.distance:.1f} meters {pos_phrase}")

        summary = ". ".join(parts)
        path = self.get_path_status()
        return f"{summary}. {path}"

    def get_scene_summary(self) -> str:
        """Generate a brief actionable overview of surroundings."""
        if not self.tracked_objects:
            return "No objects detected. Path is clear, safe to move forward."

        active = [
            obj for obj in self.tracked_objects.values()
            if not obj.is_stale(self.s_cfg.stale_timeout)
        ]
        if not active:
            return "Area appears clear. You can walk forward."

        # Group by position
        left = [o for o in active if o.position == Position.LEFT]
        center = [o for o in active if o.position == Position.CENTER]
        right = [o for o in active if o.position == Position.RIGHT]

        parts = []
        if center:
            items = []
            for o in sorted(center, key=lambda x: x.distance)[:2]:
                items.append(f"{o.class_name} at {o.distance:.1f} meters")
            parts.append(f"{', '.join(items)} ahead")
        if left:
            items = []
            for o in sorted(left, key=lambda x: x.distance)[:2]:
                items.append(f"{o.class_name} at {o.distance:.1f} meters")
            parts.append(f"{', '.join(items)} on your left")
        if right:
            items = []
            for o in sorted(right, key=lambda x: x.distance)[:2]:
                items.append(f"{o.class_name} at {o.distance:.1f} meters")
            parts.append(f"{', '.join(items)} on your right")

        path = self.get_path_status()
        summary = ". ".join(parts)
        return f"{summary}. {path}"
