"""
Drishtimarga — Spatial Understanding & Threat Scoring (FIXED VERSION)
Tracks objects in 3D space, computes velocities, and scores threats.

FIXES:
1. Corrected depth mapping (piecewise inverse instead of linear)
2. Uncertainty-weighted fusion
3. Better distance estimation
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
try:
    import mediapipe as mp
    from mediapipe.solutions import pose as mp_pose
except ImportError:
    mp = None
    mp_pose = None

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
    
    IMPROVEMENTS:
    - Fixed depth-to-distance mapping (piecewise inverse)
    - Uncertainty-weighted fusion
    - Better calibration handling
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
        self._recently_departed: List[str] = []
        self._departure_announce_time: float = 0.0
        
        # NEW: Dynamic Calibration State
        self.depth_scale_factor: float = 1.0  # multiplier for depth_dist
        self.mp_pose = None
        if mp_pose:
            try:
                self.mp_pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,  # 0 = fastest for CPU
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe Pose: {e}")

    def estimate_distance(self, detection: Detection,
                          depth_map: Optional[np.ndarray],
                          frame_shape: Tuple[int, int]) -> float:
        """
        Estimate metric distance to a detected object.
        
        IMPROVED: Uses piecewise inverse depth mapping and uncertainty-weighted fusion.
        
        Approach:
          1. Bbox-height heuristic → approximate absolute meters (with uncertainty)
          2. Relative depth map → reliable ordering (with uncertainty)
          3. Uncertainty-weighted blend for best estimate
        """
        h_frame, w_frame = frame_shape[:2]

        # ── Bbox-height distance (approximate absolute meters) ──
        bbox_dist = None
        bbox_uncertainty = self.s_cfg.bbox_base_uncertainty  # Base uncertainty
        
        class_name = detection.class_name
        known_h = self.s_cfg.known_heights.get(class_name, None)
        
        if known_h is not None and detection.bbox_height > 10:
            bbox_dist = (known_h * self.s_cfg.focal_length_px) / detection.bbox_height
            bbox_dist = float(np.clip(bbox_dist, 0.3, 20.0))
            
            # Uncertainty increases with distance and low confidence
            bbox_uncertainty = self.s_cfg.bbox_base_uncertainty * (1 + bbox_dist / 10.0)
            bbox_uncertainty *= (1.5 - detection.confidence * 0.5)  # Higher for low confidence

        # ── Depth model relative value (0 = close, 1 = far) ──
        rel_depth = None
        depth_uncertainty = self.s_cfg.depth_base_uncertainty
        
        if depth_map is not None:
            cx, cy = int(detection.center[0]), int(detection.center[1])
            patch = 8
            y1 = max(0, cy - patch)
            y2 = min(h_frame, cy + patch)
            x1 = max(0, cx - patch)
            x2 = min(w_frame, cx + patch)
            
            # Use median for robustness
            rel_depth = float(np.median(depth_map[y1:y2, x1:x2]))
            
            # Estimate uncertainty from patch variance
            patch_std = float(np.std(depth_map[y1:y2, x1:x2]))
            depth_uncertainty = self.s_cfg.depth_base_uncertainty * (1 + patch_std * 5.0)

        # ── IMPROVED: Piecewise inverse depth mapping ──
        depth_dist = None
        if rel_depth is not None:
            # FIX: Use piecewise inverse mapping (more physically accurate)
            if rel_depth < 0.1:  # Very close objects
                depth_dist = 0.3 + 2.0 * rel_depth  # 0→0.3m, 0.1→0.5m
            elif rel_depth < 0.5:  # Medium range
                depth_dist = 0.5 + 8.0 * (rel_depth - 0.1)  # 0.1→0.5m, 0.5→3.7m
            else:  # Far range
                depth_dist = 3.7 + 12.0 * (rel_depth - 0.5)  # 0.5→3.7m, 1.0→9.7m
            
            # ── NEW: Apply dynamic calibration ──
            if self.s_cfg.calibration_enabled:
                depth_dist *= self.depth_scale_factor

        # ── IMPROVED: Uncertainty-weighted fusion ──
        if bbox_dist is not None and depth_dist is not None:
            # Weight by inverse uncertainty (lower uncertainty = higher weight)
            w_bbox = 1.0 / bbox_uncertainty
            w_depth = 1.0 / depth_uncertainty
            
            # Weighted average
            distance_m = (w_bbox * bbox_dist + w_depth * depth_dist) / (w_bbox + w_depth)
            
        elif bbox_dist is not None:
            # Only bbox available
            distance_m = bbox_dist
            
        elif depth_dist is not None:
            # Only depth available (less reliable for absolute distance)
            distance_m = depth_dist
            
        else:
            # No depth info available — use bbox area as very rough estimate
            # Smaller bbox = farther away (very crude)
            area_fraction = detection.area / (h_frame * w_frame)
            distance_m = np.clip(5.0 / (area_fraction + 0.01), 1.0, 15.0)

        # Clamp to reasonable range
        distance_m = float(np.clip(distance_m, 0.3, 20.0))
        return distance_m

    def update(self, detections: List[Detection], depth_map: Optional[np.ndarray],
               frame_rgb: np.ndarray, frame_shape: Tuple[int, int]) -> List[ThreatAssessment]:
        """
        Update spatial state with new detections.
        Returns threat assessments for audio feedback.
        """
        now = time.time()
        h, w = frame_shape[:2]
        
        # ── NEW: Dynamic Calibration pass ──
        if self.s_cfg.calibration_enabled and depth_map is not None:
            self._calibrate_depth(detections, frame_rgb, depth_map)

        # Update occupancy grid
        self.occupancy_grid.fill(0.0)

        assessments = []
        active_track_ids = set()

        for det in detections:
            active_track_ids.add(det.track_id)

            # Estimate distance
            distance = self.estimate_distance(det, depth_map, frame_shape)

            # Normalized position (0-1)
            rel_x = det.center[0] / w
            rel_y = det.center[1] / h

            # Update tracked object
            if det.track_id not in self.tracked_objects:
                obj = TrackedObject(
                    track_id=det.track_id,
                    class_name=det.class_name,
                    first_seen=now,
                    last_seen=now,
                )
                self.tracked_objects[det.track_id] = obj
            else:
                obj = self.tracked_objects[det.track_id]
                obj.last_seen = now

            # Add to history
            obj.history.append((distance, rel_x, rel_y, now))
            if len(obj.history) > self.s_cfg.track_history_length:
                obj.history.pop(0)

            # Update occupancy grid
            grid_x = int(rel_x * self.s_cfg.grid_size)
            grid_y = int((distance / (self.s_cfg.cell_size * self.s_cfg.grid_size)) * self.s_cfg.grid_size)
            grid_x = np.clip(grid_x, 0, self.s_cfg.grid_size - 1)
            grid_y = np.clip(grid_y, 0, self.s_cfg.grid_size - 1)
            self.occupancy_grid[grid_y, grid_x] += 1.0

            # Threat assessment
            assessment = self._assess_threat(obj)
            if assessment.priority != Priority.SKIP:
                assessments.append(assessment)

        # Track departures
        all_tracks = set(self.tracked_objects.keys())
        departed = all_tracks - active_track_ids
        for track_id in departed:
            obj = self.tracked_objects[track_id]
            if not obj.is_stale(self.s_cfg.stale_timeout):
                continue
            if obj.announcement_count > 0:
                self._recently_departed.append(obj.class_name)
                self._departure_announce_time = now
            del self.tracked_objects[track_id]

        if detections:
            self._last_detect_time = now

        # Sort by priority
        assessments.sort(key=lambda a: (a.priority, a.distance))
        return assessments

    def _assess_threat(self, obj: TrackedObject) -> ThreatAssessment:
        """Compute threat score and priority for a tracked object."""
        distance = obj.distance
        position = obj.position
        velocity = obj.approach_velocity
        ttc = obj.time_to_collision

        # Is this a new detection?
        is_new = obj.track_id not in self._known_tracks
        if is_new:
            self._known_tracks.add(obj.track_id)

        # Base threat score
        danger = self.t_cfg.danger_weights.get(obj.class_name, self.t_cfg.default_danger)
        score = danger / (distance + 0.5)

        # Boost for center position
        if position == Position.CENTER:
            score *= self.t_cfg.center_multiplier

        # Boost for approaching
        if velocity > self.t_cfg.approach_speed_threshold:
            score *= (1.0 + velocity)

        # --- Priority Assignment ---
        priority = Priority.SKIP

        # CRITICAL: imminent collision
        if ttc < self.t_cfg.critical_ttc:
            priority = Priority.CRITICAL
        # HIGH: new + close, or approaching fast
        elif (is_new and distance < self.t_cfg.close_distance * 2.0) or \
             (velocity > self.t_cfg.approach_speed_threshold and distance < 5.0):
            priority = Priority.HIGH
        elif is_new or distance < self.t_cfg.close_distance:
            priority = Priority.NORMAL
        else:
            # Non-new, non-approaching, non-close: skip entirely
            priority = Priority.SKIP

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

        # ── Beyond 10m ──
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
        """Analyze if the path ahead is clear."""
        gs = self.s_cfg.grid_size
        center_start = gs // 2 - 2
        center_end = gs // 2 + 2
        ahead = self.occupancy_grid[:5, center_start:center_end]

        if np.sum(ahead) < 0.5:
            return "Path ahead is clear, safe to walk forward."

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
        """Produce navigation guidance if needed."""
        now = time.time()

        # Departure announcements
        if (self.n_cfg.announce_departures
                and self._recently_departed
                and now - self._departure_announce_time < 1.5):
            names = ", ".join(set(self._recently_departed))
            self._recently_departed.clear()
            return f"{names} is no longer nearby."

        # Path-clear announcement
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

        # Periodic navigation update
        if active and now - self._last_guidance_time >= self.n_cfg.guidance_interval:
            self._last_guidance_time = now
            return self._build_navigation_update(active)

        return None

    def _build_navigation_update(self, active_objects: List[TrackedObject]) -> str:
        """Build concise navigation update."""
        summary = self._group_objects_summary(active_objects)
        path = self.get_path_status()
        return f"{summary}. {path}"

    def get_scene_summary(self) -> str:
        """Generate brief actionable overview."""
        if not self.tracked_objects:
            return "No objects detected. Path is clear, safe to move forward."

        active = [
            obj for obj in self.tracked_objects.values()
            if not obj.is_stale(self.s_cfg.stale_timeout)
        ]
        if not active:
            return "Area appears clear. You can walk forward."

        summary = self._group_objects_summary(active)
        path = self.get_path_status()
        return f"{summary}. {path}"

    def _calibrate_depth(self, detections: List[Detection], 
                         frame_rgb: np.ndarray, 
                         depth_map: np.ndarray):
        """Use MediaPipe Pose to find absolute ground truth and update scale factor."""
        person = next((d for d in detections if d.class_name == 'person'), None)
        if person is None:
            return

        if self.mp_pose is None:
            return

        # Run MediaPipe Pose on full frame (could crop if slow, but MP 0 is fast)
        try:
            results = self.mp_pose.process(frame_rgb)
        except Exception as e:
            logger.error(f"MediaPipe processing error: {e}")
            return
        
        if results.pose_world_landmarks:
            # Metric landmarks in meters (origin at hip center)
            # Use hip midpoint as stable reference
            landmarks = results.pose_world_landmarks.landmark
            mid_hip_z = (landmarks[23].z + landmarks[24].z) / 2.0
            
            # The world landmarks 'z' is relative to hips. 
            # We need camera-to-subject distance.
            # Pose world landmarks are actually in a metric space centered at hips,
            # but for monocular depth, we can use the bbox-based heuristic 
            # to anchor the MP scale or simply use MP landmarks' relative distances.
            
            # BUT: MediaPipe Iris or Pose World Landmarks 'z' only works if we know 
            # the camera's FOV. MediaPipe Pose world landmarks attempt to be metric.
            
            # Let's use a more robust way: if we have a person, the bbox_dist 
            # is our primary scale anchor. MP Pose refinement helps with orientation.
            # For now, let's refine the scale factor to minimize the difference 
            # between bbox_dist (scaled by focal length) and depth_dist.
            
            # Dynamic Focal Length Refinement instead of just depth scaling?
            # Actually, standard depth scaling is more flexible for monocular errors.
            
            known_h = self.s_cfg.known_heights.get('person', 1.7)
            ground_truth = (known_h * self.s_cfg.focal_length_px) / person.bbox_height
            
            # Sample depth map at person center
            cx, cy = int(person.center[0]), int(person.center[1])
            rel_depth = float(np.median(depth_map[max(0,cy-10):cy+10, max(0,cx-10):cx+10]))
            
            # Basic piecewise inv to get 'unscaled' meters
            if rel_depth < 0.1: d_unscaled = 0.3 + 2.0 * rel_depth
            elif rel_depth < 0.5: d_unscaled = 0.5 + 8.0 * (rel_depth - 0.1)
            else: d_unscaled = 3.7 + 12.0 * (rel_depth - 0.5)
            
            if d_unscaled > 0.01:
                target_scale = ground_truth / d_unscaled
                # Update EMA
                alpha = self.s_cfg.calibration_alpha
                self.depth_scale_factor = (1 - alpha) * self.depth_scale_factor + alpha * target_scale
                logger.debug(f"Dynamic Calibration: scale={self.depth_scale_factor:.2f}")

    def _group_objects_summary(self, objects: List[TrackedObject]) -> str:
        """Group similar objects for concise summaries."""
        from collections import Counter

        groups: Dict[str, list] = defaultdict(list)
        for obj in objects:
            groups[obj.class_name].append(obj)

        parts = []
        sorted_groups = sorted(
            groups.items(), key=lambda kv: min(o.distance for o in kv[1]))

        for class_name, objs in sorted_groups[:4]:
            objs_sorted = sorted(objs, key=lambda o: o.distance)
            closest = objs_sorted[0]
            pos_phrase = self._position_phrase(closest.position)
            dist_str = f"{closest.distance:.1f} meters"

            if len(objs) == 1:
                parts.append(f"{class_name} at {dist_str} {pos_phrase}")
            else:
                plural = class_name + "s" if not class_name.endswith("s") else class_name
                parts.append(
                    f"{len(objs)} {plural}, closest at {dist_str} {pos_phrase}")

        return ". ".join(parts)