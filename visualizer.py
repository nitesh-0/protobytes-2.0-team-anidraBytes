"""
Drishtimarga — Visualization
Annotated video display with bounding boxes, depth, threat indicators, and HUD.
"""

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import DisplayConfig
from detector import Detection
from spatial_engine import Priority, ThreatAssessment


# Color palette (BGR)
COLORS = {
    Priority.CRITICAL: (0, 0, 255),     # Red
    Priority.HIGH:     (0, 140, 255),    # Orange
    Priority.NORMAL:   (0, 220, 0),      # Green
    Priority.LOW:      (180, 180, 180),  # Gray
}

CLASS_COLORS = {
    "person": (255, 180, 0),
    "car": (0, 100, 255),
    "bus": (0, 50, 200),
    "truck": (0, 50, 200),
    "motorcycle": (0, 180, 255),
    "bicycle": (255, 255, 0),
    "dog": (200, 100, 255),
    "cat": (255, 100, 200),
}
DEFAULT_COLOR = (0, 255, 128)


class Visualizer:
    """Renders annotated frames with detection overlays and HUD."""

    def __init__(self, config: DisplayConfig):
        self.config = config
        self._fps_history = []
        self._last_frame_time = time.time()

    def draw(self, frame: np.ndarray,
             detections: List[Detection],
             assessments: List[ThreatAssessment],
             depth_map: Optional[np.ndarray] = None,
             extra_info: dict = None) -> np.ndarray:
        """Draw all overlays on the frame."""
        canvas = frame.copy()

        # Build assessment lookup
        assessment_map = {a.track_id: a for a in assessments}

        # Draw detections
        for det in detections:
            assessment = assessment_map.get(det.track_id)
            self._draw_detection(canvas, det, assessment)

        # Draw HUD
        self._draw_hud(canvas, detections, assessments, extra_info)

        # Draw position grid overlay
        self._draw_grid_overlay(canvas)

        return canvas

    def _draw_detection(self, canvas: np.ndarray, det: Detection,
                        assessment: Optional[ThreatAssessment]):
        """Draw bounding box and label for a single detection."""
        x1, y1, x2, y2 = det.bbox.astype(int)
        thick = self.config.annotation_thickness

        # Color based on threat priority
        if assessment:
            box_color = COLORS.get(assessment.priority, DEFAULT_COLOR)
        else:
            box_color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)

        # Draw box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thick)

        # Build label
        label_parts = [f"#{det.track_id} {det.class_name}"]
        if assessment:
            label_parts.append(f"{assessment.distance:.1f}m")
            if assessment.approach_velocity > 0.3:
                label_parts.append(f"v={assessment.approach_velocity:.1f}m/s")
            if assessment.time_to_collision < 10:
                label_parts.append(f"TTC={assessment.time_to_collision:.1f}s")

        label = " | ".join(label_parts)
        font_scale = self.config.font_scale
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Label background
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), font, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Danger indicator for critical threats
        if assessment and assessment.priority == Priority.CRITICAL:
            # Pulsing red border
            cv2.rectangle(canvas, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                          (0, 0, 255), 3)
            cv2.putText(canvas, "!! DANGER !!", (x1, y2 + 20), font,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Direction arrow for moving objects
        if assessment and abs(assessment.approach_velocity) > 0.3:
            cx, cy = int(det.center[0]), int(det.center[1])
            if assessment.approach_velocity > 0:
                # Arrow pointing toward camera (approaching)
                cv2.arrowedLine(canvas, (cx, cy - 20), (cx, cy + 10),
                                (0, 0, 255), 2, tipLength=0.4)
            else:
                # Arrow pointing away (receding)
                cv2.arrowedLine(canvas, (cx, cy + 10), (cx, cy - 20),
                                (0, 255, 0), 2, tipLength=0.4)

    def _draw_hud(self, canvas: np.ndarray, detections: List[Detection],
                  assessments: List[ThreatAssessment], extra_info: dict = None):
        """Draw heads-up display with stats."""
        h, w = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # FPS calculation
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        fps = 1.0 / dt if dt > 0 else 0
        self._fps_history.append(fps)
        if len(self._fps_history) > 30:
            self._fps_history = self._fps_history[-30:]
        avg_fps = sum(self._fps_history) / len(self._fps_history)

        # Top-left: system stats
        stats_bg = np.zeros((90, 280, 3), dtype=np.uint8)
        stats_bg[:] = (30, 30, 30)
        canvas[0:90, 0:280] = cv2.addWeighted(canvas[0:90, 0:280], 0.3, stats_bg, 0.7, 0)

        y_off = 18
        cv2.putText(canvas, f"FPS: {avg_fps:.1f}", (10, y_off), font, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
        y_off += 20
        cv2.putText(canvas, f"Objects: {len(detections)}", (10, y_off), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y_off += 20

        if extra_info:
            det_ms = extra_info.get("detection_ms", 0)
            depth_ms = extra_info.get("depth_ms", 0)
            cv2.putText(canvas, f"Det: {det_ms:.0f}ms | Depth: {depth_ms:.0f}ms",
                        (10, y_off), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            y_off += 20
            depth_status = "GPU" if extra_info.get("depth_available") else "bbox fallback"
            cv2.putText(canvas, f"Depth: {depth_status}", (10, y_off), font, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)

        # Top-right: threat summary
        critical_count = sum(1 for a in assessments if a.priority == Priority.CRITICAL)
        high_count = sum(1 for a in assessments if a.priority == Priority.HIGH)

        if critical_count > 0:
            alert_text = f"!! {critical_count} CRITICAL THREAT{'S' if critical_count > 1 else ''} !!"
            (tw, _), _ = cv2.getTextSize(alert_text, font, 0.7, 2)
            cv2.putText(canvas, alert_text, (w - tw - 15, 30), font, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
        elif high_count > 0:
            alert_text = f"{high_count} approaching"
            (tw, _), _ = cv2.getTextSize(alert_text, font, 0.5, 1)
            cv2.putText(canvas, alert_text, (w - tw - 15, 25), font, 0.5,
                        (0, 140, 255), 1, cv2.LINE_AA)

    def _draw_grid_overlay(self, canvas: np.ndarray):
        """Draw left/center/right guide lines."""
        h, w = canvas.shape[:2]
        third = w // 3

        # Vertical guide lines (semi-transparent)
        cv2.line(canvas, (third, 0), (third, h), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.line(canvas, (2 * third, 0), (2 * third, h), (100, 100, 100), 1, cv2.LINE_AA)

        # Position labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "LEFT", (third // 2 - 20, h - 10), font, 0.4,
                    (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, "CENTER", (w // 2 - 25, h - 10), font, 0.4,
                    (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, "RIGHT", (2 * third + third // 2 - 22, h - 10), font, 0.4,
                    (150, 150, 150), 1, cv2.LINE_AA)
