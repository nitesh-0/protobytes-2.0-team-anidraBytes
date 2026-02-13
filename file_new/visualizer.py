"""
Drishtimarga v2 — Visualization
Annotated video with bounding boxes, metric distances, threat colors, and HUD.
"""

import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import DisplayConfig
from detector import Detection
from spatial_engine import Priority, ThreatAssessment
from announcement_memory import AnnouncementMemory, ObjectState

PRIORITY_COLORS = {
    Priority.CRITICAL: (0, 0, 255),
    Priority.HIGH:     (0, 140, 255),
    Priority.NORMAL:   (0, 220, 0),
    Priority.LOW:      (180, 180, 180),
}

STATE_LABELS = {
    ObjectState.NEW: "NEW",
    ObjectState.ACTIVE: "ACT",
    ObjectState.STABLE: "STB",
    ObjectState.DEPARTED: "DEP",
}
STATE_COLORS = {
    ObjectState.NEW: (255, 200, 0),
    ObjectState.ACTIVE: (0, 255, 0),
    ObjectState.STABLE: (150, 150, 150),
    ObjectState.DEPARTED: (100, 100, 100),
}

DEFAULT_COLOR = (0, 255, 128)


class Visualizer:
    def __init__(self, config: DisplayConfig):
        self.config = config
        self._fps_history: list = []
        self._last_t = time.time()

    def draw(self, frame: np.ndarray,
             detections: List[Detection],
             assessments: List[ThreatAssessment],
             memory: Optional[AnnouncementMemory] = None,
             depth_map: Optional[np.ndarray] = None,
             extra_info: dict = None) -> np.ndarray:
        canvas = frame.copy()
        a_map = {a.track_id: a for a in assessments}

        for det in detections:
            a = a_map.get(det.track_id)
            self._draw_det(canvas, det, a, memory)

        self._draw_hud(canvas, detections, assessments, memory, extra_info)
        self._draw_grid(canvas)
        return canvas

    def _draw_det(self, canvas, det: Detection,
                  a: Optional[ThreatAssessment],
                  memory: Optional[AnnouncementMemory]):
        x1, y1, x2, y2 = det.bbox.astype(int)
        th = self.config.annotation_thickness
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = self.config.font_scale

        # Box color from priority
        color = DEFAULT_COLOR
        if a:
            color = PRIORITY_COLORS.get(a.priority, DEFAULT_COLOR)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, th)

        # Label line 1: ID + class + distance
        parts = [f"#{det.track_id} {det.class_name}"]
        if a:
            parts.append(f"{a.distance:.1f}m")
            if a.approach_velocity > 0.3:
                parts.append(f"v={a.approach_velocity:.1f}")
            if a.time_to_collision < 10:
                parts.append(f"TTC={a.time_to_collision:.1f}s")
        label = " | ".join(parts)

        (tw, lh), _ = cv2.getTextSize(label, font, fs, 1)
        cv2.rectangle(canvas, (x1, y1 - lh - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), font, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Label line 2: memory state
        if memory and det.track_id in memory.entries:
            entry = memory.entries[det.track_id]
            st_label = STATE_LABELS.get(entry.state, "?")
            st_color = STATE_COLORS.get(entry.state, (200, 200, 200))
            st_text = f"[{st_label}] x{entry.announcement_count}"
            cv2.putText(canvas, st_text, (x1 + 2, y2 + 14), font, 0.4,
                        st_color, 1, cv2.LINE_AA)

        # Critical pulsing
        if a and a.priority == Priority.CRITICAL:
            cv2.rectangle(canvas, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                          (0, 0, 255), 3)
            cv2.putText(canvas, "!! DANGER !!", (x1, y2 + 30), font,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Velocity arrow
        if a and abs(a.approach_velocity) > 0.3:
            cx, cy = int(det.center[0]), int(det.center[1])
            if a.approach_velocity > 0:
                cv2.arrowedLine(canvas, (cx, cy - 20), (cx, cy + 10),
                                (0, 0, 255), 2, tipLength=0.4)
            else:
                cv2.arrowedLine(canvas, (cx, cy + 10), (cx, cy - 20),
                                (0, 255, 0), 2, tipLength=0.4)

    def _draw_hud(self, canvas, detections, assessments,
                  memory: Optional[AnnouncementMemory], extra_info):
        h, w = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # FPS
        now = time.time()
        dt = now - self._last_t
        self._last_t = now
        fps = 1.0 / dt if dt > 0 else 0
        self._fps_history.append(fps)
        self._fps_history = self._fps_history[-30:]
        avg_fps = sum(self._fps_history) / len(self._fps_history)

        # Background
        bg = np.zeros((110, 300, 3), dtype=np.uint8)
        bg[:] = (30, 30, 30)
        bh, bw = bg.shape[:2]
        canvas[0:bh, 0:bw] = cv2.addWeighted(
            canvas[0:bh, 0:bw], 0.3, bg, 0.7, 0)

        y = 18
        cv2.putText(canvas, f"FPS: {avg_fps:.1f}", (10, y), font, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
        y += 20
        cv2.putText(canvas, f"Objects: {len(detections)}", (10, y), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 20

        if extra_info:
            det_ms = extra_info.get("detection_ms", 0)
            dep_ms = extra_info.get("depth_ms", 0)
            cv2.putText(canvas, f"Det: {det_ms:.0f}ms | Depth: {dep_ms:.0f}ms",
                        (10, y), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            y += 18
            metric = "metric" if extra_info.get("depth_metric") else "bbox fallback"
            cv2.putText(canvas, f"Depth: {metric}", (10, y), font, 0.42,
                        (200, 200, 200), 1, cv2.LINE_AA)
            y += 18

        # Memory stats
        if memory:
            ms = memory.stats
            states = ms.get("states", {})
            mem_text = " ".join(f"{k}:{v}" for k, v in states.items())
            cv2.putText(canvas, f"Mem: {mem_text}", (10, y), font, 0.38,
                        (180, 180, 220), 1, cv2.LINE_AA)

        # Threat summary (top-right)
        crit = sum(1 for a in assessments if a.priority == Priority.CRITICAL)
        high = sum(1 for a in assessments if a.priority == Priority.HIGH)
        if crit > 0:
            txt = f"!! {crit} CRITICAL !!"
            (tw, _), _ = cv2.getTextSize(txt, font, 0.7, 2)
            cv2.putText(canvas, txt, (w - tw - 15, 30), font, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
        elif high > 0:
            txt = f"{high} approaching"
            (tw, _), _ = cv2.getTextSize(txt, font, 0.5, 1)
            cv2.putText(canvas, txt, (w - tw - 15, 25), font, 0.5,
                        (0, 140, 255), 1, cv2.LINE_AA)

    def _draw_grid(self, canvas):
        h, w = canvas.shape[:2]
        t = w // 3
        cv2.line(canvas, (t, 0), (t, h), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.line(canvas, (2 * t, 0), (2 * t, h), (100, 100, 100), 1, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "LEFT", (t // 2 - 18, h - 10), font, 0.38,
                    (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, "CENTER", (w // 2 - 25, h - 10), font, 0.38,
                    (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, "RIGHT", (2 * t + t // 2 - 22, h - 10), font, 0.38,
                    (150, 150, 150), 1, cv2.LINE_AA)
