"""
Drishtimarga v2 — Local Client for Modal Cloud Inference

Captures frames from ESP32-CAM (or webcam), sends them to the Modal
GPU server for YOLO + Depth inference, receives results, and does
TTS announcements + visualization locally.

Usage:
    # First deploy the server:
    modal deploy modal_server.py

    # Then run client pointing to ESP32-CAM:
    python local_client.py --source http://192.168.6.50:81/stream

    # Or with webcam:
    python local_client.py --source 0

    # Provide your Modal endpoint URL:
    python local_client.py --source http://192.168.6.50:81/stream --server-url https://YOUR_USERNAME--drishtimarga-v2-drishtimargainference-infer.modal.run

    # Audio only (no display window):
    python local_client.py --source http://192.168.6.50:81/stream --no-display

Notes:
    After `modal deploy modal_server.py`, Modal prints endpoint URLs like:
      https://<username>--drishtimarga-v2-drishtimargainference-infer.modal.run
      https://<username>--drishtimarga-v2-drishtimargainference-health.modal.run
    Copy the 'infer' URL and pass it via --server-url.
"""

import argparse
import base64
import json
import logging
import queue
import signal
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum, Enum

import cv2
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-14s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("drishtimarga-client")


# ═══════════════════════════════════════════════════════════════════
# Inline lightweight versions of Priority/ThreatAssessment/Memory
# (so the client doesn't need to import the full spatial_engine)
# ═══════════════════════════════════════════════════════════════════

class Priority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    SKIP = 4


class ObjectState(Enum):
    NEW = "new"
    ACTIVE = "active"
    STABLE = "stable"
    DEPARTED = "departed"


@dataclass
class ThreatAssessment:
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


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    history: list = field(default_factory=list)  # [(distance, rel_x, rel_y, ts)]

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
            return "left"
        elif rx > 0.67:
            return "right"
        return "center"

    @property
    def approach_velocity(self) -> float:
        if len(self.history) < 3:
            return 0.0
        recent = self.history[-min(10, len(self.history)):]
        distances = [h[0] for h in recent]
        times = [h[3] for h in recent]
        n = len(distances)
        st = sum(times)
        sd = sum(distances)
        std_ = sum(t * d for t, d in zip(times, distances))
        stt = sum(t * t for t in times)
        denom = n * stt - st * st
        if abs(denom) < 1e-10:
            return 0.0
        slope = (n * std_ - st * sd) / denom
        return -slope

    @property
    def time_to_collision(self) -> float:
        vel = self.approach_velocity
        if vel <= 0.05:
            return float("inf")
        d = self.distance
        return d / vel if d > 0 else 0.0

    @property
    def age(self) -> float:
        return self.last_seen - self.first_seen


# ═══════════════════════════════════════════════════════════════════
# Frame Grabber (reused from main.py)
# ═══════════════════════════════════════════════════════════════════

class FrameGrabber:
    """Threaded capture for webcam or ESP32-CAM with auto-reconnect."""

    ESP32_RESOLUTIONS = {
        "UXGA": (1600, 1200, 13), "SXGA": (1280, 1024, 12),
        "XGA": (1024, 768, 10), "SVGA": (800, 600, 9),
        "VGA": (640, 480, 8), "CIF": (400, 296, 6),
        "QVGA": (320, 240, 5),
    }

    def __init__(self, source, width=640, height=480):
        self.source = source
        self.width = width
        self.height = height
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._total = 0
        self._dropped = 0
        self._is_stream = isinstance(source, str) and source.startswith("http")
        self._is_snapshot = False
        self._base_url = ""
        self._reconnect_count = 0

        if self._is_stream:
            from urllib.parse import urlparse
            p = urlparse(source)
            self._base_url = f"{p.scheme}://{p.netloc}"
            self._is_snapshot = any(
                k in source.lower() for k in ["/capture", "/cam-hi", "/cam-lo", "/jpg"])

    def start(self) -> bool:
        logger.info(f"Opening camera: {self.source}")
        if self._is_stream and not self._is_snapshot:
            self._esp32_set_resolution()
        ok = self._open_capture()
        if not ok:
            return False
        self._running = True
        self._thread = threading.Thread(
            target=self._loop_snapshot if self._is_snapshot else self._loop,
            daemon=True, name="FrameGrabber")
        self._thread.start()
        return True

    def _open_capture(self) -> bool:
        if self._cap is not None:
            try: self._cap.release()
            except: pass
        if self._is_snapshot:
            frame = self._fetch_snapshot()
            if frame is not None:
                h, w = frame.shape[:2]
                logger.info(f"ESP32-CAM snapshot ready: {w}x{h}")
                return True
            return False
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            logger.error(f"Cannot open camera: {self.source}")
            return False
        if not self._is_stream:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera ready: {w}x{h}")
        return True

    def _esp32_set_resolution(self):
        if not self._base_url:
            return
        best_val = 8
        best_diff = 999999
        for _, (rw, rh, val) in self.ESP32_RESOLUTIONS.items():
            diff = abs(rw - self.width) + abs(rh - self.height)
            if diff < best_diff:
                best_diff = diff
                best_val = val
        try:
            import urllib.request
            urllib.request.urlopen(
                f"{self._base_url}/control?var=framesize&val={best_val}", timeout=3)
        except: pass

    def _fetch_snapshot(self) -> Optional[np.ndarray]:
        try:
            import urllib.request
            resp = urllib.request.urlopen(self.source, timeout=5)
            data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except: return None

    def _loop(self):
        fails = 0
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                fails += 1
                if self._is_stream and fails > 10:
                    if not self._reconnect(): break
                    fails = 0
                else: time.sleep(0.05)
                continue
            fails = 0
            self._total += 1
            if self._q.full():
                try: self._q.get_nowait(); self._dropped += 1
                except queue.Empty: pass
            
            # Rotate 90 degrees anti-clockwise (use existing frame variable)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self._q.put((frame, time.time()))

    def _loop_snapshot(self):
        fails = 0
        while self._running:
            frame = self._fetch_snapshot()
            if frame is None:
                fails += 1
                if fails > 20: break
                time.sleep(min(2.0, 0.1 * fails))
                continue
            fails = 0
            self._total += 1
            if self._q.full():
                try: self._q.get_nowait(); self._dropped += 1
                except queue.Empty: pass
            
            # Rotate 90 degrees anti-clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self._q.put((frame, time.time()))

    def _reconnect(self) -> bool:
        self._reconnect_count += 1
        if self._reconnect_count > 50: return False
        backoff = min(10.0, 0.5 * (2 ** min(self._reconnect_count, 5)))
        logger.warning(f"Stream lost — reconnect in {backoff:.1f}s")
        time.sleep(backoff)
        try:
            if self._cap: self._cap.release()
            self._cap = cv2.VideoCapture(self.source)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return True
        except: pass
        return self._reconnect()

    def get(self, timeout=1.0):
        try: return self._q.get(timeout=timeout)
        except queue.Empty: return None, None

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=2.0)
        if self._cap: self._cap.release()


# ═══════════════════════════════════════════════════════════════════
# Local Spatial Tracker (tracks objects across frames using server results)
# ═══════════════════════════════════════════════════════════════════

class LocalSpatialTracker:
    """
    Lightweight local tracker that takes server detection results and
    computes velocities, threat scores, and generates messages.
    """

    DANGER_WEIGHTS = {
        "bus": 5.0, "truck": 5.0, "car": 4.0, "motorcycle": 3.5,
        "bicycle": 2.5, "person": 1.5, "dog": 2.0, "cat": 1.0,
        "skateboard": 2.0, "train": 6.0,
        "chair": 1.0, "bench": 1.0, "fire hydrant": 1.5,
    }

    def __init__(self):
        self.tracked: Dict[int, TrackedObject] = {}
        self._smoothing_alpha = 0.3

    def update(self, server_detections: List[dict]) -> List[ThreatAssessment]:
        now = time.time()
        assessments = []
        active_ids = set()

        for det in server_detections:
            tid = det["track_id"]
            active_ids.add(tid)
            distance = det["distance"]
            rel_x = det.get("rel_x", 0.5)

            is_new = tid not in self.tracked
            if is_new:
                self.tracked[tid] = TrackedObject(
                    track_id=tid,
                    class_name=det["class_name"],
                    first_seen=now,
                    last_seen=now,
                )

            obj = self.tracked[tid]
            obj.last_seen = now
            obj.class_name = det["class_name"]

            # EMA distance smoothing
            if obj.history:
                prev_dist = obj.history[-1][0]
                distance = self._smoothing_alpha * distance + (1.0 - self._smoothing_alpha) * prev_dist

            obj.history.append((distance, rel_x, 0.5, now))
            if len(obj.history) > 20:
                obj.history = obj.history[-20:]

            assessment = self._assess_threat(obj, is_new)
            assessments.append(assessment)

        # Cleanup stale
        stale = [tid for tid, o in self.tracked.items()
                 if (now - o.last_seen) > 2.5 and tid not in active_ids]
        for tid in stale:
            del self.tracked[tid]

        assessments.sort(key=lambda a: a.threat_score, reverse=True)
        return assessments

    def _assess_threat(self, obj: TrackedObject, is_new: bool) -> ThreatAssessment:
        distance = obj.distance
        velocity = obj.approach_velocity
        ttc = obj.time_to_collision
        position = obj.position

        score = 0.0
        if distance < 15:
            score += max(0, (15 - distance)) * 5
        if velocity > 0.3:
            score += velocity * 25
        dw = self.DANGER_WEIGHTS.get(obj.class_name, 1.0)
        score *= dw
        if position == "center":
            score *= 1.5
        if ttc < 3.0:
            score *= 2.0

        if ttc < 3.0 and velocity > 0.5:
            priority = Priority.CRITICAL
        elif (is_new and distance < 5.0) or velocity > 0.3:
            priority = Priority.HIGH
        elif is_new or distance < 1.5:
            priority = Priority.NORMAL
        else:
            priority = Priority.LOW

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

    def _quantize_distance(self, d: float) -> str:
        d_ft = d * 3.28084
        if d_ft < 3.0: return "very close"
        elif d_ft < 10.0: return f"{d_ft:.1f} feet"
        elif d_ft < 20.0: return f"about {round(d_ft)} feet"
        else: return f"about {round(d_ft/5)*5} feet"

    def _build_message(self, obj, distance, position, velocity, ttc, priority):
        cls = obj.class_name
        d = self._quantize_distance(distance)
        if priority == Priority.CRITICAL:
            return f"DANGER! {cls} approaching fast! {d}, {position}!"
        if velocity > 0.3:
            return f"Warning, {cls} approaching, {d}, {position}"
        if distance < 1.5:
            return f"{cls}, very close, {position}"
        return f"{cls}, {d}, {position}"

    def get_scene_summary(self) -> str:
        now = time.time()
        active = [o for o in self.tracked.values() if (now - o.last_seen) < 2.5]
        if not active:
            return "Area appears clear"
        left = [o for o in active if o.position == "left"]
        center = [o for o in active if o.position == "center"]
        right = [o for o in active if o.position == "right"]
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
        return ". ".join(parts) if parts else "Area appears clear"


# ═══════════════════════════════════════════════════════════════════
# Local Announcement Memory (prevents repetition)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    track_id: int
    class_name: str
    state: ObjectState
    first_seen: float
    last_seen: float
    last_announced: float = 0.0
    announced_distance: float = float("inf")
    announced_position: str = ""
    announced_velocity: float = 0.0
    announced_priority: Priority = Priority.LOW
    announcement_count: int = 0
    _stable_since: float = 0.0


class LocalAnnouncementMemory:
    """Simplified announcement memory for client-side filtering."""

    # Thresholds
    COOLDOWN_NEW = 0.0
    COOLDOWN_ACTIVE = 7.0
    COOLDOWN_STABLE = 60.0
    COOLDOWN_CRITICAL = 2.5
    STABLE_AFTER = 6.0
    DIST_THRESH_NEAR = 0.8
    DIST_THRESH_FAR = 2.0
    VEL_THRESH = 0.8
    MAX_PER_CYCLE = 2
    DEPART_AFTER = 2.5
    DEPART_MIN_LIFE = 5.0
    SUMMARY_INTERVAL = 20.0

    def __init__(self):
        self.entries: Dict[int, MemoryEntry] = {}
        self._departure_announced: set = set()
        self._last_summary_time = 0.0
        self._last_summary_hash = ""

    def _dist_bucket(self, d: float) -> float:
        if not np.isfinite(d): return float("inf")
        if d < 2.0: return round(d * 2) / 2
        elif d < 5.0: return round(d)
        else: return round(d / 2) * 2

    def _meaningful_change(self, e: MemoryEntry, a: ThreatAssessment) -> bool:
        cb = self._dist_bucket(a.distance)
        pb = self._dist_bucket(e.announced_distance)
        if cb != pb:
            thresh = self.DIST_THRESH_NEAR if a.distance < 3 else self.DIST_THRESH_FAR
            if abs(a.distance - e.announced_distance) > thresh:
                return True
        if a.position != e.announced_position and e.announced_position:
            return True
        was_app = e.announced_velocity > self.VEL_THRESH
        now_app = a.approach_velocity > self.VEL_THRESH
        if was_app != now_app:
            return True
        if a.priority < e.announced_priority:
            return True
        return False

    def filter(self, assessments: List[ThreatAssessment]) -> Tuple[List[ThreatAssessment], List[str]]:
        now = time.time()
        active_ids = {a.track_id for a in assessments}

        # Update entries
        for a in assessments:
            self._update(a, now)

        # Departures
        departures = []
        for tid, e in self.entries.items():
            if tid in active_ids or tid in self._departure_announced:
                continue
            if (now - e.last_seen) > self.DEPART_AFTER:
                if (e.last_seen - e.first_seen) >= self.DEPART_MIN_LIFE:
                    departures.append(f"{e.class_name} has left")
                    self._departure_announced.add(tid)
                    e.state = ObjectState.DEPARTED

        # Filter
        to_speak = []
        for a in sorted(assessments, key=lambda x: x.threat_score, reverse=True):
            if len(to_speak) >= self.MAX_PER_CYCLE:
                break
            e = self.entries.get(a.track_id)
            if not e:
                continue
            should = self._should_announce(e, a, now)
            if should:
                to_speak.append(a)
                e.last_announced = now
                e.announced_distance = a.distance
                e.announced_position = a.position
                e.announced_velocity = a.approach_velocity
                e.announced_priority = a.priority
                e.announcement_count += 1

        # Cleanup
        to_rm = [tid for tid, e in self.entries.items()
                 if tid not in active_ids and (now - e.last_seen) > 5.0]
        for tid in to_rm:
            del self.entries[tid]
            self._departure_announced.discard(tid)

        return to_speak, departures

    def _update(self, a: ThreatAssessment, now: float):
        tid = a.track_id
        if tid not in self.entries:
            self.entries[tid] = MemoryEntry(
                track_id=tid, class_name=a.class_name, state=ObjectState.NEW,
                first_seen=now, last_seen=now, _stable_since=now)
            return
        e = self.entries[tid]
        e.last_seen = now
        e.class_name = a.class_name
        changed = self._meaningful_change(e, a)

        if e.state == ObjectState.NEW and e.announcement_count > 0:
            e.state = ObjectState.ACTIVE
            e._stable_since = now
        elif e.state == ObjectState.ACTIVE:
            if changed:
                e._stable_since = now
            elif (now - e._stable_since) > self.STABLE_AFTER:
                e.state = ObjectState.STABLE
        elif e.state == ObjectState.STABLE:
            if changed:
                e.state = ObjectState.ACTIVE
                e._stable_since = now
            if a.priority == Priority.CRITICAL and e.announced_priority != Priority.CRITICAL:
                e.state = ObjectState.ACTIVE
                e._stable_since = now

    def _should_announce(self, e: MemoryEntry, a: ThreatAssessment, now: float) -> bool:
        if e.state == ObjectState.NEW:
            return True
        if a.priority == Priority.CRITICAL:
            return (now - e.last_announced) >= self.COOLDOWN_CRITICAL
        if e.state == ObjectState.STABLE:
            return (now - e.last_announced) >= self.COOLDOWN_STABLE
        if e.state == ObjectState.ACTIVE:
            if (now - e.last_announced) < self.COOLDOWN_ACTIVE:
                return False
            return self._meaningful_change(e, a)
        return False

    def should_do_summary(self, text: str) -> bool:
        now = time.time()
        if (now - self._last_summary_time) < self.SUMMARY_INTERVAL:
            return False
        if text == self._last_summary_hash:
            return False
        self._last_summary_time = now
        self._last_summary_hash = text
        return True


# ═══════════════════════════════════════════════════════════════════
# Audio Engine (local TTS)
# ═══════════════════════════════════════════════════════════════════

class LocalAudioEngine:
    """Local TTS engine — runs on your machine, not on Modal."""

    def __init__(self, rate=185, volume=1.0):
        self._rate = rate
        self._volume = volume
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._urgent_lock = threading.Lock()
        self._urgent_msg: Optional[str] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._engine_ready = threading.Event()
        self._total_spoken = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="TTS")
        self._thread.start()
        self._engine_ready.wait(timeout=5.0)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def speak_assessments(self, assessments: List[ThreatAssessment]):
        for a in assessments:
            if a.priority == Priority.CRITICAL:
                with self._urgent_lock:
                    self._urgent_msg = a.message
            else:
                self._enqueue(int(a.priority), a.message)

    def speak_departures(self, msgs: List[str]):
        for m in msgs:
            self._enqueue(int(Priority.NORMAL), m)

    def speak_summary(self, text: str):
        self._enqueue(int(Priority.LOW), text)

    def speak_now(self, text: str):
        self._enqueue(1, text)

    def _enqueue(self, pri: int, msg: str):
        try: self._queue.put_nowait((pri, time.time(), msg))
        except queue.Full: pass

    def _loop(self):
        if sys.platform == "win32":
            try:
                import comtypes; comtypes.CoInitialize()
            except:
                try: import pythoncom; pythoncom.CoInitialize()
                except: pass

        try:
            engine = self._init_tts()
            self._engine_ready.set()
        except Exception as e:
            logger.error(f"TTS init failed: {e}")
            self._engine_ready.set()
            return

        while self._running:
            try:
                with self._urgent_lock:
                    if self._urgent_msg:
                        msg = self._urgent_msg
                        self._urgent_msg = None
                        self._speak(engine, msg)
                        continue
                try:
                    pri, ts, msg = self._queue.get(timeout=0.15)
                    if (time.time() - ts) < 10.0:
                        self._speak(engine, msg)
                    self._queue.task_done()
                except queue.Empty:
                    pass
            except Exception as e:
                logger.error(f"TTS error: {e}")
                time.sleep(0.2)

    def _init_tts(self):
        try:
            import comtypes
            from comtypes.client import CreateObject
            voice = CreateObject("SAPI.SpVoice")
            voice.Rate = max(-10, min(10, (self._rate - 150) // 20))
            voice.Volume = int(self._volume * 100)
            logger.info("TTS: Windows SAPI")
            return ("sapi", voice)
        except: pass
        import pyttsx3
        eng = pyttsx3.init()
        eng.setProperty("rate", self._rate)
        eng.setProperty("volume", self._volume)
        logger.info("TTS: pyttsx3")
        return ("pyttsx3", eng)

    def _speak(self, engine_tuple, text: str):
        backend, engine = engine_tuple
        logger.info(f"TTS> {text}")
        if backend == "sapi":
            engine.Speak(text, 0)
        else:
            engine.say(text)
            engine.runAndWait()
            try: engine._inLoop = False
            except: pass
        self._total_spoken += 1


# ═══════════════════════════════════════════════════════════════════
# Visualizer (simplified local version)
# ═══════════════════════════════════════════════════════════════════

PRIORITY_COLORS = {
    Priority.CRITICAL: (0, 0, 255),
    Priority.HIGH: (0, 140, 255),
    Priority.NORMAL: (0, 220, 0),
    Priority.LOW: (180, 180, 180),
}


def draw_frame(frame, server_dets, assessments, extra_info=None):
    """Annotate frame with detection boxes and distances."""
    canvas = frame.copy()
    a_map = {a.track_id: a for a in assessments}
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in server_dets:
        tid = det["track_id"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        a = a_map.get(tid)
        color = PRIORITY_COLORS.get(a.priority, (0, 255, 128)) if a else (0, 255, 128)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        parts = [f"#{tid} {det['class_name']}"]
        if a:
            dist_ft = a.distance * 3.28084
            parts.append(f"{dist_ft:.1f}ft")
            if a.approach_velocity > 0.3:
                parts.append(f"v={a.approach_velocity:.1f}")
        label = " | ".join(parts)

        (tw, lh), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - lh - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        if a and a.priority == Priority.CRITICAL:
            cv2.rectangle(canvas, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), 3)

    # HUD
    if extra_info:
        y = 20
        for k, v in extra_info.items():
            cv2.putText(canvas, f"{k}: {v}", (10, y), font, 0.45,
                        (0, 255, 0), 1, cv2.LINE_AA)
            y += 18

    # Grid lines
    h, w = canvas.shape[:2]
    t = w // 3
    cv2.line(canvas, (t, 0), (t, h), (100, 100, 100), 1)
    cv2.line(canvas, (2 * t, 0), (2 * t, h), (100, 100, 100), 1)

    return canvas


# ═══════════════════════════════════════════════════════════════════
# Modal Remote Caller
# ═══════════════════════════════════════════════════════════════════

class ModalClient:
    """Sends frames to Modal endpoint and receives inference results."""

    def __init__(self, server_url: str, jpeg_quality: int = 75, timeout: float = 15.0):
        self.server_url = server_url.rstrip("/")
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"
        self._consecutive_errors = 0
        self._total_calls = 0
        self._total_ms = 0.0

    def infer(self, frame: np.ndarray) -> Optional[dict]:
        """Send frame to Modal, return result dict or None on error."""
        # Encode frame as JPEG → base64
        ok, jpg = cv2.imencode(".jpg", frame,
                                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return None

        b64 = base64.b64encode(jpg.tobytes()).decode("ascii")

        try:
            t0 = time.perf_counter()
            resp = self._session.post(
                self.server_url,
                json={"frame_b64": b64},
                timeout=self.timeout,
            )
            rtt_ms = (time.perf_counter() - t0) * 1000
            self._total_calls += 1
            self._total_ms += rtt_ms

            if resp.status_code != 200:
                self._consecutive_errors += 1
                logger.error(f"Server returned {resp.status_code}: {resp.text[:200]}")
                return None

            self._consecutive_errors = 0
            result = resp.json()
            result["rtt_ms"] = round(rtt_ms, 1)
            return result

        except requests.exceptions.Timeout:
            self._consecutive_errors += 1
            logger.warning("Server timeout")
            return None
        except requests.exceptions.ConnectionError:
            self._consecutive_errors += 1
            logger.warning("Server connection error — is Modal app running?")
            return None
        except Exception as e:
            self._consecutive_errors += 1
            logger.error(f"Server error: {e}")
            return None

    @property
    def avg_rtt_ms(self) -> float:
        return self._total_ms / self._total_calls if self._total_calls > 0 else 0


# ═══════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════

class DrishtimargaCloudPipeline:
    """
    Local pipeline that:
    1. Grabs frames from ESP32-CAM
    2. Sends to Modal for GPU inference
    3. Tracks objects + filters announcements locally
    4. Does TTS locally
    """

    def __init__(self, source, server_url: str, show_video=True,
                 jpeg_quality=75, speech_rate=185):
        self._running = False

        logger.info("=" * 60)
        logger.info("  DRISHTIMARGA v2 — Cloud Mode (Modal GPU)")
        logger.info("=" * 60)

        self.grabber = FrameGrabber(source, 640, 480)
        self.modal_client = ModalClient(server_url, jpeg_quality=jpeg_quality)
        self.tracker = LocalSpatialTracker()
        self.memory = LocalAnnouncementMemory()
        self.audio = LocalAudioEngine(rate=speech_rate)
        self.show_video = show_video
        self._cycle_times: list = []

    def run(self):
        if not self.grabber.start():
            logger.error("Cannot open camera — exiting")
            return

        self.audio.start()
        time.sleep(0.3)
        self.audio.speak_now("Drishtimarga cloud mode ready. Scanning surroundings.")

        self._running = True
        logger.info("Pipeline running — press 'q' to quit, 's' for summary")
        logger.info("-" * 60)
        skip_counter = 0

        try:
            while self._running:
                t0 = time.perf_counter()

                # 1. Grab frame
                frame, ts = self.grabber.get(timeout=2.0)
                if frame is None:
                    continue

                # 2. Send to Modal for inference
                result = self.modal_client.infer(frame)
                if result is None:
                    # Server unavailable — skip
                    if self.modal_client._consecutive_errors == 3:
                        self.audio.speak_now("Server connection lost. Retrying.")
                    if self.modal_client._consecutive_errors > 20:
                        self.audio.speak_now("Cannot reach server. Stopping.")
                        break
                    continue

                server_dets = result.get("detections", [])

                # 3. Local spatial tracking + threat scoring
                assessments = self.tracker.update(server_dets)

                # 4. Announcement memory filtering
                to_speak, departures = self.memory.filter(assessments)

                # 5. Audio
                self.audio.speak_assessments(to_speak)
                self.audio.speak_departures(departures)

                summary = self.tracker.get_scene_summary()
                if self.memory.should_do_summary(summary):
                    self.audio.speak_summary(summary)

                # 6. Visualization
                if self.show_video:
                    info = {
                        "RTT": f"{result.get('rtt_ms', 0):.0f}ms",
                        "Det": f"{result.get('detection_ms', 0):.0f}ms",
                        "Depth": f"{result.get('depth_ms', 0):.0f}ms",
                        "Objects": len(server_dets),
                        "Avg RTT": f"{self.modal_client.avg_rtt_ms:.0f}ms",
                    }
                    annotated = draw_frame(frame, server_dets, assessments, info)
                    cv2.imshow("Drishtimarga v2 (Cloud)", annotated)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("s"):
                        self.audio.speak_now(summary)

                ms = (time.perf_counter() - t0) * 1000
                self._cycle_times.append(ms)
                if len(self._cycle_times) > 100:
                    self._cycle_times = self._cycle_times[-100:]

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self._shutdown()

    def _shutdown(self):
        self._running = False
        self.grabber.stop()
        self.audio.stop()
        cv2.destroyAllWindows()
        if self._cycle_times:
            avg = sum(self._cycle_times) / len(self._cycle_times)
            logger.info(f"Avg cycle: {avg:.0f}ms, "
                        f"Server calls: {self.modal_client._total_calls}, "
                        f"Avg RTT: {self.modal_client.avg_rtt_ms:.0f}ms")
        logger.info("Drishtimarga stopped.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Drishtimarga v2 — Local client for Modal cloud inference")
    p.add_argument("--source", default=0,
                   help="Camera index (0,1) or ESP32-CAM URL "
                        "(e.g. http://192.168.6.50:81/stream)")
    p.add_argument("--server-url", required=True,
                   help="Modal endpoint URL for the infer endpoint "
                        "(from `modal deploy modal_server.py`)")
    p.add_argument("--jpeg-quality", type=int, default=75,
                   help="JPEG compression quality for frames sent to server (1-100)")
    p.add_argument("--no-display", action="store_true",
                   help="Audio only, no video window")
    p.add_argument("--speech-rate", type=int, default=185)
    return p.parse_args()


def main():
    args = parse_args()

    try:
        source = int(args.source)
    except (ValueError, TypeError):
        source = args.source

    pipeline = DrishtimargaCloudPipeline(
        source=source,
        server_url=args.server_url,
        show_video=not args.no_display,
        jpeg_quality=args.jpeg_quality,
        speech_rate=args.speech_rate,
    )

    def on_signal(sig, frame):
        pipeline._running = False

    signal.signal(signal.SIGINT, on_signal)
    pipeline.run()


if __name__ == "__main__":
    main()
