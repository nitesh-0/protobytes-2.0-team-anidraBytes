"""
Drishtimarga v2 — Main Pipeline

Changes from v1:
  • Detection:   YOLOv11x (56.1 mAP) replaces YOLOv8n (37.3 mAP)
  • Depth:       Metric Depth Anything V2 (outputs real meters)
  • Audio:       AnnouncementMemory state machine eliminates repetition
  • Camera:      Full ESP32-CAM support (MJPEG stream + snapshot mode)

Usage:
    python main.py                                          # webcam
    python main.py --source http://192.168.1.50:81/stream   # ESP32-CAM MJPEG
    python main.py --source http://192.168.1.50/capture     # ESP32-CAM snapshot
    python main.py --no-depth                               # skip depth model
    python main.py --model yolo11n.pt                       # lighter YOLO
    python main.py --no-display                             # audio only
"""

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import AppConfig
from detector import ObjectDetector
from depth_estimator import DepthEstimator
from spatial_engine import SpatialEngine
from announcement_memory import AnnouncementMemory
from audio_engine import AudioEngine
from visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-14s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("drishtimarga")


class FrameGrabber:
    """
    Threaded capture — always returns the freshest frame.
    Supports:
      - Local webcam (source=0, 1, …)
      - ESP32-CAM MJPEG stream (source="http://<ip>:81/stream")
      - ESP32-CAM snapshot URL  (source="http://<ip>/capture")
    Auto-reconnects on stream failure with exponential back-off.
    """

    # ESP32-CAM resolution map  (set via http://<ip>/control?var=framesize&val=N)
    ESP32_RESOLUTIONS = {
        "UXGA": (1600, 1200, 13), "SXGA": (1280, 1024, 12),
        "XGA":  (1024, 768, 10),  "SVGA": (800, 600, 9),
        "VGA":  (640, 480, 8),    "CIF":  (400, 296, 6),
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
        self._max_reconnects = 50

        if self._is_stream:
            self._base_url = self._extract_base_url(source)
            self._is_snapshot = any(
                k in source.lower() for k in ["/capture", "/cam-hi", "/cam-lo", "/jpg"])

    @staticmethod
    def _extract_base_url(url: str) -> str:
        """Extract http://<ip>:<port> from a full ESP32-CAM URL."""
        from urllib.parse import urlparse
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}"

    def start(self) -> bool:
        logger.info(f"Opening camera: {self.source}")

        if self._is_stream and not self._is_snapshot:
            # MJPEG stream — configure ESP32 resolution first
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
        """Open (or reopen) the VideoCapture."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

        if self._is_snapshot:
            # Snapshot mode — test one fetch
            frame = self._fetch_snapshot()
            if frame is not None:
                h, w = frame.shape[:2]
                logger.info(f"ESP32-CAM snapshot ready: {w}x{h}")
                return True
            logger.error(f"Cannot fetch snapshot from: {self.source}")
            return False

        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            logger.error(f"Cannot open camera: {self.source}")
            return False

        if not self._is_stream:
            # Only set resolution for local cameras (useless for HTTP streams)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera ready: {w}x{h} ({'stream' if self._is_stream else 'local'})")
        return True

    def _esp32_set_resolution(self):
        """Tell the ESP32-CAM to switch to the closest resolution via its HTTP API."""
        if not self._base_url:
            return
        # Find best matching resolution
        best_name, best_val = "VGA", 8
        best_diff = 999999
        for name, (rw, rh, val) in self.ESP32_RESOLUTIONS.items():
            diff = abs(rw - self.width) + abs(rh - self.height)
            if diff < best_diff:
                best_diff = diff
                best_name, best_val = name, val
        try:
            import urllib.request
            url = f"{self._base_url}/control?var=framesize&val={best_val}"
            urllib.request.urlopen(url, timeout=3)
            logger.info(f"ESP32-CAM resolution set to {best_name} (val={best_val})")
        except Exception as e:
            logger.warning(f"Could not set ESP32-CAM resolution: {e}")

    def _fetch_snapshot(self) -> Optional[np.ndarray]:
        """Fetch a single JPEG frame from ESP32-CAM snapshot URL."""
        try:
            import urllib.request
            resp = urllib.request.urlopen(self.source, timeout=5)
            data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def _loop(self):
        """Capture loop for local camera or MJPEG stream (with auto-reconnect)."""
        consecutive_fails = 0
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                consecutive_fails += 1
                if self._is_stream and consecutive_fails > 10:
                    # Stream likely died — reconnect
                    if not self._reconnect():
                        break
                    consecutive_fails = 0
                else:
                    time.sleep(0.05)
                continue

            consecutive_fails = 0
            self._total += 1
            if self._q.full():
                try:
                    self._q.get_nowait()
                    self._dropped += 1
                except queue.Empty:
                    pass
            self._q.put((frame, time.time()))

    def _loop_snapshot(self):
        """Capture loop using repeated HTTP snapshot requests (most reliable for ESP32-CAM)."""
        consecutive_fails = 0
        while self._running:
            frame = self._fetch_snapshot()
            if frame is None:
                consecutive_fails += 1
                if consecutive_fails > 20:
                    logger.error("ESP32-CAM snapshot: too many failures, stopping")
                    break
                backoff = min(2.0, 0.1 * consecutive_fails)
                time.sleep(backoff)
                continue

            consecutive_fails = 0
            self._total += 1
            if self._q.full():
                try:
                    self._q.get_nowait()
                    self._dropped += 1
                except queue.Empty:
                    pass
            self._q.put((frame, time.time()))

    def _reconnect(self) -> bool:
        """Try to reconnect to the ESP32-CAM stream with exponential back-off."""
        self._reconnect_count += 1
        if self._reconnect_count > self._max_reconnects:
            logger.error("Max reconnection attempts reached — giving up")
            return False

        backoff = min(10.0, 0.5 * (2 ** min(self._reconnect_count, 5)))
        logger.warning(f"Stream lost — reconnecting in {backoff:.1f}s "
                       f"(attempt {self._reconnect_count}/{self._max_reconnects})")
        time.sleep(backoff)

        try:
            if self._cap:
                self._cap.release()
            self._cap = cv2.VideoCapture(self.source)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info("Reconnected to stream")
                return True
        except Exception as e:
            logger.error(f"Reconnect failed: {e}")
        return self._reconnect()  # retry recursively (bounded by max_reconnects)

    def get(self, timeout=1.0):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info(f"Camera: {self._total} captured, {self._dropped} dropped"
                    f"{f', {self._reconnect_count} reconnects' if self._reconnect_count else ''}")


class DrishtimargaPipeline:
    """Main orchestrator connecting all components."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._running = False

        logger.info("=" * 60)
        logger.info("  DRISHTIMARGA v2")
        logger.info("  Lighting the Path Beyond Sight")
        logger.info("=" * 60)

        self.grabber = FrameGrabber(
            config.camera.source, config.camera.width, config.camera.height)
        self.detector = ObjectDetector(config.detector)
        self.depth = DepthEstimator(config.depth)
        self.spatial = SpatialEngine(config.spatial, config.threat)
        self.memory = AnnouncementMemory(config.announcement)
        self.audio = AudioEngine(config.audio)
        self.vis = Visualizer(config.display)
        self._cycle_times: list = []

    def run(self):
        if not self.grabber.start():
            return

        self.audio.start()
        time.sleep(0.5)
        self.audio.announce_startup()

        self._running = True
        logger.info("Pipeline running — press 'q' to quit, 'd' depth, 's' summary")
        logger.info("-" * 60)

        try:
            while self._running:
                t0 = time.perf_counter()

                # 1. Grab frame
                frame, ts = self.grabber.get(timeout=1.0)
                if frame is None:
                    continue

                # 2. Detect + track
                detections = self.detector.detect_and_track(frame)

                # 3. Depth
                depth_map = None
                if self.config.depth.enabled:
                    depth_map = self.depth.estimate(frame)

                # 4. Spatial analysis (pass depth_estimator for metric distance)
                assessments = self.spatial.update(
                    detections, depth_map, frame.shape[:2], self.depth)

                # 5. Announcement memory filtering (THE key change for anti-repetition)
                to_speak, departures = self.memory.filter_assessments(assessments)

                # 6. Audio
                self.audio.speak_assessments(to_speak)
                self.audio.speak_departures(departures)

                # Scene summary (only if scene actually changed)
                summary = self.spatial.get_scene_summary()
                if self.memory.should_do_scene_summary(summary):
                    self.audio.speak_summary(summary)

                # 7. Visualization
                if self.config.display.show_video:
                    info = {
                        "detection_ms": self.detector.inference_ms,
                        "depth_ms": self.depth.inference_ms,
                        "depth_metric": self.depth.is_metric,
                    }
                    annotated = self.vis.draw(
                        frame, detections, assessments,
                        memory=self.memory, depth_map=depth_map,
                        extra_info=info)
                    cv2.imshow("Drishtimarga v2", annotated)

                    if self.config.display.show_depth and depth_map is not None:
                        dv = self.depth.get_visualization(depth_map)
                        if dv is not None:
                            cv2.imshow("Depth", dv)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("d"):
                        self.config.display.show_depth = not self.config.display.show_depth
                    elif key == ord("s"):
                        self.audio.speak_now(self.spatial.get_scene_summary())

                # Track performance
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
        logger.info("-" * 60)
        self.grabber.stop()
        self.audio.stop()
        cv2.destroyAllWindows()

        if self._cycle_times:
            avg = sum(self._cycle_times) / len(self._cycle_times)
            p95 = sorted(self._cycle_times)[int(len(self._cycle_times) * 0.95)]
            logger.info(f"Perf: avg={avg:.1f}ms, p95={p95:.1f}ms, "
                        f"~{1000/avg:.1f} FPS")

        a = self.audio.stats
        m = self.memory.stats
        logger.info(f"Audio: {a['total_spoken']} spoken, {a['total_dropped']} dropped")
        logger.info(f"Memory: {m['total_tracked']} tracked, "
                    f"states={m['states']}, "
                    f"{m['departures_announced']} departures")
        logger.info("Drishtimarga stopped.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Drishtimarga v2 — AI Navigation for the Visually Impaired")
    p.add_argument("--source", default=0,
                   help="Camera index (0,1) or ESP32-CAM URL "
                        "(e.g. http://192.168.1.50:81/stream or "
                        "http://192.168.1.50/capture)")
    p.add_argument("--model", default="yolo11x.pt",
                   help="YOLO model (yolo11x.pt, yolo11l.pt, yolo11n.pt, yolov8n.pt)")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--size", type=int, default=640,
                   help="Detection input size (416, 640)")
    p.add_argument("--no-depth", action="store_true",
                   help="Disable depth model")
    p.add_argument("--depth-model", default="metric-outdoor-base",
                   help="Depth model variant (see depth_estimator.py for list)")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--speech-rate", type=int, default=185)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = AppConfig()

    # Camera
    try:
        cfg.camera.source = int(args.source)
    except (ValueError, TypeError):
        cfg.camera.source = args.source

    # Detector
    cfg.detector.model_name = args.model
    cfg.detector.confidence = args.conf
    cfg.detector.input_size = args.size
    cfg.detector.device = args.device

    # Depth
    cfg.depth.enabled = not args.no_depth
    cfg.depth.model_name = args.depth_model
    cfg.depth.device = args.device

    # Audio
    cfg.audio.rate = args.speech_rate

    # Display
    cfg.display.show_video = not args.no_display

    pipeline = DrishtimargaPipeline(cfg)

    def on_signal(sig, frame):
        pipeline._running = False

    signal.signal(signal.SIGINT, on_signal)
    pipeline.run()


if __name__ == "__main__":
    main()
