"""
Drishtimarga — Main Pipeline
Real-time object detection, depth estimation, spatial tracking,
and audio feedback for visually impaired navigation.

Usage:
    python main.py                     # Use webcam with defaults
    python main.py --source 0          # Specific camera index
    python main.py --source "http://192.168.4.1:81/stream"  # ESP32-CAM
    python main.py --no-depth          # Disable depth model (bbox fallback)
    python main.py --no-display        # Headless mode (audio only)
    python main.py --model yolov8s.pt   # Use YOLOv8 small model (default)
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
from detector import ObjectDetector, Detection
from depth_estimator import DepthEstimator
from spatial_engine import SpatialEngine
from audio_engine import AudioEngine
from visualizer import Visualizer

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-14s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("drishtimarga")


class FrameGrabber:
    """
    Threaded frame grabber — always provides the latest frame.
    Drops stale frames to prevent pipeline backup.
    """

    def __init__(self, source, width: int = 640, height: int = 480):
        self.source = source
        self.width = width
        self.height = height
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._total_frames = 0
        self._dropped_frames = 0

    def start(self) -> bool:
        """Open camera and start capture thread."""
        logger.info(f"Opening camera source: {self.source}")
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            logger.error(f"Failed to open camera: {self.source}")
            return False

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened: {actual_w}x{actual_h}")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True,
                                        name="FrameGrabber")
        self._thread.start()
        return True

    def _capture_loop(self):
        """Continuously read frames, keeping only the latest."""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Frame capture failed, retrying...")
                time.sleep(0.1)
                continue

            self._total_frames += 1

            # Drop old frame if queue is full
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                    self._dropped_frames += 1
                except queue.Empty:
                    pass

            self._frame_queue.put((frame, time.time()))

    def get_frame(self, timeout: float = 1.0):
        """Get the latest frame. Returns (frame, timestamp) or (None, None)."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info(f"Camera closed (captured: {self._total_frames}, "
                    f"dropped: {self._dropped_frames})")


class DrishtimargaPipeline:
    """
    Main orchestrator: connects frame capture → detection → depth →
    spatial analysis → threat scoring → audio feedback.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._running = False

        # Initialize components
        logger.info("=" * 60)
        logger.info("  DRISHTIMARGA — Lighting the Path Beyond Sight")
        logger.info("=" * 60)

        logger.info("Initializing components...")

        self.grabber = FrameGrabber(
            source=config.camera.source,
            width=config.camera.width,
            height=config.camera.height,
        )

        self.detector = ObjectDetector(config.detector)
        self.depth = DepthEstimator(config.depth)
        self.spatial = SpatialEngine(
            config.spatial, config.threat, config.navigation)
        self.audio = AudioEngine(config.audio)
        self.visualizer = Visualizer(config.display)

        # Performance tracking
        self._cycle_times = []

    def run(self):
        """Main processing loop."""
        # Start subsystems
        if not self.grabber.start():
            logger.error("Cannot start camera — exiting")
            return

        self.audio.start()
        time.sleep(0.5)
        self.audio.announce_startup()

        self._running = True
        logger.info("Pipeline running — press 'q' to quit")
        logger.info("-" * 60)

        try:
            while self._running:
                cycle_start = time.perf_counter()

                # ── 1. Grab latest frame ──
                frame, timestamp = self.grabber.get_frame(timeout=1.0)
                if frame is None:
                    continue

                # ── 2. Run object detection + tracking ──
                detections = self.detector.detect_and_track(frame)

                # ── 3. Run depth estimation ──
                depth_map = self.depth.estimate(
                    frame) if self.config.depth.enabled else None

                # ── 4. Spatial analysis + threat scoring ──
                assessments = self.spatial.update(
                    detections, depth_map, frame.shape[:2]
                )

                # Debug: log when objects are first detected
                if detections and assessments:
                    for a in assessments:
                        if a.is_new:
                            logger.info(
                                f"NEW: {a.class_name} | {a.distance:.1f}m "
                                f"{a.position} | pri={a.priority.name} | "
                                f"msg=\"{a.message}\""
                            )

                # ── 5. Audio feedback ──
                self.audio.process_assessments(assessments)

                # ── 6. Navigation guidance ──
                # Path-clear, dodge advice, departure notices
                nav_guidance = self.spatial.get_navigation_guidance()
                self.audio.process_navigation_guidance(nav_guidance)

                # Scene summary only if no nav guidance was given this cycle
                if nav_guidance is None:
                    self.audio.maybe_scene_summary(
                        self.spatial.get_scene_summary)

                # ── 7. Visualization ──
                if self.config.display.show_video:
                    extra_info = {
                        "detection_ms": self.detector.inference_ms,
                        "depth_ms": self.depth.inference_ms,
                        "depth_available": self.depth.is_available,
                    }
                    annotated = self.visualizer.draw(
                        frame, detections, assessments, depth_map, extra_info
                    )
                    cv2.imshow("Drishtimarga", annotated)

                    # Show depth map if enabled
                    if self.config.display.show_depth and depth_map is not None:
                        depth_vis = self.depth.get_depth_visualization(
                            depth_map)
                        if depth_vis is not None:
                            cv2.imshow("Depth Map", depth_vis)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Quit requested")
                        break
                    elif key == ord("d"):
                        # Toggle depth display
                        self.config.display.show_depth = not self.config.display.show_depth
                    elif key == ord("s"):
                        # Force scene summary
                        summary = self.spatial.get_scene_summary()
                        logger.info(f"Scene: {summary}")
                        self.audio._enqueue(1, summary)

                # Track cycle time
                cycle_ms = (time.perf_counter() - cycle_start) * 1000
                self._cycle_times.append(cycle_ms)
                if len(self._cycle_times) > 100:
                    self._cycle_times = self._cycle_times[-100:]

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown of all components."""
        self._running = False
        logger.info("-" * 60)
        logger.info("Shutting down...")

        self.grabber.stop()
        self.audio.stop()
        cv2.destroyAllWindows()

        # Print performance summary
        if self._cycle_times:
            avg = sum(self._cycle_times) / len(self._cycle_times)
            p95 = sorted(self._cycle_times)[int(len(self._cycle_times) * 0.95)]
            fps = 1000.0 / avg if avg > 0 else 0
            logger.info(
                f"Performance: avg={avg:.1f}ms, p95={p95:.1f}ms, ~{fps:.1f} FPS")

        audio_stats = self.audio.stats
        logger.info(f"Audio: {audio_stats['total_announcements']} spoken, "
                    f"{audio_stats['total_suppressed']} suppressed, "
                    f"{audio_stats['total_dropped']} dropped stale, "
                    f"{audio_stats['total_nav_guidance']} nav guidance")

        logger.info("Drishtimarga stopped. Stay safe!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drishtimarga — AI Navigation for the Visually Impaired",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Webcam with defaults
  python main.py --source 1                   # External USB camera
  python main.py --source "http://192.168.4.1:81/stream"  # ESP32-CAM
  python main.py --no-depth                   # Faster, no depth model
  python main.py --no-display                 # Audio-only (headless)
  python main.py --model yolov8s.pt           # Better accuracy model
  python main.py --conf 0.4 --size 640        # Custom detection params
        """,
    )
    parser.add_argument("--source", default=0,
                        help="Camera source: index (0,1,2) or URL for ESP32-CAM")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="YOLO model (yolov8n.pt, yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--conf", type=float, default=0.50,
                        help="Detection confidence threshold")
    parser.add_argument("--size", type=int, default=640,
                        help="YOLO input size (320, 416, or 640)")
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable depth model (use bbox-size fallback)")
    parser.add_argument("--depth-model", default="small",
                        choices=["small", "base", "large"],
                        help="Depth Anything V2 relative model variant")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode — audio only, no video window")
    parser.add_argument("--speech-rate", type=int, default=190,
                        help="TTS speech rate (words per minute)")
    parser.add_argument("--cooldown", type=float, default=8.0,
                        help="Seconds between re-announcing same object (default: 8)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Compute device")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build configuration
    config = AppConfig()

    # Camera
    try:
        config.camera.source = int(args.source)
    except (ValueError, TypeError):
        config.camera.source = args.source  # URL string

    # Detector
    config.detector.model_name = args.model
    config.detector.confidence = args.conf
    config.detector.input_size = args.size
    config.detector.device = args.device

    # Depth
    config.depth.enabled = not args.no_depth
    config.depth.model_name = args.depth_model
    config.depth.device = args.device

    # Audio
    config.audio.rate = args.speech_rate
    config.audio.cooldown = args.cooldown

    # Display
    config.display.show_video = not args.no_display

    # Handle graceful shutdown on Ctrl+C
    def signal_handler(sig, frame):
        logger.info("Signal received, shutting down...")
        pipeline._running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Run pipeline
    pipeline = DrishtimargaPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
