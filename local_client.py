"""
Drishtimarga v3 — Local Client for Modal Cloud Inference

Captures frames from ESP32-CAM (or webcam), sends to Modal GPU server,
receives detection results + ElevenLabs audio, plays audio locally.

The server handles ALL heavy computation:
  YOLO detection → Depth estimation → GPT analysis → ElevenLabs TTS

This client just captures, sends, plays, and visualizes.

Usage:
    # First deploy the server:
    modal secret create drishtimarga-secrets \\
        OPENAI_API_KEY="sk-proj-..." \\
        ELEVENLABS_API_KEY="sk_..."
    modal deploy modal_server.py

    # Then run client:
    python local_client.py --source 0 --server-url https://YOUR--drishtimarga-v3-drishtimargainference-infer.modal.run
    python local_client.py --source http://192.168.6.50:81/stream --server-url <URL>
    python local_client.py --source http://192.168.6.50/capture --server-url <URL> --no-display
"""

import argparse
import base64
import io
import json
import logging
import os
import queue
import signal
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

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
# Audio Player — plays MP3 bytes received from server
# ═══════════════════════════════════════════════════════════════════

class AudioPlayer:
    """
    Non-blocking audio player for ElevenLabs MP3 output.
    Uses platform-appropriate playback (pygame, playsound, or ffplay).
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=5)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._total_played = 0
        self._backend = None
        self._init_backend()

    def _init_backend(self):
        """Find available audio playback backend."""
        # Try pygame first (best cross-platform)
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2048)
            self._backend = "pygame"
            logger.info("Audio backend: pygame")
            return
        except Exception:
            pass

        # Try pydub + simpleaudio
        try:
            from pydub import AudioSegment
            from pydub.playback import play
            self._backend = "pydub"
            logger.info("Audio backend: pydub")
            return
        except Exception:
            pass

        # Fallback: system command (ffplay/afplay/aplay)
        if sys.platform == "darwin":
            self._backend = "afplay"
        elif sys.platform == "win32":
            self._backend = "winmm"
        else:
            self._backend = "ffplay"
        logger.info(f"Audio backend: {self._backend} (system)")

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._playback_loop, daemon=True, name="AudioPlayer")
        self._thread.start()
        logger.info("Audio player started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def play_audio_b64(self, audio_b64: str, urgency: str = "low"):
        """Queue base64 MP3 audio for playback."""
        try:
            audio_bytes = base64.b64decode(audio_b64)
            # Priority: critical=0, high=1, medium=2, low=3
            pri_map = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}
            pri = pri_map.get(urgency, 3)

            # For critical: clear queue and play immediately
            if urgency == "critical":
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break

            self._queue.put_nowait((pri, time.time(), audio_bytes))
        except queue.Full:
            logger.debug("Audio queue full, dropping")
        except Exception as e:
            logger.error(f"Audio queue error: {e}")

    def _playback_loop(self):
        """Background loop that plays queued audio."""
        while self._running:
            try:
                pri, ts, audio_bytes = self._queue.get(timeout=0.2)

                # Drop old audio (>12 seconds old)
                if time.time() - ts > 12.0:
                    continue

                self._play_bytes(audio_bytes)
                self._total_played += 1
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Playback error: {e}")
                time.sleep(0.1)

    def _play_bytes(self, audio_bytes: bytes):
        """Play MP3 bytes using available backend."""
        if self._backend == "pygame":
            self._play_pygame(audio_bytes)
        elif self._backend == "pydub":
            self._play_pydub(audio_bytes)
        else:
            self._play_system(audio_bytes)

    def _play_pygame(self, audio_bytes: bytes):
        import pygame
        try:
            buf = io.BytesIO(audio_bytes)
            pygame.mixer.music.load(buf, "mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        except Exception as e:
            logger.error(f"pygame playback error: {e}")

    def _play_pydub(self, audio_bytes: bytes):
        from pydub import AudioSegment
        from pydub.playback import play
        try:
            buf = io.BytesIO(audio_bytes)
            audio = AudioSegment.from_mp3(buf)
            play(audio)
        except Exception as e:
            logger.error(f"pydub playback error: {e}")

    def _play_system(self, audio_bytes: bytes):
        """Fallback: write to temp file and play with system command."""
        import subprocess
        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.write(audio_bytes)
            tmp.close()

            if self._backend == "afplay":
                subprocess.run(["afplay", tmp.name], timeout=15,
                               capture_output=True)
            elif self._backend == "winmm":
                # Windows: use MediaPlayer via PowerShell (supports MP3)
                # Robust script that waits for media to load before checking duration
                ps_cmd = (
                    f"$p = New-Object System.Windows.Media.MediaPlayer; "
                    f"$p.Open('{tmp.name}'); "
                    f"$p.Volume = 1.0; "
                    f"Start-Sleep -m 200; "
                    f"$i = 0; while ($p.NaturalDuration.HasTimeSpan -eq $false -and $i++ -lt 20) {{ Start-Sleep -m 100 }}; "
                    f"$p.Play(); "
                    f"if ($p.NaturalDuration.HasTimeSpan) {{ "
                    f"  while($p.Position -lt $p.NaturalDuration.TimeSpan) {{ Start-Sleep -m 100 }} "
                    f"}} else {{ Start-Sleep -s 5 }}; "
                    f"Start-Sleep -m 500"
                )
                subprocess.run(
                    ["powershell", "-c", "Add-Type -AssemblyName PresentationCore; " + ps_cmd],
                    timeout=25, capture_output=True)
            else:
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp.name],
                    timeout=15, capture_output=True)
        except Exception as e:
            logger.error(f"System playback error: {e}")
        finally:
            if tmp:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

    @property
    def stats(self):
        return {
            "total_played": self._total_played,
            "queue_size": self._queue.qsize(),
            "backend": self._backend,
        }


# ═══════════════════════════════════════════════════════════════════
# Frame Grabber (ESP32-CAM + webcam support)
# ═══════════════════════════════════════════════════════════════════

class FrameGrabber:
    """Threaded capture for webcam or ESP32-CAM with auto-reconnect."""

    ESP32_RESOLUTIONS = {
        "UXGA": (1600, 1200, 13), "SXGA": (1280, 1024, 12),
        "XGA": (1024, 768, 10), "SVGA": (800, 600, 9),
        "VGA": (640, 480, 8), "CIF": (400, 296, 6),
        "QVGA": (320, 240, 5),
    }

    def __init__(self, source, width=640, height=480, rotate=False):
        self.source = source
        self.width = width
        self.height = height
        self.rotate = rotate
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
            try:
                self._cap.release()
            except Exception:
                pass
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
        except Exception:
            pass

    def _fetch_snapshot(self) -> Optional[np.ndarray]:
        try:
            import urllib.request
            resp = urllib.request.urlopen(self.source, timeout=5)
            data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _loop(self):
        fails = 0
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                fails += 1
                if self._is_stream and fails > 10:
                    if not self._reconnect():
                        break
                    fails = 0
                else:
                    time.sleep(0.05)
                continue
            fails = 0
            self._total += 1
            if self._q.full():
                try:
                    self._q.get_nowait()
                    self._dropped += 1
                except queue.Empty:
                    pass
            if self.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self._q.put((frame, time.time()))

    def _loop_snapshot(self):
        fails = 0
        while self._running:
            frame = self._fetch_snapshot()
            if frame is None:
                fails += 1
                if fails > 20:
                    break
                time.sleep(min(2.0, 0.1 * fails))
                continue
            fails = 0
            self._total += 1
            if self._q.full():
                try:
                    self._q.get_nowait()
                    self._dropped += 1
                except queue.Empty:
                    pass
            if self.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self._q.put((frame, time.time()))

    def _reconnect(self) -> bool:
        self._reconnect_count += 1
        if self._reconnect_count > 50:
            return False
        backoff = min(10.0, 0.5 * (2 ** min(self._reconnect_count, 5)))
        logger.warning(f"Stream lost — reconnect in {backoff:.1f}s")
        time.sleep(backoff)
        try:
            if self._cap:
                self._cap.release()
            self._cap = cv2.VideoCapture(self.source)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return True
        except Exception:
            pass
        return self._reconnect()

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


# ═══════════════════════════════════════════════════════════════════
# Modal Remote Caller
# ═══════════════════════════════════════════════════════════════════

class ModalClient:
    """Sends frames to Modal endpoint and receives inference results + audio."""

    def __init__(self, server_url: str, jpeg_quality: int = 60, timeout: float = 60.0):
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
                logger.error(f"Server {resp.status_code}: {resp.text[:200]}")
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
# Visualization
# ═══════════════════════════════════════════════════════════════════

def draw_frame(frame: np.ndarray, detections: list, info: dict,
               last_announcement: str = "") -> np.ndarray:
    """Draw bounding boxes, distances, and HUD on frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    COLORS = {
        "critical": (0, 0, 255),
        "high": (0, 140, 255),
        "medium": (0, 220, 220),
        "low": (0, 220, 0),
    }

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        class_name = det["class_name"]
        dist_ft = det["distance_ft"]
        vel = det.get("velocity_mps", 0)
        pos = det["position"]
        tid = det["track_id"]

        # Color based on danger level
        if dist_ft < 5 and vel > 0.3:
            color = COLORS["critical"]
        elif vel > 0.3:
            color = COLORS["high"]
        elif dist_ft < 10:
            color = COLORS["medium"]
        else:
            color = COLORS["low"]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"#{tid} {class_name} {dist_ft:.0f}ft"
        if vel > 0.3:
            label += f" v={vel:.1f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.45
        (tw, lh), _ = cv2.getTextSize(label, font, fs, 1)
        cv2.rectangle(canvas, (x1, y1 - lh - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), font, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Critical pulsing
        if dist_ft < 5 and vel > 0.3:
            cv2.rectangle(canvas, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                          (0, 0, 255), 3)

    # ── HUD ──
    bg = np.zeros((90, 340, 3), dtype=np.uint8)
    bg[:] = (20, 20, 20)
    bh, bw = bg.shape[:2]
    if bh <= h and bw <= w:
        canvas[0:bh, 0:bw] = cv2.addWeighted(canvas[0:bh, 0:bw], 0.3, bg, 0.7, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 18
    rtt = info.get("rtt_ms", 0)
    if isinstance(rtt, str):
        rtt_str = rtt
    else:
        rtt_str = f"{rtt:.0f}ms"
    cv2.putText(canvas, f"RTT: {rtt_str} | Objects: {len(detections)}",
                (10, y), font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    y += 18
    det_ms = info.get("detection_ms", info.get("Det", 0))
    dep_ms = info.get("depth_ms", info.get("Depth", 0))
    cv2.putText(canvas, f"Det: {det_ms}ms | Depth: {dep_ms}ms",
                (10, y), font, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(canvas, f"Avg RTT: {info.get('avg_rtt', '?')}",
                (10, y), font, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    # Last announcement (bottom of screen)
    if last_announcement:
        y_ann = h - 15
        ann_bg = np.zeros((35, w, 3), dtype=np.uint8)
        ann_bg[:] = (30, 30, 30)
        if y_ann - 25 >= 0:
            canvas[y_ann - 25:y_ann + 10, 0:w] = cv2.addWeighted(
                canvas[y_ann - 25:y_ann + 10, 0:w], 0.3, ann_bg[:35, :w], 0.7, 0)
        # Try to render Nepali (may show as boxes on systems without Nepali fonts)
        cv2.putText(canvas, last_announcement[:80], (10, y_ann),
                    font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # Grid lines
    t = w // 3
    cv2.line(canvas, (t, 0), (t, h), (100, 100, 100), 1)
    cv2.line(canvas, (2 * t, 0), (2 * t, h), (100, 100, 100), 1)

    return canvas


# ═══════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════

class DrishtimargaCloudPipeline:
    """
    Local pipeline:
    1. Grabs frames from ESP32-CAM / webcam
    2. Sends to Modal for GPU inference + LLM + TTS
    3. Plays received ElevenLabs audio locally
    4. Shows annotated video
    """

    def __init__(self, source, server_url: str, show_video=True,
                 jpeg_quality=75, rotate=False):
        self._running = False

        logger.info("=" * 60)
        logger.info("  DRISHTIMARGA v3 — Cloud Mode")
        logger.info("  YOLO11x + Depth + GPT-4o-mini + ElevenLabs")
        logger.info("=" * 60)

        self.grabber = FrameGrabber(source, 480, 360, rotate=rotate)
        self.modal_client = ModalClient(server_url, jpeg_quality=jpeg_quality)
        self.audio = AudioPlayer()
        self.show_video = show_video
        self._cycle_times: list = []
        self._last_announcement = ""
        self._frames_sent = 0

    def run(self):
        if not self.grabber.start():
            logger.error("Cannot open camera — exiting")
            return

        self.audio.start()
        self._running = True
        logger.info("Pipeline running — press 'q' to quit")
        logger.info("-" * 60)

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
                    if self.modal_client._consecutive_errors == 3:
                        logger.warning("Server connection lost. Retrying...")
                    if self.modal_client._consecutive_errors > 20:
                        logger.error("Cannot reach server. Stopping.")
                        break
                    continue

                self._frames_sent += 1
                detections = result.get("detections", [])

                # 3. Play audio if server returned any
                audio_b64 = result.get("audio_b64")
                llm_text = result.get("llm_text", "")
                urgency = result.get("urgency", "none")
                total_ms = result.get("total_ms", 0)

                # DIAGNOSTIC: Always log what server returned
                if llm_text:
                    logger.info(f"📝 Server LLM: '{llm_text}' | audio={'YES' if audio_b64 else 'NO'} | total={total_ms}ms")

                if audio_b64:
                    self.audio.play_audio_b64(audio_b64, urgency)
                    self._last_announcement = llm_text
                    logger.info(f"🔊 [{urgency}] {llm_text}")
                elif llm_text:
                    logger.warning(f"⚠️ Got LLM text but NO audio: '{llm_text[:60]}'")

                # 4. Visualization
                if self.show_video:
                    info = {
                        "rtt_ms": result.get("rtt_ms", 0),
                        "detection_ms": result.get("detection_ms", 0),
                        "depth_ms": result.get("depth_ms", 0),
                        "avg_rtt": f"{self.modal_client.avg_rtt_ms:.0f}ms",
                    }
                    annotated = draw_frame(
                        frame, detections, info,
                        last_announcement=self._last_announcement)
                    cv2.imshow("Drishtimarga v3", annotated)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

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
                        f"Frames sent: {self._frames_sent}, "
                        f"Avg RTT: {self.modal_client.avg_rtt_ms:.0f}ms")
        audio_stats = self.audio.stats
        logger.info(f"Audio: {audio_stats['total_played']} played "
                    f"(backend: {audio_stats['backend']})")
        logger.info("Drishtimarga stopped.")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Drishtimarga v3 — Local client for Modal cloud inference")
    p.add_argument("--source", default=0,
                   help="Camera index (0,1) or ESP32-CAM URL")
    p.add_argument("--server-url", required=True,
                   help="Modal infer endpoint URL")
    p.add_argument("--jpeg-quality", type=int, default=75,
                   help="JPEG quality for frames sent to server (1-100)")
    p.add_argument("--no-display", action="store_true",
                   help="Audio only, no video window")
    p.add_argument("--rotate", action="store_true",
                   help="Rotate frames 90° counterclockwise (for mounted ESP32)")
    return p.parse_args()


def main():
    args = parse_args()

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  CAMERA SOURCE TOGGLE — comment/uncomment to switch      ║
    # ╚═══════════════════════════════════════════════════════════╝
    CAMERA_SOURCE = 0                                          # Laptop webcam
    # CAMERA_SOURCE = "http://192.168.6.50:81/stream"          # ESP32-CAM

    # Override with CLI --source if provided, otherwise use toggle above
    if args.source != 0:
        try:
            source = int(args.source)
        except (ValueError, TypeError):
            source = args.source
    else:
        source = CAMERA_SOURCE

    pipeline = DrishtimargaCloudPipeline(
        source=source,
        server_url=args.server_url,
        show_video=not args.no_display,
        jpeg_quality=args.jpeg_quality,
        rotate=args.rotate,
    )

    def on_signal(sig, frame):
        pipeline._running = False

    signal.signal(signal.SIGINT, on_signal)
    pipeline.run()


if __name__ == "__main__":
    main()
