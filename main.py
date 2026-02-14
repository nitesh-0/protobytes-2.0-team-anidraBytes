"""
Drishtimarga v3 — Local Pipeline (no Modal)

Runs everything locally:
  YOLO11x → Depth Anything V2 Metric → GPT-4o-mini → ElevenLabs TTS

Requires a GPU for real-time performance.
For cloud deployment, use modal_server.py + local_client.py instead.

Usage:
    export OPENAI_API_KEY="sk-proj-..."
    export ELEVENLABS_API_KEY="sk_..."

    python main.py                                          # webcam
    python main.py --source http://192.168.1.50:81/stream   # ESP32-CAM
    python main.py --no-depth                               # skip depth model
    python main.py --model yolo11n.pt                       # lighter YOLO
    python main.py --no-display                             # audio only
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
from typing import Optional

import cv2
import numpy as np

from config import AppConfig
from detector import ObjectDetector
from depth_estimator import DepthEstimator
from spatial_engine import SpatialEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-14s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("drishtimarga")


# ═══════════════════════════════════════════════════════════════════
# LLM Analyzer — calls GPT-4o-mini for intelligent Nepali announcements
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """तपाईं एक दृष्टिविहीन व्यक्तिको लागि नेभिगेसन सहायक हुनुहुन्छ। तपाईंले वस्तु पत्ता लगाउने प्रणालीबाट डेटा प्राप्त गर्नुहुन्छ।

तपाईंको काम:
1. Detection data विश्लेषण गर्नुहोस् (objects, distances in feet, positions, velocities)
2. केवल महत्त्वपूर्ण कुरा बोल्नुहोस् — जस्तै खतरा, नजिक आउँदै गरेको वस्तु, नयाँ वस्तु, बाटोमा रोकावट
3. नेपालीमा अति छोटो वाक्य दिनुहोस् (५-१५ शब्द मात्र)
4. यदि कुनै महत्त्वपूर्ण परिवर्तन छैन भने खाली string "" फर्काउनुहोस्

नियमहरू:
- दूरी फिटमा भन्नुहोस्
- स्थिति: बायाँ, दायाँ, अगाडि (center = अगाडि)
- खतरा: "सावधान!" वा "खतरा!" प्रयोग गर्नुहोस्
- नजिक आउँदैछ भने भन्नुहोस्, तर स्थिर वस्तुको बारेमा बारम्बार नभन्नुहोस्
- "बाटो खुला छ" भन्नुहोस् यदि अगाडि केही छैन भने (तर बारम्बार नभन्नुहोस्)
- एउटै वस्तु बारेमा बारम्बार नबोल्नुहोस् जबसम्म यो नजिक नआउँदैछ वा खतरा बढ्दैन

उदाहरणहरू:
- "अगाडि मान्छे, ६ फिट"
- "सावधान! गाडी दायाँबाट आउँदैछ"
- "खतरा! ट्रक नजिक, ३ फिट अगाडि!"
- "कुकुर बायाँतिर"
- "बाटो खुला छ"
- "" (कुनै परिवर्तन छैन भने खाली)

तपाईंले JSON response दिनुहोस्:
{"speak": "नेपालीमा छोटो वाक्य वा खाली string", "urgency": "none|low|medium|high|critical"}"""


class LLMAnalyzer:
    """Calls GPT-4o-mini to analyze detections and generate Nepali announcements."""

    ANALYSIS_INTERVAL = 2.5  # seconds between LLM calls

    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self._last_call_time = 0.0
        self._last_response = ""
        self._scene_history = []
        self._consecutive_empty = 0

    def should_analyze(self, detections: list) -> bool:
        """Check if we should call the LLM this cycle."""
        now = time.time()
        elapsed = now - self._last_call_time

        # Always analyze if critical threat
        for d in detections:
            dist_ft = d.get("distance_ft", 100)
            vel = d.get("velocity_mps", 0)
            if dist_ft < 5.0 and vel > 0.5:
                return True
            if dist_ft < 3.0:
                return True

        return elapsed >= self.ANALYSIS_INTERVAL

    def analyze(self, detections: list) -> tuple:
        """
        Call GPT-4o-mini. Returns (nepali_text, urgency).
        Returns ("", "none") if nothing to say.
        """
        now = time.time()
        self._last_call_time = now

        try:
            scene_data = self._build_scene(detections)
            context = ""
            if self._scene_history:
                last = self._scene_history[-1]
                context = f"\nPrevious scene ({now - last['time']:.1f}s ago): {last['summary']}"
                if self._last_response:
                    context += f"\nLast announcement: \"{self._last_response}\""

            user_msg = f"Current scene:\n{scene_data}\n{context}\n\nJSON only:"

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=150,
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            text = result.get("speak", "").strip()
            urgency = result.get("urgency", "none")

            # Update history
            summary = "; ".join(
                f"{d['class_name']}@{d['distance_ft']}ft/{d['position']}"
                for d in detections[:5]
            ) if detections else "empty"
            self._scene_history.append({"time": now, "summary": summary})
            if len(self._scene_history) > 5:
                self._scene_history = self._scene_history[-5:]

            if text:
                self._last_response = text
                self._consecutive_empty = 0
            else:
                self._consecutive_empty += 1

            return text, urgency

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "", "none"

    def _build_scene(self, detections: list) -> str:
        if not detections:
            return "No objects detected. Scene is clear."
        lines = []
        for d in sorted(detections, key=lambda x: x["distance_ft"])[:8]:
            parts = [d["class_name"], f"#{d['track_id']}", f"{d['distance_ft']}ft", d["position"]]
            vel = d.get("velocity_mps", 0)
            if vel > 0.3:
                parts.append(f"approaching@{vel:.1f}m/s")
                ttc = d.get("distance_m", 10) / vel if vel > 0 else 99
                if ttc < 10:
                    parts.append(f"TTC={ttc:.1f}s")
            lines.append(" | ".join(parts))
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# ElevenLabs TTS Engine
# ═══════════════════════════════════════════════════════════════════

ELEVENLABS_VOICE_ID = "pFZP5JQG7iQjIQuC4Bku"  # "Lily"
ELEVENLABS_MODEL = "eleven_multilingual_v2"


class ElevenLabsTTS:
    """Converts Nepali text to speech via ElevenLabs API."""

    def __init__(self, api_key: str, voice_id: str = ELEVENLABS_VOICE_ID):
        self.api_key = api_key
        self.voice_id = voice_id

    def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to MP3 bytes. Returns None on error."""
        import requests
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            }
            payload = {
                "text": text,
                "model_id": ELEVENLABS_MODEL,
                "voice_settings": {
                    "stability": 0.6,
                    "similarity_boost": 0.8,
                    "style": 0.2,
                    "use_speaker_boost": True,
                },
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.content
            else:
                logger.error(f"ElevenLabs {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            logger.error(f"ElevenLabs error: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════
# Audio Player (same as in local_client.py, simplified)
# ═══════════════════════════════════════════════════════════════════

class AudioPlayer:
    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=5)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._backend = self._find_backend()

    def _find_backend(self):
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2048)
            return "pygame"
        except Exception:
            pass
        try:
            from pydub import AudioSegment
            return "pydub"
        except Exception:
            pass
        return "ffplay" if sys.platform != "darwin" else "afplay"

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def play(self, audio_bytes: bytes):
        try:
            self._queue.put_nowait((time.time(), audio_bytes))
        except queue.Full:
            pass

    def _loop(self):
        while self._running:
            try:
                ts, audio_bytes = self._queue.get(timeout=0.2)
                if time.time() - ts > 8.0:
                    continue
                self._play_bytes(audio_bytes)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Play error: {e}")

    def _play_bytes(self, audio_bytes: bytes):
        if self._backend == "pygame":
            import pygame
            buf = io.BytesIO(audio_bytes)
            pygame.mixer.music.load(buf, "mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        elif self._backend == "pydub":
            from pydub import AudioSegment
            from pydub.playback import play
            buf = io.BytesIO(audio_bytes)
            play(AudioSegment.from_mp3(buf))
        else:
            import subprocess
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            try:
                cmd = ["afplay", tmp.name] if self._backend == "afplay" else \
                      ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp.name]
                subprocess.run(cmd, timeout=15, capture_output=True)
            finally:
                os.unlink(tmp.name)


# ═══════════════════════════════════════════════════════════════════
# Frame Grabber
# ═══════════════════════════════════════════════════════════════════

class FrameGrabber:
    """Threaded frame capture with ESP32-CAM + webcam support."""

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
        self._thread = None
        self._cap = None
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
            return frame is not None
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            return False
        if not self._is_stream:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

    def _fetch_snapshot(self):
        try:
            import urllib.request
            resp = urllib.request.urlopen(self.source, timeout=5)
            return cv2.imdecode(
                np.frombuffer(resp.read(), np.uint8), cv2.IMREAD_COLOR)
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
            self._q.put((frame, time.time()))

    def _reconnect(self) -> bool:
        self._reconnect_count += 1
        if self._reconnect_count > 50:
            return False
        time.sleep(min(10.0, 0.5 * (2 ** min(self._reconnect_count, 5))))
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
# Pipeline
# ═══════════════════════════════════════════════════════════════════

class DrishtimargaPipeline:
    """Main local orchestrator: YOLO → Depth → GPT → ElevenLabs."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._running = False

        logger.info("=" * 60)
        logger.info("  DRISHTIMARGA v3 — Local Mode")
        logger.info("  YOLO11x + Depth + GPT-4o-mini + ElevenLabs")
        logger.info("=" * 60)

        # Check API keys
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "")
        if not openai_key:
            logger.error("OPENAI_API_KEY env var not set!")
            raise ValueError("Set OPENAI_API_KEY environment variable")
        if not elevenlabs_key:
            logger.error("ELEVENLABS_API_KEY env var not set!")
            raise ValueError("Set ELEVENLABS_API_KEY environment variable")

        self.grabber = FrameGrabber(
            config.camera.source, config.camera.width, config.camera.height)
        self.detector = ObjectDetector(config.detector)
        self.depth = DepthEstimator(config.depth)
        self.spatial = SpatialEngine(config.spatial, config.threat)
        self.llm = LLMAnalyzer(openai_key)
        self.tts = ElevenLabsTTS(elevenlabs_key)
        self.audio = AudioPlayer()
        self._cycle_times: list = []
        self._last_announcement = ""

    def run(self):
        if not self.grabber.start():
            return

        self.audio.start()
        self._running = True
        logger.info("Pipeline running — press 'q' to quit")

        try:
            while self._running:
                t0 = time.perf_counter()

                frame, ts = self.grabber.get(timeout=1.0)
                if frame is None:
                    continue

                # Detect + track
                detections = self.detector.detect_and_track(frame)

                # Depth
                depth_map = None
                if self.config.depth.enabled:
                    depth_map = self.depth.estimate(frame)

                # Spatial analysis
                assessments = self.spatial.update(
                    detections, depth_map, frame.shape[:2], self.depth)

                # Build detection data for LLM
                det_data = []
                for det, assessment in zip(detections, assessments):
                    dist_m = assessment.distance
                    dist_ft = round(dist_m * 3.28084, 1)
                    det_data.append({
                        "track_id": det.track_id,
                        "class_name": det.class_name,
                        "distance_m": round(dist_m, 2),
                        "distance_ft": dist_ft,
                        "velocity_mps": round(assessment.approach_velocity, 2),
                        "position": assessment.position,
                    })

                # LLM analysis (frame pooling — only every ~2.5s)
                if self.llm.should_analyze(det_data):
                    text, urgency = self.llm.analyze(det_data)
                    if text:
                        logger.info(f"🔊 [{urgency}] {text}")
                        audio_bytes = self.tts.synthesize(text)
                        if audio_bytes:
                            self.audio.play(audio_bytes)
                        self._last_announcement = text

                # Visualization
                if self.config.display.show_video:
                    annotated = self._draw(frame, detections, assessments)
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

    def _draw(self, frame, detections, assessments):
        """Simple visualization."""
        canvas = frame.copy()
        a_map = {a.track_id: a for a in assessments}

        for det in detections:
            a = a_map.get(det.track_id)
            x1, y1, x2, y2 = det.bbox.astype(int)
            dist_ft = round(a.distance * 3.28084) if a else 0
            color = (0, 0, 255) if a and a.distance < 1.5 else (0, 255, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"#{det.track_id} {det.class_name} {dist_ft}ft"
            cv2.putText(canvas, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Last announcement
        if self._last_announcement:
            h = canvas.shape[0]
            cv2.putText(canvas, self._last_announcement[:80], (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        return canvas

    def _shutdown(self):
        self._running = False
        self.grabber.stop()
        self.audio.stop()
        cv2.destroyAllWindows()
        if self._cycle_times:
            avg = sum(self._cycle_times) / len(self._cycle_times)
            logger.info(f"Avg cycle: {avg:.1f}ms (~{1000/avg:.1f} FPS)")
        logger.info("Drishtimarga stopped.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Drishtimarga v3 — AI Navigation for the Visually Impaired")
    p.add_argument("--source", default=0,
                   help="Camera index (0,1) or ESP32-CAM URL")
    p.add_argument("--model", default="yolo11x.pt",
                   help="YOLO model (yolo11x.pt, yolo11l.pt, yolo11n.pt)")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--depth-model", default="metric-outdoor-large")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = AppConfig()

    try:
        cfg.camera.source = int(args.source)
    except (ValueError, TypeError):
        cfg.camera.source = args.source

    cfg.detector.model_name = args.model
    cfg.detector.confidence = args.conf
    cfg.detector.input_size = args.size
    cfg.detector.device = args.device
    cfg.depth.enabled = not args.no_depth
    cfg.depth.model_name = args.depth_model
    cfg.depth.device = args.device
    cfg.display.show_video = not args.no_display

    pipeline = DrishtimargaPipeline(cfg)

    def on_signal(sig, frame):
        pipeline._running = False

    signal.signal(signal.SIGINT, on_signal)
    pipeline.run()


if __name__ == "__main__":
    main()
