"""
Drishtimarga — Audio Feedback Engine
Priority-based TTS with cooldowns, urgent interrupts, and scene summaries.
"""

import logging
import queue
import sys
import threading
import time
from typing import Dict, Optional

from config import AudioConfig
from spatial_engine import Priority, ThreatAssessment

logger = logging.getLogger(__name__)

# Maximum age (seconds) a queued message can be before it's discarded.
# Must be long enough to survive while TTS is busy speaking another message.
_MAX_MESSAGE_AGE = 12.0


class AudioEngine:
    """
    Non-blocking audio engine with:
    - Priority queue (critical > high > normal > low)
    - Per-object cooldown to prevent repetition
    - Urgent interrupt capability for danger alerts
    - Periodic scene summaries
    - Navigation guidance (path-clear, dodge advice, departure notices)
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        # {track_id: (last_time, last_distance)}
        self._cooldowns: Dict[int, tuple] = {}
        self._urgent_lock = threading.Lock()
        self._urgent_message: Optional[str] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_summary_time = time.time()
        self._speaking = False
        self._engine_ready = threading.Event()

        # Stats
        self._total_announcements = 0
        self._total_suppressed = 0
        self._total_dropped = 0
        self._total_nav_guidance = 0

    def start(self):
        """Start the audio engine background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._audio_loop, daemon=True, name="AudioEngine")
        self._thread.start()
        self._engine_ready.wait(timeout=5.0)
        logger.info("Audio engine started")

    def stop(self):
        """Stop the audio engine."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("Audio engine stopped")

    def _init_tts_engine(self):
        """
        Create a TTS engine.  Tries (in order):
        1. Windows SAPI via comtypes  (reliable in threads)
        2. pyttsx3 fallback
        """
        # ── Attempt 1: Direct SAPI via comtypes (works reliably in threads) ──
        try:
            import comtypes
            from comtypes.client import CreateObject
            sp_voice = CreateObject("SAPI.SpVoice")
            # Rate: SAPI uses -10..+10; map from WPM roughly
            # 190 WPM ≈ SAPI rate +2
            sapi_rate = max(-10, min(10, (self.config.rate - 150) // 20))
            sp_voice.Rate = sapi_rate
            sp_voice.Volume = int(self.config.volume * 100)
            logger.info("Using Windows SAPI (comtypes) for TTS")
            return ("sapi", sp_voice)
        except Exception as e:
            logger.warning(f"SAPI via comtypes failed: {e}")

        # ── Attempt 2: pyttsx3 fallback ──
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.config.rate)
            engine.setProperty("volume", self.config.volume)
            voices = engine.getProperty("voices")
            if voices and len(voices) > 1:
                for v in voices:
                    if "english" in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
            logger.info("Using pyttsx3 for TTS")
            return ("pyttsx3", engine)
        except Exception as e:
            logger.warning(f"pyttsx3 failed: {e}")

        raise RuntimeError("No working TTS backend found")

    def _audio_loop(self):
        """Main audio loop running in background thread."""
        # ── COM must be STA-initialised for Windows SAPI in a worker thread ──
        _com_initialized = False
        if sys.platform == "win32":
            try:
                import comtypes
                comtypes.CoInitialize()          # STA — required by SAPI
                _com_initialized = True
                logger.debug("COM initialized (comtypes) for audio thread")
            except Exception:
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                    _com_initialized = True
                    logger.debug(
                        "COM initialized (pythoncom) for audio thread")
                except Exception as exc:
                    logger.warning(
                        f"Could not init COM for audio thread: {exc}")

        try:
            engine = self._init_tts_engine()
            self._engine_ready.set()
            logger.info(f"TTS initialized (rate={self.config.rate})")
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            self._engine_ready.set()
            return

        consecutive_errors = 0

        try:
            while self._running:
                try:
                    # Check for urgent message first (interrupts everything)
                    with self._urgent_lock:
                        if self._urgent_message:
                            msg = self._urgent_message
                            self._urgent_message = None
                            self._speak(engine, msg)
                            consecutive_errors = 0
                            continue

                    # Process normal queue
                    try:
                        priority, timestamp, message = self._queue.get(
                            timeout=0.2)
                        age = time.time() - timestamp

                        if age < _MAX_MESSAGE_AGE:
                            logger.debug(
                                f"Speaking (pri={priority}, age={age:.1f}s): "
                                f"{message}")
                            self._speak(engine, message)
                            consecutive_errors = 0
                        else:
                            self._total_dropped += 1
                            logger.debug(
                                f"Dropped stale message ({age:.1f}s old): "
                                f"{message}")

                        self._queue.task_done()
                    except queue.Empty:
                        pass

                except Exception as e:
                    logger.error(f"Audio loop error: {e}")
                    consecutive_errors += 1

                    # If TTS keeps failing, try to reinitialize the engine
                    if consecutive_errors >= 3:
                        logger.warning(
                            "Multiple TTS failures — reinitializing engine")
                        try:
                            backend, eng = engine
                            if backend == "pyttsx3":
                                eng.stop()
                        except Exception:
                            pass
                        try:
                            engine = self._init_tts_engine()
                            consecutive_errors = 0
                            logger.info("TTS engine reinitialized")
                        except Exception as e2:
                            logger.error(f"TTS reinit failed: {e2}")

                    time.sleep(0.2)
        finally:
            # Release COM apartment when the loop exits
            if _com_initialized:
                try:
                    import comtypes
                    comtypes.CoUninitialize()
                except Exception:
                    try:
                        import pythoncom
                        pythoncom.CoUninitialize()
                    except Exception:
                        pass

    def _speak(self, engine_tuple, text: str):
        """Speak a message using the active TTS backend."""
        backend, engine = engine_tuple
        try:
            self._speaking = True
            logger.info(f"TTS> {text}")

            if backend == "sapi":
                # SVSFlagsAsync = 1, SVSFPurgeBeforeSpeak = 2
                # We speak synchronously (flags=0) so we block until done
                engine.Speak(text, 0)
            else:
                # pyttsx3 fallback
                engine.say(text)
                engine.runAndWait()
                try:
                    engine._inLoop = False
                except AttributeError:
                    pass

            self._speaking = False
            self._total_announcements += 1
        except Exception as e:
            self._speaking = False
            logger.error(f"TTS speak error: {e}")
            raise  # re-raise so _audio_loop can track consecutive failures

    def process_assessments(self, assessments: list):
        """
        Filter and queue announcements from threat assessments.
        Applies cooldown logic and priority filtering.
        """
        now = time.time()
        announced_this_cycle = 0

        for assessment in assessments:
            if announced_this_cycle >= self.config.max_announcements_per_cycle:
                break

            if assessment.priority == Priority.SKIP:
                continue

            # Check cooldown (critical alerts bypass cooldown)
            if assessment.priority > Priority.CRITICAL:
                if not self._should_announce(assessment, now):
                    self._total_suppressed += 1
                    continue

            # Route by priority
            if assessment.priority == Priority.CRITICAL:
                self._send_urgent(assessment.message)
            else:
                self._enqueue(assessment.priority, assessment.message)
                logger.debug(
                    f"Queued [{assessment.priority.name}]: {assessment.message}")

            # Update cooldown state
            self._cooldowns[assessment.track_id] = (now, assessment.distance)
            announced_this_cycle += 1

    def _should_announce(self, assessment: ThreatAssessment, now: float) -> bool:
        """Check if we should announce this object (cooldown logic)."""
        tid = assessment.track_id

        if tid not in self._cooldowns:
            return True  # never announced before

        last_time, last_distance = self._cooldowns[tid]

        # Cooldown expired?
        if now - last_time > self.config.cooldown:
            return True

        # Distance changed significantly?
        if abs(assessment.distance - last_distance) > self.config.distance_change_threshold:
            return True

        # New object (just appeared)?
        if assessment.is_new:
            return True

        return False

    def _send_urgent(self, message: str):
        """Send an urgent message that interrupts current speech."""
        with self._urgent_lock:
            self._urgent_message = message
        logger.warning(f"URGENT: {message}")

    def _enqueue(self, priority: Priority, message: str):
        """Add a message to the priority queue."""
        try:
            self._queue.put_nowait((int(priority), time.time(), message))
        except queue.Full:
            pass  # drop if queue is full

    def maybe_scene_summary(self, summary_func):
        """
        Call periodically to trigger scene summaries.
        summary_func should return a string.
        """
        now = time.time()
        if now - self._last_summary_time >= self.config.scene_summary_interval:
            self._last_summary_time = now
            summary = summary_func()
            if summary:
                self._enqueue(Priority.NORMAL, summary)

    def process_navigation_guidance(self, guidance: Optional[str]):
        """
        Queue a navigation guidance message (path-clear, dodge, departure).
        These are lower priority than threat alerts but important for usability.
        """
        if guidance is None:
            return
        self._total_nav_guidance += 1
        # Navigation guidance goes at NORMAL priority so it doesn't
        # stomp on urgent/high threat alerts but is still spoken promptly.
        self._enqueue(Priority.NORMAL, guidance)

    def announce_startup(self):
        """Announce that the system is ready."""
        self._enqueue(
            Priority.HIGH, "Drishtimarga is ready. Scanning surroundings.")

    @property
    def stats(self) -> dict:
        return {
            "total_announcements": self._total_announcements,
            "total_suppressed": self._total_suppressed,
            "total_dropped": self._total_dropped,
            "total_nav_guidance": self._total_nav_guidance,
            "queue_size": self._queue.qsize(),
            "is_speaking": self._speaking,
        }
