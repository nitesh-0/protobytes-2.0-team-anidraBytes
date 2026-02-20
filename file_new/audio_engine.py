"""
Drishtimarga v2 — Audio Feedback Engine
Priority-based TTS with SAPI (Windows) / pyttsx3 fallback.
The announcement memory (announcement_memory.py) handles all filtering
logic — this module only handles queuing and speaking.
"""

import logging
import queue
import sys
import threading
import time
from typing import List, Optional

from config import AudioConfig
from spatial_engine import Priority, ThreatAssessment

logger = logging.getLogger(__name__)

_MAX_MESSAGE_AGE = 10.0  # seconds — drop messages older than this


class AudioEngine:
    """
    Non-blocking audio engine.
    Receives already-filtered messages and speaks them in priority order.
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._urgent_lock = threading.Lock()
        self._urgent_message: Optional[str] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._speaking = False
        self._engine_ready = threading.Event()
        self._total_spoken = 0
        self._total_dropped = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._audio_loop, daemon=True, name="AudioEngine")
        self._thread.start()
        self._engine_ready.wait(timeout=5.0)
        logger.info("Audio engine started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    # ── Public API ──

    def speak_assessments(self, assessments: List[ThreatAssessment]):
        """Queue already-filtered assessments for speaking."""
        for a in assessments:
            if a.priority == Priority.CRITICAL:
                self._send_urgent(a.message)
            else:
                self._enqueue(int(a.priority), a.message)

    def speak_departures(self, messages: List[str]):
        """Queue departure messages."""
        for msg in messages:
            self._enqueue(int(Priority.NORMAL), msg)

    def speak_summary(self, text: str):
        """Queue a scene summary."""
        self._enqueue(int(Priority.LOW), text)

    def speak_now(self, text: str, priority: int = 1):
        """Queue an arbitrary message."""
        self._enqueue(priority, text)

    def announce_startup(self):
        self._enqueue(int(Priority.HIGH),
                      "Drishtimarga is ready. Scanning surroundings.")

    # ── Internal ──

    def _send_urgent(self, message: str):
        with self._urgent_lock:
            self._urgent_message = message
        logger.warning(f"URGENT: {message}")

    def _enqueue(self, priority: int, message: str):
        try:
            self._queue.put_nowait((priority, time.time(), message))
        except queue.Full:
            pass

    def _audio_loop(self):
        """Background TTS loop."""
        _com_init = False
        if sys.platform == "win32":
            try:
                import comtypes
                comtypes.CoInitialize()
                _com_init = True
            except Exception:
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                    _com_init = True
                except Exception:
                    pass

        try:
            engine = self._init_tts()
            self._engine_ready.set()
        except Exception as e:
            logger.error(f"TTS init failed: {e}")
            self._engine_ready.set()
            return

        errors = 0
        try:
            while self._running:
                try:
                    # Urgent first
                    with self._urgent_lock:
                        if self._urgent_message:
                            msg = self._urgent_message
                            self._urgent_message = None
                            self._speak(engine, msg)
                            errors = 0
                            continue

                    # Normal queue
                    try:
                        pri, ts, msg = self._queue.get(timeout=0.15)
                        age = time.time() - ts
                        if age < _MAX_MESSAGE_AGE:
                            self._speak(engine, msg)
                            errors = 0
                        else:
                            self._total_dropped += 1
                        self._queue.task_done()
                    except queue.Empty:
                        pass

                except Exception as e:
                    errors += 1
                    logger.error(f"Audio error: {e}")
                    if errors >= 3:
                        try:
                            engine = self._init_tts()
                            errors = 0
                        except Exception:
                            pass
                    time.sleep(0.2)
        finally:
            if _com_init:
                try:
                    import comtypes
                    comtypes.CoUninitialize()
                except Exception:
                    try:
                        import pythoncom
                        pythoncom.CoUninitialize()
                    except Exception:
                        pass

    def _init_tts(self):
        """Try SAPI (Windows) then pyttsx3 fallback."""
        # SAPI via comtypes
        try:
            import comtypes
            from comtypes.client import CreateObject
            voice = CreateObject("SAPI.SpVoice")
            voice.Rate = max(-10, min(10, (self.config.rate - 150) // 20))
            voice.Volume = int(self.config.volume * 100)
            logger.info("TTS: Windows SAPI (comtypes)")
            return ("sapi", voice)
        except Exception:
            pass

        # pyttsx3
        import pyttsx3
        eng = pyttsx3.init()
        eng.setProperty("rate", self.config.rate)
        eng.setProperty("volume", self.config.volume)
        voices = eng.getProperty("voices")
        if voices and len(voices) > 1:
            for v in voices:
                if "english" in v.name.lower():
                    eng.setProperty("voice", v.id)
                    break
        logger.info("TTS: pyttsx3")
        return ("pyttsx3", eng)

    def _speak(self, engine_tuple, text: str):
        backend, engine = engine_tuple
        try:
            self._speaking = True
            logger.info(f"TTS> {text}")
            if backend == "sapi":
                engine.Speak(text, 0)
            else:
                engine.say(text)
                engine.runAndWait()
                try:
                    engine._inLoop = False
                except AttributeError:
                    pass
            self._speaking = False
            self._total_spoken += 1
        except Exception as e:
            self._speaking = False
            logger.error(f"TTS error: {e}")
            raise

    @property
    def stats(self) -> dict:
        return {
            "total_spoken": self._total_spoken,
            "total_dropped": self._total_dropped,
            "queue_size": self._queue.qsize(),
            "is_speaking": self._speaking,
        }
