import json
import logging
import requests
import threading
import time
from typing import List, Optional
from config import HfConfig, MODAL_NARRATE_URL

logger = logging.getLogger("drishtimarga-narrator")

class MistralNarrator:
    """
    Modular narrator that uses our own Modal-hosted Mistral-7B.
    """
    
    def __init__(self, config: HfConfig):
        self.config = config
        self.api_url = MODAL_NARRATE_URL
        self._last_narration_time = 0
        self._lock = threading.Lock()
        self._is_thinking = False

    def should_narrate(self) -> bool:
        """Check if enough time has passed to warrant a new narration."""
        return (time.time() - self._last_narration_time) >= self.config.narrative_cooldown

    def narrate_async(self, detections: List[dict], callback):
        """
        Sends detections to Modal in a background thread.
        """
        if self._is_thinking:
            return # Skip if still processing previous request

        thread = threading.Thread(
            target=self._run_inference,
            args=(detections, callback),
            daemon=True
        )
        thread.start()

    def _run_inference(self, detections: List[dict], callback):
        with self._lock:
            self._is_thinking = True
        
        try:
            scene_desc = self._format_scene(detections)
            logger.info(f"LLM INPUT DATA: {scene_desc}")
            
            prompt = (
                f"<s>[INST] You are a highly concise assistant for a blind person. "
                f"Describe the surroundings naturally in one short sentence based on these detections: {scene_desc}. "
                f"Be brief, prioritize closest items. Do not use phrases like 'I see' or 'The scene contains'. [/INST]"
            )

            # Modal doesn't need Auth header if it's public (like yours)
            payload = {"prompt": prompt}

            response = requests.post(self.api_url, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "").strip()
                if text:
                    logger.info(f"LLM NARRATION: {text}")
                    self._last_narration_time = time.time()
                    callback(text)
            else:
                logger.error(f"Modal Narration Error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Narration error: {e}")
        finally:
            with self._lock:
                self._is_thinking = False

    def _format_scene(self, detections: List[dict]) -> str:
        """Converts raw detection dicts into a compact string for the prompt."""
        parts = []
        for d in detections:
            # We use feet because we updated the client to feet earlier
            dist_ft = round(d['distance'] * 3.28084, 1)
            parts.append(f"{d['class_name']} at {dist_ft} feet on the {d['position']}")
        return "; ".join(parts)
