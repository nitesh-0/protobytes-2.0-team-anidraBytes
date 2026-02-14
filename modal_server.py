"""
Drishtimarga v3 — Modal GPU Server with LLM + ElevenLabs TTS

Architecture:
    Frame → YOLO11x detection + ByteTrack tracking
         → Depth Anything V2 Metric Outdoor Large (real meters)
         → Frame pooling (accumulate detections)
         → Every ~2-3s: GPT-4o-mini analyzes scene → short Nepali text
         → ElevenLabs TTS → audio bytes returned to client

The LLM decides WHAT to say (and whether to say anything at all).
No more repetitive announcements — the model understands context.

Deploy:
    modal secret create drishtimarga-secrets \\
        OPENAI_API_KEY="sk-proj-..." \\
        ELEVENLABS_API_KEY="sk_..."

    modal deploy modal_server.py

Test locally:
    modal serve modal_server.py
"""

import base64
import io
import json
import logging
import time
from typing import Dict, List, Optional

import modal

logger = logging.getLogger("drishtimarga-v3")

# ── Modal App & Image ──

app = modal.App("drishtimarga-v3")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1", "ffmpeg")
    .pip_install(
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
        "Pillow>=10.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.3.0",
        "transformers>=4.36.0",
        "fastapi[standard]",
        "openai>=1.12.0",
        "requests>=2.28.0",
    )
)

model_volume = modal.Volume.from_name("drishtimarga-models", create_if_missing=True)

# ── Model config ──
YOLO_MODEL = "yolo11x.pt"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

# ── ElevenLabs config ──
ELEVENLABS_VOICE_ID = "pFZP5JQG7iQjIQuC4Bku"  # "Lily" — clear female, good for multilingual
ELEVENLABS_MODEL = "eleven_multilingual_v2"

# ── Narration config ──
MIN_AUDIO_INTERVAL = 6.0    # Minimum wait between narrations
MAX_AUDIO_INTERVAL = 12.0   # Force narration if no change for this long
STABILITY_THRESHOLD = 15    # Number of frames to check for scene stability

# ── GPT System Prompt ──
SYSTEM_PROMPT = """You are 'Drishti Spatial Assistant', a highly precise and natural assistant for a blind person.
Your task is to describe the surroundings based ONLY on the provided detection data.

Movement Tags Guide:
- "नजिकिँदै" (coming/approaching)
- "टाढिँदै" (going/receding)
- "स्थिर" (static/still)

Rules:
1. Data Strictness: ONLY describe objects present in the JSON input. Do NOT assume a "person" exists if the data only contains "chair" or "laptop".
2. Natural Phrasing: Use the style: "[Object] is approximately [Distance] feet [Position] of you and is [Movement]."
   Example: "एक कुर्सी तपाईको बायाँ तिर १० फिटको दुरीमा स्थिर अवस्थामा छ।"
3. Pure Nepali: Speak only in Nepali. Convert all numbers to pure Nepali words.
4. Brevity: One short, natural sentence. No prefixes like "I see" or "Detections are".
5. Movement: Explicitly mention if something is coming towards or going away from the user.
"""


@app.cls(
    image=gpu_image,
    gpu="H100",
    timeout=300,
    scaledown_window=120,
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("drishtimarga-secrets")],
)
@modal.concurrent(max_inputs=4)
class DrishtimargaInference:
    """
    Stateful Modal class: loads models once, serves inference.
    Maintains per-session state for tracking, frame pooling, and LLM context.
    """

    @modal.enter()
    def load_models(self):
        """Called once on container start — load YOLO + Depth + init OpenAI."""
        import numpy as np
        import os
        import torch
        from ultralytics import YOLO
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        from openai import OpenAI

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")

        # ── YOLO11x ──
        logger.info(f"Loading YOLO: {YOLO_MODEL}")
        self.yolo = YOLO(YOLO_MODEL)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.yolo.predict(dummy, verbose=False, device=self.device)
        logger.info("YOLO ready")

        # ── Depth Anything V2 Metric Outdoor Large ──
        logger.info(f"Loading depth: {DEPTH_MODEL_ID}")
        self.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
        self.depth_model.to(self.device).eval()
        if self.device == "cuda":
            self.depth_model = self.depth_model.half()
            logger.info("Depth using FP16")
        self._prev_depth_map = None
        self._temporal_alpha = 0.6
        self._run_depth(dummy)
        logger.info("Depth ready")

        # ── OpenAI client ──
        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        logger.info("OpenAI client ready")

        # ── ElevenLabs API key ──
        self.elevenlabs_key = os.environ["ELEVENLABS_API_KEY"]
        logger.info("ElevenLabs key loaded")

        # ── Per-session state ──
        # Track history for LLM context
        self._track_counter = 0
        self._last_llm_time = 0.0
        self._last_llm_detections = []  # what we last told the LLM
        self._last_llm_response = ""
        self._scene_history = []  # rolling window of recent scenes for LLM context
        self._consecutive_empty = 0  # how many times LLM said nothing

        # Per-object EMA smoothed distances
        self._smoothed_distances: Dict[int, float] = {}
        self._distance_alpha = 0.3

        # Nepali number mapping
        self._nepali_num_map = {
            '0': 'शुन्य', '1': 'एक', '2': 'दुई', '3': 'तीन', '4': 'चार', 
            '5': 'पाँच', '6': 'छ', '7': 'सात', '8': 'आठ', '9': 'नौ', '10': 'दश',
            '11': 'एघार', '12': 'बाह्र', '13': 'तेह्र', '14': 'चौध', '15': 'पन्ध्र',
            '16': 'सोह्र', '17': 'सत्र', '18': 'अठार', '19': 'उन्नाइस', '20': 'बीस',
            '21': 'एककाइस', '22': 'बाइस', '23': 'तेइस', '24': 'चौबिस', '25': 'पच्चिस',
            '26': 'छब्बीस', '27': 'सत्ताइस', '28': 'अट्ठाइस', '29': 'उनन्तीस', '30': 'तीस',
            '31': 'एकतीस', '32': 'बत्तीस', '33': 'तेत्तीस', '34': 'चौँतीस', '35': 'पैँतीस',
            '36': 'छत्तिस', '37': 'सर्र्तीस', '38': 'अठतीस', '39': 'उनन्चालीस', '40': 'चालीस',
            '41': 'एकचालीस', '42': 'बयालीस', '43': 'त्रिचालीस', '44': 'चौवालीस', '45': 'पैँतालीस',
            '46': 'छयालीस', '47': 'सतचालीस', '48': 'अठचालीस', '49': 'उनन्पचास', '50': 'पचास',
            '60': 'साठी', '70': 'सत्तरी', '80': 'अस्ती', '90': 'नब्बे', '100': 'सय'
        }

        # Per-object velocity tracking: {track_id: [(distance, timestamp), ...]}
        self._velocity_history: Dict[int, list] = {}

        # ── Refactored Narration State ──
        self._last_announcement_time = 0.0
        self._frame_buffer = []  # Buffer for 15-frame consensus
        self._last_narrated_scene_fingerprint = ""

    # ════════════════════════════════════════════════════════════
    # Depth inference
    # ════════════════════════════════════════════════════════════

    def _run_depth(self, frame_bgr):
        """Run metric depth inference. Returns depth map in meters."""
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        inputs = self.depth_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            depth = outputs.predicted_depth

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1).float(),
            size=(frame_bgr.shape[0], frame_bgr.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        depth = np.clip(depth, 0.1, 80.0)

        # Temporal EMA smoothing
        if (self._prev_depth_map is not None
                and self._prev_depth_map.shape == depth.shape):
            alpha = self._temporal_alpha
            depth = alpha * depth + (1.0 - alpha) * self._prev_depth_map
        self._prev_depth_map = depth.copy()

        return depth

    # ════════════════════════════════════════════════════════════
    # Distance sampling from depth map
    # ════════════════════════════════════════════════════════════

    def _get_distance(self, depth_map, bbox, cx, cy, bbox_h, class_name):
        """Sample metric distance (meters) from depth map at a bounding box."""
        import numpy as np

        KNOWN_HEIGHTS = {
            "person": 1.7, "car": 1.5, "bus": 2.8, "truck": 3.0,
            "motorcycle": 1.1, "bicycle": 1.0, "dog": 0.5, "cat": 0.3,
            "chair": 0.8, "bottle": 0.25, "cup": 0.12, "laptop": 0.25,
            "tv": 0.5, "cell phone": 0.14, "backpack": 0.5,
        }

        if depth_map is not None:
            h, w = depth_map.shape[:2]
            x1b, y1b, x2b, y2b = bbox
            bw, bh = x2b - x1b, y2b - y1b
            margin_x = bw * 0.25
            margin_y = bh * 0.25
            ix1 = int(np.clip(x1b + margin_x, 0, w - 1))
            iy1 = int(np.clip(y1b + margin_y, 0, h - 1))
            ix2 = int(np.clip(x2b - margin_x, 1, w))
            iy2 = int(np.clip(y2b - margin_y, 1, h))
            patch = depth_map[iy1:iy2, ix1:ix2]

            if patch.size > 0:
                flat = patch.flatten()
                if len(flat) > 10:
                    low = np.percentile(flat, 15)
                    high = np.percentile(flat, 85)
                    trimmed = flat[(flat >= low) & (flat <= high)]
                    if len(trimmed) > 0:
                        distance = float(np.median(trimmed))
                    else:
                        distance = float(np.median(flat))
                else:
                    distance = float(np.median(flat))
                return float(np.clip(distance, 0.2, 80.0))

        # Fallback: bbox-height heuristic
        known_h = KNOWN_HEIGHTS.get(class_name, 1.0)
        if bbox_h > 10:
            distance = (500.0 * known_h) / bbox_h
        else:
            distance = 20.0
        return float(np.clip(distance, 0.3, 50.0))

    # ════════════════════════════════════════════════════════════
    # Velocity computation
    # ════════════════════════════════════════════════════════════

    def _compute_velocity(self, track_id: int, distance: float, now: float) -> float:
        """Compute approach velocity (m/s, positive = getting closer) using linear regression."""
        if track_id not in self._velocity_history:
            self._velocity_history[track_id] = []

        hist = self._velocity_history[track_id]
        hist.append((distance, now))
        # Keep last 15 samples
        if len(hist) > 15:
            self._velocity_history[track_id] = hist[-15:]
            hist = self._velocity_history[track_id]

        if len(hist) < 3:
            return 0.0

        distances = [h[0] for h in hist]
        times = [h[1] for h in hist]
        n = len(distances)
        st = sum(times)
        sd = sum(distances)
        std_ = sum(t * d for t, d in zip(times, distances))
        stt = sum(t * t for t in times)
        denom = n * stt - st * st
        if abs(denom) < 1e-10:
            return 0.0
        slope = (n * std_ - st * sd) / denom
        return max(0.0, -slope)  # positive = approaching

    # ════════════════════════════════════════════════════════════
    # Full frame processing pipeline
    # ════════════════════════════════════════════════════════════

    def _process_frame(self, frame_bgr):
        """
        Full pipeline on one frame:
        1. YOLO detect + track
        2. Depth estimation
        3. Build detection list with distances + velocities
        4. Frame pooling: if enough time passed, call GPT + ElevenLabs
        """
        import numpy as np
        import time as _time

        t0 = _time.perf_counter()
        now = _time.time()
        h_frame, w_frame = frame_bgr.shape[:2]

        # ── 1. YOLO detect + track ──
        self._track_counter += 1
        results = self.yolo.track(
            frame_bgr,
            persist=True,
            conf=0.40,
            iou=0.50,
            imgsz=640,
            tracker="bytetrack.yaml",
            half=(self.device == "cuda"),
            device=self.device,
            verbose=False,
        )
        det_ms = (_time.perf_counter() - t0) * 1000

        # ── 2. Depth ──
        t1 = _time.perf_counter()
        depth_map = self._run_depth(frame_bgr)
        depth_ms = (_time.perf_counter() - t1) * 1000

        # ── 3. Build detections with distances + velocities ──
        detections = []
        active_track_ids = set()

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                tid = int(box.id[0].item())
                cid = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                conf = float(box.conf[0].item())
                class_name = results[0].names[cid]
                active_track_ids.add(tid)

                # Raw metric distance
                raw_distance = self._get_distance(
                    depth_map, [x1, y1, x2, y2], cx, cy, bbox_h, class_name)

                # EMA smooth per-object distance
                if tid in self._smoothed_distances:
                    prev = self._smoothed_distances[tid]
                    distance = self._distance_alpha * raw_distance + (1 - self._distance_alpha) * prev
                else:
                    distance = raw_distance
                self._smoothed_distances[tid] = distance

                # Velocity
                velocity = self._compute_velocity(tid, distance, now)

                # Position
                rel_x = cx / w_frame if w_frame > 0 else 0.5
                if rel_x < 0.33:
                    position = "left"
                elif rel_x > 0.67:
                    position = "right"
                else:
                    position = "center"

                # Distance in feet for the LLM (round to integer)
                distance_ft = int(round(distance * 3.28084))

                detections.append({
                    "track_id": tid,
                    "class_name": class_name,
                    "class_id": cid,
                    "confidence": round(conf, 3),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "center": [round(cx, 1), round(cy, 1)],
                    "bbox_width": round(bbox_w, 1),
                    "bbox_height": round(bbox_h, 1),
                    "distance_m": round(distance, 2),
                    "distance_ft": distance_ft,
                    "velocity_mps": round(velocity, 2),
                    "position": position,
                    "rel_x": round(rel_x, 3),
                })

        # Clean up stale velocity history and smoothed distances
        stale = [tid for tid in self._velocity_history if tid not in active_track_ids]
        for tid in stale:
            if now - self._velocity_history[tid][-1][1] > 5.0:
                del self._velocity_history[tid]
                self._smoothed_distances.pop(tid, None)

        # ── 4. Refactored Narration Logic (15-frame Consensus) ──
        audio_b64 = None
        llm_text = ""
        urgency = "none"
        
        # Add current detections to buffer
        self._frame_buffer.append(detections)
        if len(self._frame_buffer) > 15:
            self._frame_buffer.pop(0)

        time_since_last = now - self._last_announcement_time
        
        # ── Decide if we should try to narrate ──
        # We process the buffer when it hits exactly 15 frames or at heartbeat interval
        should_evaluate = len(self._frame_buffer) >= 15 or time_since_last >= MAX_AUDIO_INTERVAL

        if should_evaluate and detections:
            # 1. Consensus Voting: only keep objects present in >= 50% of frames
            # This eliminates 'flickering' hallucinations
            from collections import Counter
            all_ids = [d["track_id"] for frame in self._frame_buffer for d in frame]
            counts = Counter(all_ids)
            stable_ids = {tid for tid, count in counts.items() if count >= 8}
            
            stable_detections = []
            for d in detections:
                if d["track_id"] in stable_ids:
                    # Enhance with movement tag
                    d["movement"] = self._get_movement_tag(d)
                    stable_detections.append(d)
            
            if stable_detections:
                # 2. Build Scene Fingerprint
                # Fingerprint includes Class, Position, and rounded Distance to detect real changes
                scene_fingerprint = "|".join(
                    f"{d['class_name']}-{d['position']}-{round(d['distance_ft']/5)*5}"
                    for d in sorted(stable_detections, key=lambda x: x['distance_ft'])[:3]
                )

                is_scene_changed = scene_fingerprint != self._last_narrated_scene_fingerprint
                
                # 3. Trigger Logic
                should_call_llm = False
                if is_scene_changed and time_since_last >= MIN_AUDIO_INTERVAL:
                    should_call_llm = True
                elif time_since_last >= MAX_AUDIO_INTERVAL:
                    should_call_llm = True
                
                # Critical immediate trigger
                if self._has_critical_threat(stable_detections) and time_since_last >= 3.0:
                    should_call_llm = True

                if should_call_llm:
                    t2 = _time.perf_counter()
                    llm_text, urgency = self._call_llm(stable_detections, now)
                    llm_ms = (_time.perf_counter() - t2) * 1000

                    if llm_text:
                        llm_text = self._to_nepali_words(llm_text)
                        t3 = _time.perf_counter()
                        audio_b64 = self._call_elevenlabs(llm_text)
                        tts_ms = (_time.perf_counter() - t3) * 1000
                        logger.info(f"LLM: '{llm_text}' | urgency={urgency} | consensus_size={len(stable_detections)}")
                        
                        self._last_announcement_time = now
                        self._last_narrated_scene_fingerprint = scene_fingerprint
                        self._frame_buffer = [] # Clear buffer after successful narration

        total_ms = (_time.perf_counter() - t0) * 1000

        return {
            "detections": detections,
            "frame_shape": [h_frame, w_frame],
            "detection_ms": round(det_ms, 1),
            "depth_ms": round(depth_ms, 1),
            "total_ms": round(total_ms, 1),
            "depth_metric": True,
            # LLM + TTS results (may be empty if not this cycle)
            "llm_text": llm_text,
            "urgency": urgency,
            "audio_b64": audio_b64,  # base64 MP3 or None
        }

    def _get_movement_tag(self, d: dict) -> str:
        """Determine movement type (approaching/receding/static) based on velocity."""
        vel = d.get("velocity_mps", 0)
        dist = d.get("distance_ft", 0)
        
        # Approaching is positive velocity in my compute_velocity
        if vel > 0.4:
            return "नजिकिँदै"
        elif vel < -0.4:
            return "टाढिँदै"
        else:
            return "स्थिर"

    def _has_critical_threat(self, detections: list) -> bool:
        """Check if any detection is critically close + approaching fast."""
        for d in detections:
            if d["distance_ft"] < 5.0 and d["velocity_mps"] > 0.5:
                return True
            if d["distance_ft"] < 3.0:
                return True
        return False

    # ════════════════════════════════════════════════════════════
    # Nepali Number Utility
    # ════════════════════════════════════════════════════════════

    def _to_nepali_words(self, text: str) -> str:
        """Replace all ASCII digits in text with Nepali words."""
        import re
        
        # Handle decimal numbers first (e.g. 24.5 -> 24)
        text = re.sub(r'(\d+)\.\d+', r'\1', text)
        
        # Find all multi-digit numbers and replace longest matches first
        nums = sorted(re.findall(r'\d+', text), key=len, reverse=True)
        for num in nums:
            if num in self._nepali_num_map:
                text = text.replace(num, self._nepali_num_map[num])
            else:
                # Individual digits as fallback
                digits = "".join(self._nepali_num_map.get(d, d) for d in num)
                text = text.replace(num, digits)
        return text

    # ════════════════════════════════════════════════════════════
    # GPT-4o-mini LLM call
    # ════════════════════════════════════════════════════════════

    def _call_llm(self, detections: list, now: float) -> tuple:
        """
        Call GPT-4o-mini to analyze the scene and generate a short Nepali announcement.
        Returns (text, urgency) — text may be empty string if nothing to say.
        """
        try:
            # Build compact scene description for the LLM
            scene_data = self._build_scene_for_llm(detections)

            # Include previous context so LLM knows what was already said
            context = ""
            if self._scene_history:
                last = self._scene_history[-1]
                context = f"\nPrevious scene ({last['ago']:.1f}s ago): {last['summary']}"
                if self._last_llm_response:
                    context += f"\nLast announcement: \"{self._last_llm_response}\""
                if self._consecutive_empty > 0:
                    context += f"\n(No announcement for last {self._consecutive_empty} cycles)"

            user_msg = f"""Current scene:
{scene_data}
{context}

Respond with JSON only: {{"speak": "...", "urgency": "none|low|medium|high|critical"}}"""

            response = self.openai_client.chat.completions.create(
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

            self._scene_history.append({
                "time": now,
                "ago": 0.0,
                "summary": summary,
            })
            # Keep last 5 scenes
            if len(self._scene_history) > 5:
                self._scene_history = self._scene_history[-5:]
            # Update ago times
            for s in self._scene_history:
                s["ago"] = now - s["time"]

            if text:
                self._last_llm_response = text
                self._consecutive_empty = 0
            else:
                self._consecutive_empty += 1

            self._last_llm_detections = detections
            return text, urgency

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "", "none"

    def _build_scene_for_llm(self, detections: list) -> str:
        """Build a compact, structured scene description for GPT."""
        if not detections:
            return "No objects detected. Scene is clear."

        lines = []
        # Sort by distance (closest first)
        sorted_dets = sorted(detections, key=lambda d: d["distance_ft"])

        for d in sorted_dets[:8]:  # max 8 objects for token efficiency
            parts = [
                f"{d['class_name']}",
                f"#{d['track_id']}",
                f"{d['distance_ft']}ft",
                f"{d['position']}",
            ]
            if d["velocity_mps"] > 0.3:
                parts.append(f"approaching@{d['velocity_mps']:.1f}m/s")
                # Time to collision
                if d["velocity_mps"] > 0.1:
                    ttc = d["distance_m"] / d["velocity_mps"]
                    if ttc < 10:
                        parts.append(f"TTC={ttc:.1f}s")
            lines.append(" | ".join(parts))

        return "\n".join(lines)

    # ════════════════════════════════════════════════════════════
    # ElevenLabs TTS
    # ════════════════════════════════════════════════════════════

    def _call_elevenlabs(self, text: str) -> Optional[str]:
        """Convert Nepali text to speech via ElevenLabs API. Returns base64 MP3."""
        import requests

        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
            headers = {
                "xi-api-key": self.elevenlabs_key,
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
                audio_bytes = resp.content
                return base64.b64encode(audio_bytes).decode("ascii")
            else:
                logger.error(f"ElevenLabs error {resp.status_code}: {resp.text[:200]}")
                return None

        except Exception as e:
            logger.error(f"ElevenLabs error: {e}")
            return None

    # ════════════════════════════════════════════════════════════
    # FastAPI endpoints
    # ════════════════════════════════════════════════════════════

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def infer(self, request: dict):
        """
        Accept a base64-encoded JPEG frame, run full pipeline.

        Request:
            {"frame_b64": "<base64 JPEG>"}

        Response:
            {
                "detections": [...],
                "frame_shape": [H, W],
                "detection_ms": ...,
                "depth_ms": ...,
                "total_ms": ...,
                "depth_metric": true,
                "llm_text": "नेपाली text or empty",
                "urgency": "none|low|medium|high|critical",
                "audio_b64": "<base64 MP3 or null>"
            }
        """
        import cv2
        import numpy as np

        frame_b64 = request.get("frame_b64", "")
        if not frame_b64:
            return {"error": "No frame_b64 in request body"}

        jpg_bytes = base64.b64decode(frame_b64)
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Failed to decode JPEG frame"}

        return self._process_frame(frame)

    @modal.fastapi_endpoint(method="GET", docs=True)
    async def health(self):
        """Health check."""
        return {
            "status": "ok",
            "model": "drishtimarga-v3",
            "gpu": self.device,
            "yolo": YOLO_MODEL,
            "depth": DEPTH_MODEL_ID,
            "llm": "gpt-4o-mini",
            "tts": "elevenlabs",
        }
