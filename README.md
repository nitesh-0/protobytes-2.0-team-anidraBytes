# Drishtimarga v3

## What Changed from v2 (and Why)

### 1. No More Annoying Repetitive Audio
**v2 problem:** Rule-based state machine tried to suppress repetition but still spoke too often — every object got announced with fixed cooldowns regardless of context.

**v3 solution:** GPT-4o-mini analyzes the scene and decides what to say (and whether to say anything at all). The LLM understands context: it won't repeat "person ahead, 6 feet" if nothing changed. It only speaks when there's a meaningful update.

| | v2 | v3 |
|---|---|---|
| Decision engine | Rule-based state machine | **GPT-4o-mini** (understands context) |
| Language | English | **Nepali** (नेपाली) |
| TTS | pyttsx3 / SAPI (robotic) | **ElevenLabs** (natural voice) |
| Frequency | Every 2-7s per object | **Only when meaningful** (~2-5s intervals) |

### 2. Architecture: Frame Pooling + LLM
```
Every frame:     Camera → YOLO11x → Depth Anything V2 Large → detections with distances
Every ~2.5s:     Pooled detections → GPT-4o-mini → short Nepali text → ElevenLabs TTS → audio
Critical threat: Bypass timer, announce immediately
```

The LLM receives:
- All detected objects with class, distance (feet), position (left/center/right), velocity
- Previous scene context (what was already announced)
- Instructions to only speak when something important changed

### 3. Models (already best available)
| Component | Model | Notes |
|---|---|---|
| Detection | YOLOv11x (56.1 mAP) | Best YOLO model available |
| Depth | Depth Anything V2 **Metric Outdoor Large** | Outputs real meters, largest variant |
| Analysis | GPT-4o-mini | Fast, cheap, good at structured analysis |
| TTS | ElevenLabs Multilingual v2 | Natural Nepali speech |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Modal secrets
```bash
# Create Modal secret with your API keys
modal secret create drishtimarga-secrets \
    OPENAI_API_KEY="sk-proj-your-key-here" \
    ELEVENLABS_API_KEY="sk_your-key-here"
```

### 3. Deploy to Modal
```bash
modal deploy modal_server.py
```
Modal will print endpoint URLs like:
```
https://YOUR_USERNAME--drishtimarga-v3-drishtimargainference-infer.modal.run
https://YOUR_USERNAME--drishtimarga-v3-drishtimargainference-health.modal.run
```

### 4. Run the client
```bash
# With webcam
python local_client.py --source 0 \
    --server-url https://YOUR_USERNAME--drishtimarga-v3-drishtimargainference-infer.modal.run

# With ESP32-CAM
python local_client.py \
    --source http://192.168.6.50:81/stream \
    --server-url https://YOUR_USERNAME--drishtimarga-v3-drishtimargainference-infer.modal.run

# Audio only (no display window)
python local_client.py --source 0 --server-url <URL> --no-display

# With frame rotation (for mounted ESP32)
python local_client.py --source <URL> --server-url <URL> --rotate
```

### Local mode (no Modal, requires local GPU)
```bash
export OPENAI_API_KEY="sk-proj-..."
export ELEVENLABS_API_KEY="sk_..."

python main.py                    # webcam, full pipeline
python main.py --no-depth         # skip depth model
python main.py --model yolo11n.pt # faster detection
```

## Files

| File | Purpose |
|---|---|
| `modal_server.py` | **Modal GPU server** — YOLO + Depth + GPT + ElevenLabs |
| `local_client.py` | **Local client** — captures frames, sends to Modal, plays audio |
| `main.py` | Local-only mode (everything runs locally, needs GPU) |
| `config.py` | All tunable parameters |
| `detector.py` | YOLOv11x + ByteTrack |
| `depth_estimator.py` | Metric Depth Anything V2 Large |
| `spatial_engine.py` | Distance, velocity, threat scoring |
| `requirements.txt` | Python dependencies |

## How the LLM Announcements Work

The GPT-4o-mini system prompt instructs it to:

1. **Analyze** detected objects (class, distance in feet, position, velocity)
2. **Compare** with previous scene (what was already announced)
3. **Decide** if there's something worth saying
4. **Generate** a very short Nepali sentence (5-15 words max)
5. **Return empty** if nothing meaningful changed

Example outputs:
| Scenario | Nepali Output |
|---|---|
| Person detected ahead | "अगाडि मान्छे, ६ फिट" |
| Car approaching from right | "सावधान! गाडी दायाँबाट आउँदैछ" |
| Truck very close | "खतरा! ट्रक नजिक, ३ फिट अगाडि!" |
| Dog on left | "कुकुर बायाँतिर" |
| Path is clear | "बाटो खुला छ" |
| Nothing changed | "" (no announcement) |

## CLI Options

### local_client.py (Modal cloud mode)
```
--source          Camera index or ESP32-CAM URL
--server-url      Modal endpoint URL (required)
--jpeg-quality    JPEG compression 1-100 (default: 75)
--no-display      Audio only, no video window
--rotate          Rotate frames 90° CCW
```

### main.py (local mode)
```
--source          Camera index or ESP32-CAM URL
--model           yolo11x.pt (best), yolo11l.pt, yolo11n.pt (fast)
--conf            Detection confidence (default: 0.40)
--size            Input size 416 or 640 (default: 640)
--no-depth        Disable depth model
--depth-model     metric-outdoor-large (default), metric-outdoor-base, etc.
--no-display      Audio only
--device          auto / cuda / cpu
```

## Keys (when display is on)
| Key | Action |
|---|---|
| q | Quit |
