# Drishtimarga — Lighting the Path Beyond Sight

Real-time AI-powered wearable navigation system for the visually impaired.
Detects objects, estimates distances, tracks moving hazards, and delivers
spoken audio feedback.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with webcam (simplest — starts immediately)
python main.py

# 3. Run with ESP32-CAM
python main.py --source "http://192.168.4.1:81/stream"
```

## Architecture

```
Camera (Webcam / ESP32-CAM)
    │
    ▼
┌─────────────────────────────────────┐
│  FrameGrabber (threaded)            │ ← Always grabs latest frame
│  Drops stale frames automatically   │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌──────────┐     ┌──────────────────┐
│ YOLO11m  │     │ Depth Anything   │   ← Run in same thread but depth
│ +ByteTrack│    │ V2 Base          │     skips frames for speed
└────┬─────┘     └───────┬──────────┘
     │                   │
     └─────────┬─────────┘
               ▼
    ┌─────────────────────┐
    │ SpatialEngine       │ ← Merges detection + depth
    │ • Distance estimate │    Computes velocity, TTC
    │ • Velocity tracking │    Scores threats by priority
    │ • Threat scoring    │
    │ • Occupancy grid    │
    └──────────┬──────────┘
               ▼
    ┌─────────────────────┐
    │ AudioEngine         │ ← Priority queue
    │ • Urgent interrupts │    Cooldown logic
    │ • Scene summaries   │    Non-blocking TTS
    └─────────────────────┘
```

## Project Files

| File | Purpose |
|---|---|
| `main.py` | Pipeline orchestrator — entry point |
| `config.py` | All tunable parameters (thresholds, models, etc.) |
| `detector.py` | YOLO11 detection + ByteTrack tracking |
| `depth_estimator.py` | Depth Anything V2 monocular depth |
| `spatial_engine.py` | 3D tracking, velocity, threat scoring, grouped summaries |
| `audio_engine.py` | Priority TTS with cooldowns, class dedup & interrupts |
| `visualizer.py` | Annotated video display with HUD |

## Command-Line Options

```bash
python main.py --help

Options:
  --source SOURCE       Camera index (0,1) or ESP32-CAM URL
  --model MODEL         YOLO model (yolo11n.pt, yolo11m.pt, yolov8s.pt)
  --conf FLOAT          Detection confidence (default: 0.35)
  --size INT            YOLO input size: 320, 416, 640 (default: 640)
  --no-depth            Disable depth model (faster, uses bbox fallback)
  --depth-model NAME    Depth variant: small, base, or large (default: base)
  --no-display          Audio-only mode (no video window)
  --speech-rate INT     TTS words per minute (default: 190)
  --cooldown FLOAT      Seconds between re-announcing same object (default: 5)
  --device DEVICE       auto, cuda, or cpu (default: auto)
```

## Keyboard Controls (when display is on)

| Key | Action |
|---|---|
| `q` | Quit |
| `d` | Toggle depth map display |
| `s` | Force scene summary announcement |

## Features

- **80+ object classes** detected out of the box (COCO dataset)
- **Persistent tracking** — same object keeps its ID across frames
- **Distance estimation** via Depth Anything V2 (inverted disparity → calibrated meters)
- **Velocity & TTC** — detects approaching objects, computes time-to-collision
- **Priority-based audio** — critical dangers interrupt, background objects suppressed
- **Smart grouping** — "3 chairs nearby" instead of announcing each chair
- **Cooldown system** — per-object and per-class dedup, no repetitive announcements
- **Scene summaries** — periodic grouped overview of surroundings
- **Occupancy grid** — spatial memory of nearby objects

## Performance Expectations

| Setup | Expected FPS | Latency |
|---|---|---|
| RTX 3060 + depth model | ~25-35 FPS | ~150ms |
| RTX 3060 no depth | ~50-80 FPS | ~80ms |
| GTX 1650 + depth model | ~15-20 FPS | ~250ms |
| CPU only (no depth) | ~8-15 FPS | ~400ms |

## ESP32-CAM Setup

1. Flash the CameraWebServer example from Arduino IDE
2. Set `FRAMESIZE_VGA` (640x480) and `jpeg_quality = 12`
3. Connect laptop to ESP32's WiFi AP
4. Run: `python main.py --source "http://192.168.4.1:81/stream"`

## Customization

Edit `config.py` to tune:
- **Detection sensitivity** — confidence thresholds, input size
- **Danger weights** — which objects are more threatening
- **Audio behavior** — cooldown, speech rate, summary intervals
- **Distance calibration** — focal length, known object sizes
