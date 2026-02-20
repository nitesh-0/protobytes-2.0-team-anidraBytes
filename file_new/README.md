# Drishtimarga v2

## What Changed from v1 (and Why)

### 1. Detection: YOLOv8n → YOLOv11x
| | v1 | v2 |
|---|---|---|
| Model | YOLOv8n (3.2M params) | YOLOv11x (56.9M params) |
| mAP | 37.3 | **56.1** (+50% better) |
| Small objects | Often missed | Reliably detected |
| Overlapping objects | Frequent misses | Handled well |

### 2. Depth: Relative → Metric (the critical fix)
**Why v1 depth was wrong:**
```
v1 normalized every frame to 0-1 independently:
  depth = (depth - min) / (max - min)    ← destroys absolute scale
  meters = 0.3 + depth * 19.7           ← meaningless mapping

Result: "0.5" could mean 2m in one frame, 12m in the next
```

**v2 uses the METRIC model that outputs actual meters directly.**
No normalization, no mapping. The model output IS the distance.

| | v1 | v2 |
|---|---|---|
| Model | Depth Anything V2 Small (relative) | Depth Anything V2 **Metric** Outdoor Base |
| Output | Arbitrary 0-1 values | **Real meters** (0.1m – 80m) |
| "Chair at 2m" | Could report 0.5m or 8m depending on scene | Reports ~2m consistently |

### 3. Audio Repetition → State Machine Memory
**Why v1 was repetitive:**
- Flat 3-second cooldown per object
- No concept of "this object hasn't changed, stop talking about it"
- Scene summaries on a timer regardless of changes

**v2 uses `AnnouncementMemory` with per-object state machine:**
```
NEW → ACTIVE → STABLE → (change detected) → ACTIVE
                       → DEPARTED
```

| State | Behavior |
|---|---|
| NEW | Announce immediately on first detection |
| ACTIVE | Announce with 4s cooldown, only if distance/position/velocity changed |
| STABLE | Suppress (30s cooldown). Object hasn't changed — stop talking about it |
| DEPARTED | Announce once "car has left", then remove |

**Re-triggers from STABLE → ACTIVE:**
- Distance changed >0.4m (near) or >1m (far)
- Position changed (left ↔ center ↔ right)
- Velocity changed (was still → now approaching)
- Threat escalated to CRITICAL

## Quick Start

```bash
pip install -r requirements.txt
python main.py                    # webcam, full pipeline
python main.py --no-depth         # skip depth download, use bbox fallback
python main.py --model yolo11n.pt # faster detection
```

## Files

| File | Lines | Purpose |
|---|---|---|
| `main.py` | Pipeline orchestrator |
| `config.py` | All tunable parameters |
| `detector.py` | YOLOv11x + ByteTrack |
| `depth_estimator.py` | Metric Depth Anything V2 |
| `spatial_engine.py` | Distance, velocity, threat scoring |
| `announcement_memory.py` | **NEW** — anti-repetition state machine |
| `audio_engine.py` | Priority TTS queue |
| `visualizer.py` | Annotated video + HUD |

## CLI Options

```
--source          Camera index or ESP32 URL
--model           yolo11x.pt (best), yolo11l.pt, yolo11n.pt (fast)
--conf            Detection confidence (default: 0.40)
--size            Input size 416 or 640 (default: 640)
--no-depth        Disable depth model
--depth-model     metric-outdoor-base, metric-outdoor-small, etc.
--no-display      Audio only
--speech-rate     TTS WPM (default: 185)
--device          auto / cuda / cpu
```

## Keys
| Key | Action |
|---|---|
| q | Quit |
| d | Toggle depth map |
| s | Force scene summary |
