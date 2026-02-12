
import time
import numpy as np
from config import AppConfig, SpatialConfig, ThreatConfig, AudioConfig
from spatial_engine import SpatialEngine, TrackedObject, ThreatAssessment, Priority
from audio_engine import AudioEngine
from detector import Detection

def test_depth_inversion():
    print("Testing Depth Inversion...")
    s_cfg = SpatialConfig()
    t_cfg = ThreatConfig()
    engine = SpatialEngine(s_cfg, t_cfg)
    
    # Mock detection
    det = Detection(
        track_id=1, class_name="person", class_id=0, confidence=0.9,
        bbox=np.array([100, 100, 200, 200]), center=np.array([150, 150]),
        bbox_width=100, bbox_height=100, timestamp=time.time()
    )
    
    # Mock depth map (normalized 0..1)
    # 1.0 = closest (should be low distance)
    # 0.0 = furthest (should be high distance)
    
    # Test Close (High disparity/rel_depth)
    depth_map_close = np.ones((480, 640), dtype=float) * 0.95
    dist_close = engine.estimate_distance(det, depth_map_close, (480, 640))
    print(f"Rel Depth 0.95 -> Distance: {dist_close:.2f}m (Expected ~0.5-1.0m)")
    
    # Test Far (Low disparity/rel_depth)
    depth_map_far = np.ones((480, 640), dtype=float) * 0.05
    dist_far = engine.estimate_distance(det, depth_map_far, (480, 640))
    print(f"Rel Depth 0.05 -> Distance: {dist_far:.2f}m (Expected >10m)")
    
    if dist_close < dist_far:
        print("PASS: Inversion logic works (Close < Far)")
    else:
        print("FAIL: Inversion logic broken (Close >= Far)")

def test_audio_suppression():
    print("\nTesting Audio Suppression...")
    a_cfg = AudioConfig(suppress_static_objects=True, cooldown=1.0)
    engine = AudioEngine(a_cfg)
    
    # Mock assessment
    assess = ThreatAssessment(
        track_id=1, class_name="chair", distance=2.0, position="center",
        approach_velocity=0.0, time_to_collision=float("inf"),
        threat_score=1.0, priority=Priority.NORMAL, is_new=False, message="Chair ahead"
    )
    
    # 1. First time (not in cooldowns) -> Should announce (if we call process_assessments, but we test inner logic)
    # Manually populate cooldown to simulate previous announcement
    engine._cooldowns[1] = (time.time() - 2.0, 2.0) # Last announce 2s ago at 2.0m
    
    # Now: 2s passed ( > cooldown 1.0), but dist same, vel same.
    should = engine._should_announce(assess, time.time())
    print(f"Static object after cooldown: Should Announce? {should} (Expected: False)")
    
    # 2. Moving object
    assess_moving = ThreatAssessment(
        track_id=1, class_name="chair", distance=1.8, position="center",
        approach_velocity=0.5, time_to_collision=float("inf"), # Velocity > 0.2
        threat_score=1.0, priority=Priority.NORMAL, is_new=False, message="Chair detected"
    )
    should_move = engine._should_announce(assess_moving, time.time())
    print(f"Moving object: Should Announce? {should_move} (Expected: True)")
    
    # 3. Moved object (teleport)
    assess_moved = ThreatAssessment(
        track_id=1, class_name="chair", distance=1.0, position="center", # Changed by 1m
        approach_velocity=0.0, time_to_collision=float("inf"),
        threat_score=1.0, priority=Priority.NORMAL, is_new=False, message="Chair detected"
    )
    should_changed = engine._should_announce(assess_moved, time.time())
    print(f"Moved object (dist change): Should Announce? {should_changed} (Expected: True)")

if __name__ == "__main__":
    test_depth_inversion()
    test_audio_suppression()
