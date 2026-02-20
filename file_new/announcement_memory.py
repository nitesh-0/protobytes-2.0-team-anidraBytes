"""
Drishtimarga v2 — Announcement Memory

Solves the repetition problem by maintaining a state machine per tracked
object.  Each object transitions through states:

    NEW  →  ACTIVE  →  STABLE  →  (re-trigger)  →  ACTIVE
                                  →  DEPARTED     →  (removed)

State transitions control WHEN and WHETHER to speak:

    NEW:       Object just appeared.  Announce immediately.
    ACTIVE:    Object changed recently (moved, got closer, changed lane).
               Announce with moderate cooldown.
    STABLE:    Object hasn't meaningfully changed in a while.
               Suppress announcements (very long cooldown).
    DEPARTED:  Object left the scene.  Announce departure once, then remove.

Re-trigger from STABLE → ACTIVE happens when:
    - Distance changed significantly
    - Position changed (left ↔ center ↔ right)
    - Velocity changed (was still, now approaching)
    - Threat level escalated (became CRITICAL)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import AnnouncementConfig
from spatial_engine import Priority, ThreatAssessment

logger = logging.getLogger(__name__)


class ObjectState(Enum):
    NEW = "new"
    ACTIVE = "active"
    STABLE = "stable"
    DEPARTED = "departed"


@dataclass
class ObjectMemoryEntry:
    """Memory of what we've told the user about one tracked object."""
    track_id: int
    class_name: str
    state: ObjectState

    # Timing
    first_seen: float
    last_seen: float
    last_announced: float = 0.0
    last_state_change: float = 0.0

    # What was announced last
    announced_distance: float = float("inf")
    announced_position: str = ""
    announced_velocity: float = 0.0
    announced_priority: Priority = Priority.LOW
    announcement_count: int = 0

    # Current values (updated every frame)
    current_distance: float = float("inf")
    current_position: str = ""
    current_velocity: float = 0.0
    current_priority: Priority = Priority.LOW

    # Scene fingerprint: used to detect if the overall scene changed
    _stable_since: float = 0.0


class AnnouncementMemory:
    """
    Intelligent announcement filtering.

    Maintains per-object state machines and decides which assessments
    should actually be spoken, and which should be suppressed.
    """

    def __init__(self, config: AnnouncementConfig):
        self.config = config
        self.entries: Dict[int, ObjectMemoryEntry] = {}
        self._last_summary_time = 0.0
        self._last_summary_hash = ""
        self._departure_announced: set = set()  # track_ids already departure-announced

    def filter_assessments(
        self, assessments: List[ThreatAssessment]
    ) -> Tuple[List[ThreatAssessment], List[str]]:
        """
        Given raw threat assessments from SpatialEngine, return:
          - filtered_assessments: only those that should be spoken
          - departure_messages: messages for objects that left the scene

        This is the main entry point called every frame.
        """
        now = time.time()
        active_ids = {a.track_id for a in assessments}

        # ── Update memory for all current assessments ──
        for a in assessments:
            self._update_entry(a, now)

        # ── Detect departures ──
        departure_messages = self._check_departures(active_ids, now)

        # ── Filter: decide which assessments to speak ──
        to_speak: List[ThreatAssessment] = []

        # Sort by threat score (highest first) so we pick the most important
        sorted_assessments = sorted(assessments, key=lambda a: a.threat_score, reverse=True)

        for a in sorted_assessments:
            if len(to_speak) >= self.config.max_announcements_per_cycle:
                break

            entry = self.entries.get(a.track_id)
            if entry is None:
                continue

            should, reason = self._should_announce(entry, a, now)
            if should:
                to_speak.append(a)
                self._record_announcement(entry, a, now)
                logger.debug(f"SPEAK [{entry.state.value}] {reason}: {a.message}")
            else:
                logger.debug(f"SUPPRESS [{entry.state.value}] {reason}: "
                             f"{a.class_name} #{a.track_id}")

        # ── Clean up old entries ──
        self._cleanup(active_ids, now)

        return to_speak, departure_messages

    def _update_entry(self, a: ThreatAssessment, now: float):
        """Create or update a memory entry for this assessment."""
        tid = a.track_id

        if tid not in self.entries:
            # Brand new object
            self.entries[tid] = ObjectMemoryEntry(
                track_id=tid,
                class_name=a.class_name,
                state=ObjectState.NEW,
                first_seen=now,
                last_seen=now,
                last_state_change=now,
                current_distance=a.distance,
                current_position=a.position,
                current_velocity=a.approach_velocity,
                current_priority=a.priority,
                _stable_since=now,
            )
            return

        entry = self.entries[tid]
        entry.last_seen = now
        entry.class_name = a.class_name

        # Check if anything meaningful changed
        changed = self._has_meaningful_change(entry, a)

        # Update current values
        entry.current_distance = a.distance
        entry.current_position = a.position
        entry.current_velocity = a.approach_velocity
        entry.current_priority = a.priority

        # ── State transitions ──
        if entry.state == ObjectState.NEW:
            # After first announcement, move to ACTIVE
            if entry.announcement_count > 0:
                entry.state = ObjectState.ACTIVE
                entry.last_state_change = now
                entry._stable_since = now

        elif entry.state == ObjectState.ACTIVE:
            if changed:
                # Reset stability timer
                entry._stable_since = now
            elif (now - entry._stable_since) > self.config.stable_after_seconds:
                # Nothing changed for a while → STABLE
                entry.state = ObjectState.STABLE
                entry.last_state_change = now
                logger.debug(f"#{tid} {a.class_name} → STABLE "
                             f"(unchanged for {self.config.stable_after_seconds:.0f}s)")

        elif entry.state == ObjectState.STABLE:
            if changed:
                # Something meaningful changed → back to ACTIVE
                entry.state = ObjectState.ACTIVE
                entry.last_state_change = now
                entry._stable_since = now
                logger.debug(f"#{tid} {a.class_name} → ACTIVE (re-triggered)")

            # Critical escalation always re-triggers
            if a.priority == Priority.CRITICAL and entry.announced_priority != Priority.CRITICAL:
                entry.state = ObjectState.ACTIVE
                entry.last_state_change = now
                entry._stable_since = now

    def _distance_bucket(self, distance: float) -> float:
        """Bucket distance to reduce sensitivity to depth noise.
        Objects at 3.1m and 3.4m both bucket to 3m, preventing chatter."""
        if not np.isfinite(distance):
            return float("inf")
        if distance < 2.0:
            return round(distance * 2) / 2   # 0.5m buckets
        elif distance < 5.0:
            return round(distance)            # 1m buckets
        else:
            return round(distance / 2) * 2   # 2m buckets

    def _has_meaningful_change(self, entry: ObjectMemoryEntry,
                                a: ThreatAssessment) -> bool:
        """Check if the object changed enough to warrant re-announcement.
        Uses distance bucketing + absolute threshold to avoid noise-induced chatter."""
        # Distance change: require BOTH bucket change AND raw threshold exceeded
        curr_bucket = self._distance_bucket(a.distance)
        prev_bucket = self._distance_bucket(entry.announced_distance)

        if curr_bucket != prev_bucket:
            if a.distance < 3.0:
                d_thresh = self.config.distance_change_near
            else:
                d_thresh = self.config.distance_change_far
            if abs(a.distance - entry.announced_distance) > d_thresh:
                return True

        # Position change (left ↔ center ↔ right)
        if self.config.position_change and a.position != entry.announced_position:
            if entry.announced_position != "":  # skip if never announced
                return True

        # Velocity change (was still → now approaching, or vice versa)
        was_approaching = entry.announced_velocity > self.config.velocity_change_threshold
        now_approaching = a.approach_velocity > self.config.velocity_change_threshold
        if was_approaching != now_approaching:
            return True

        # Priority escalation
        if a.priority < entry.announced_priority:  # lower number = higher priority
            return True

        return False

    def _should_announce(self, entry: ObjectMemoryEntry,
                          a: ThreatAssessment, now: float) -> Tuple[bool, str]:
        """
        Decide whether to announce this object right now.
        Returns (should_speak, reason_string).
        """
        # ── NEW: always announce immediately ──
        if entry.state == ObjectState.NEW:
            return True, "new object"

        # ── CRITICAL: always announce with short cooldown ──
        if a.priority == Priority.CRITICAL:
            if (now - entry.last_announced) >= self.config.cooldown_critical:
                return True, "critical threat"
            return False, "critical cooldown"

        # ── STABLE: very rarely re-announce ──
        if entry.state == ObjectState.STABLE:
            if (now - entry.last_announced) >= self.config.cooldown_stable:
                return True, "stable periodic refresh"
            return False, "stable suppressed"

        # ── ACTIVE: moderate cooldown, announce if changed ──
        if entry.state == ObjectState.ACTIVE:
            time_since = now - entry.last_announced
            if time_since < self.config.cooldown_active:
                return False, f"active cooldown ({time_since:.1f}s < {self.config.cooldown_active}s)"

            # Only speak if something actually changed since last announcement
            if self._has_meaningful_change(entry, a):
                return True, "active with change"

            return False, "active but no change"

        # ── DEPARTED: handled separately ──
        return False, "departed/unknown state"

    def _record_announcement(self, entry: ObjectMemoryEntry,
                              a: ThreatAssessment, now: float):
        """Record that we just announced this object."""
        entry.last_announced = now
        entry.announced_distance = a.distance
        entry.announced_position = a.position
        entry.announced_velocity = a.approach_velocity
        entry.announced_priority = a.priority
        entry.announcement_count += 1

    def _check_departures(self, active_ids: set, now: float) -> List[str]:
        """Generate departure messages for objects that left the scene."""
        if not self.config.announce_departures:
            return []

        messages = []
        for tid, entry in self.entries.items():
            if tid in active_ids:
                continue
            if tid in self._departure_announced:
                continue

            since_seen = now - entry.last_seen
            if since_seen > self.config.depart_after_seconds:
                lifetime = entry.last_seen - entry.first_seen
                if lifetime >= self.config.departure_min_lifetime:
                    msg = f"{entry.class_name} has left"
                    messages.append(msg)
                    self._departure_announced.add(tid)
                    entry.state = ObjectState.DEPARTED
                    logger.debug(f"#{tid} {entry.class_name} → DEPARTED "
                                 f"(seen for {lifetime:.1f}s)")

        return messages

    def _cleanup(self, active_ids: set, now: float):
        """Remove entries that have been gone for a long time."""
        to_remove = []
        for tid, entry in self.entries.items():
            if tid not in active_ids:
                gone_for = now - entry.last_seen
                if gone_for > self.config.gone_after_seconds:
                    to_remove.append(tid)

        for tid in to_remove:
            del self.entries[tid]
            self._departure_announced.discard(tid)

    def should_do_scene_summary(self, summary_text: str) -> bool:
        """
        Check if we should speak a scene summary.
        Only speaks if enough time passed AND the scene actually changed.
        """
        now = time.time()
        if (now - self._last_summary_time) < self.config.scene_summary_interval:
            return False

        if self.config.scene_summary_only_on_change:
            if summary_text == self._last_summary_hash:
                return False

        self._last_summary_time = now
        self._last_summary_hash = summary_text
        return True

    @property
    def stats(self) -> dict:
        states = {}
        for entry in self.entries.values():
            s = entry.state.value
            states[s] = states.get(s, 0) + 1
        return {
            "total_tracked": len(self.entries),
            "states": states,
            "departures_announced": len(self._departure_announced),
        }
