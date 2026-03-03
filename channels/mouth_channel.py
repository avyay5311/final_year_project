# channels/mouth_channel.py

import time
from core_ai.mouth_tracker import MouthState


class MouthChannel:
    """
    Behavior Channel: Mouth Activity Aggregation Layer

    Responsibilities:
    - Track mouth active intervals (MOUTH_OPEN or MOUTH_MOVING)
    - Measure active episode durations
    - Detect sustained speech episodes
    - Detect rapid burst patterns (whispering)
    - Store timestamps for cross-channel correlation

    Does NOT:
    - Compute cheating scores
    - Interpret suspicious patterns
    - Know anything about identity internals
    """

    def __init__(
        self,
        sustained_threshold=3.0,   # seconds — deliberate speech
        rapid_threshold=1.5,       # seconds — yawn / single word
        burst_window=15.0,         # seconds — window to detect burst
        burst_min_events=3,        # rapid events within burst_window = burst
    ):
        self.sustained_threshold = sustained_threshold
        self.rapid_threshold     = rapid_threshold
        self.burst_window        = burst_window
        self.burst_min_events    = burst_min_events

        # Active interval tracking
        self.current_active_start = None

        # Metrics
        self.total_active_duration   = 0.0
        self.longest_active_duration = 0.0
        self.sustained_active_count  = 0
        self.rapid_active_count      = 0
        self.burst_count             = 0

        # Stored data
        self.active_intervals     = []   # (start, end) pairs
        self.rapid_active_events  = []   # timestamps of rapid events
        self.sustained_events     = []   # timestamps of sustained events
        self.burst_events         = []   # timestamps of burst detections ← ADDED
        self.all_event_durations  = []   # every episode duration

        # State memory
        self.last_mouth_state  = None
        self._last_burst_time  = None   # moved out of hasattr check

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    @staticmethod
    def _is_active(mouth_state):
        return mouth_state in (
            MouthState.MOUTH_OPEN,
            MouthState.MOUTH_MOVING,
        )

    # -------------------------------------------------
    # Update — called every frame
    # -------------------------------------------------

    def update(self, mouth_state, identity_valid):

        now = time.time()

        # Identity invalid → close any active interval
        if not identity_valid:
            if self.current_active_start is not None:
                self._close_interval(now)
            self.last_mouth_state = None
            return

        # First valid frame
        if self.last_mouth_state is None:
            self.last_mouth_state = mouth_state
            if self._is_active(mouth_state):
                self.current_active_start = now
            return

        was_active = self._is_active(self.last_mouth_state)
        is_active  = self._is_active(mouth_state)

        # CLOSED → ACTIVE
        if not was_active and is_active:
            self.current_active_start = now

        # ACTIVE → CLOSED
        if was_active and not is_active:
            if self.current_active_start is not None:
                self._close_interval(now)

        self.last_mouth_state = mouth_state

    # -------------------------------------------------
    # Close Interval
    # -------------------------------------------------

    def _close_interval(self, end_time):

        start_time = self.current_active_start
        duration   = end_time - start_time

        # Ignore ghost events under 50ms
        if duration < 0.05:
            self.current_active_start = None
            return

        self.active_intervals.append((start_time, end_time))
        self.all_event_durations.append(duration)
        self.total_active_duration += duration

        if duration > self.longest_active_duration:
            self.longest_active_duration = duration

        # Sustained episode — deliberate speech
        if duration >= self.sustained_threshold:
            self.sustained_active_count += 1
            self.sustained_events.append(end_time)

        # Rapid episode — yawn / single word
        elif duration < self.rapid_threshold:
            self.rapid_active_count += 1
            self.rapid_active_events.append(end_time)
            self._check_burst(end_time)

        self.current_active_start = None

    # -------------------------------------------------
    # Burst Detection
    # -------------------------------------------------

    def _check_burst(self, now):
        """
        A burst = burst_min_events rapid events
        within burst_window seconds.
        Whispering produces exactly this pattern.
        """
        window_start = now - self.burst_window

        recent = [
            t for t in self.rapid_active_events
            if t >= window_start
        ]

        if len(recent) >= self.burst_min_events:
            # Only count a new burst if last burst was
            # outside the current window — no double counting
            if (
                self._last_burst_time is None
                or self._last_burst_time < window_start
            ):
                self.burst_count += 1
                self.burst_events.append(now)        # ← ADDED
                self._last_burst_time = now

    # -------------------------------------------------
    # Finalize
    # -------------------------------------------------

    def finalize(self):
        if self.current_active_start is not None:
            self._close_interval(time.time())

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    def get_summary(self):
        return {
            "total_active_duration":   self.total_active_duration,
            "longest_active_duration": self.longest_active_duration,
            "sustained_active_count":  self.sustained_active_count,
            "rapid_active_count":      self.rapid_active_count,
            "burst_count":             self.burst_count,
            "active_intervals":        self.active_intervals,
            "rapid_active_events":     self.rapid_active_events,
            "sustained_events":        self.sustained_events,
            "burst_events":            self.burst_events,  # ← ADDED
            "all_event_durations":     self.all_event_durations,
        }