# channels/blink_channel.py

import time


class BlinkChannel:
    """
    Behavior Channel: Blink Aggregation Layer

    Responsibilities:
    - Track blink events and durations
    - Classify blinks into rapid and long categories
    - Detect burst patterns (rapid blinking clusters)
    - Store timestamps for cross-channel correlation

    Does NOT:
    - Compute cheating scores
    - Interpret suspicious patterns
    - Know identity internals
    """

    def __init__(
        self,
        long_blink_threshold=0.4,    # sec — deliberate eye closure
        rapid_blink_threshold=0.25,  # sec — normal/stress blink
        burst_window=10.0,           # sec — window for burst detection
        burst_min_events=4,          # blinks within window = burst
    ):
        self.long_blink_threshold  = long_blink_threshold
        self.rapid_blink_threshold = rapid_blink_threshold
        self.burst_window          = burst_window
        self.burst_min_events      = burst_min_events

        # Active blink tracking
        self.current_blink_start = None

        # Metrics
        self.total_blink_count = 0
        self.long_blink_count  = 0
        self.rapid_blink_count = 0
        self.burst_count       = 0

        # Stored data
        self.blink_events       = []  # all blink timestamps
        self.blink_durations    = []  # all blink durations
        self.rapid_blink_events = []  # timestamps of rapid blinks
        self.long_blink_events  = []  # timestamps of long blinks
        self.burst_events       = []  # timestamps of burst detections

        # State memory
        self.last_blink_state  = None
        self._last_burst_time  = None

    # -------------------------------------------------
    # Update — called every frame
    # -------------------------------------------------

    def update(self, blink_state, identity_valid):

        now = time.time()

        # Identity invalid → close active blink if any
        if not identity_valid:
            if self.current_blink_start is not None:
                self._close_blink(now)
            self.last_blink_state = None
            return

        # First valid frame
        if self.last_blink_state is None:
            self.last_blink_state = blink_state
            if blink_state.name == "BLINK":
                self.current_blink_start = now
            return

        # NO_BLINK → BLINK
        if (
            self.last_blink_state.name == "NO_BLINK"
            and blink_state.name == "BLINK"
        ):
            self.current_blink_start = now

        # BLINK → NO_BLINK
        if (
            self.last_blink_state.name == "BLINK"
            and blink_state.name == "NO_BLINK"
        ):
            if self.current_blink_start is not None:
                self._close_blink(now)

        self.last_blink_state = blink_state

    # -------------------------------------------------
    # Close Blink
    # -------------------------------------------------

    def _close_blink(self, end_time):

        start_time = self.current_blink_start
        duration   = end_time - start_time

        # Ignore ghost events under 50ms
        if duration < 0.05:
            self.current_blink_start = None
            return

        self.total_blink_count += 1
        self.blink_events.append(end_time)
        self.blink_durations.append(duration)

        # Long blink — deliberate eye closure
        if duration >= self.long_blink_threshold:
            self.long_blink_count += 1
            self.long_blink_events.append(end_time)

        # Rapid blink — normal or stress blink
        elif duration < self.rapid_blink_threshold:
            self.rapid_blink_count += 1
            self.rapid_blink_events.append(end_time)
            self._check_burst(end_time)

        self.current_blink_start = None

    # -------------------------------------------------
    # Burst Detection
    # -------------------------------------------------

    def _check_burst(self, now):
        """
        A burst = burst_min_events rapid blinks
        within burst_window seconds.
        Indicates stress or anxiety spike.
        """
        window_start = now - self.burst_window

        recent = [
            t for t in self.rapid_blink_events
            if t >= window_start
        ]

        if len(recent) >= self.burst_min_events:
            # Only count a new burst if the last burst
            # was outside the current window
            if (
                self._last_burst_time is None
                or self._last_burst_time < window_start
            ):
                self.burst_count += 1
                self.burst_events.append(now)
                self._last_burst_time = now

    # -------------------------------------------------
    # Finalize
    # -------------------------------------------------

    def finalize(self):
        if self.current_blink_start is not None:
            self._close_blink(time.time())

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    def get_summary(self):
        return {
            "total_blink_count":   self.total_blink_count,
            "long_blink_count":    self.long_blink_count,
            "rapid_blink_count":   self.rapid_blink_count,
            "burst_count":         self.burst_count,
            "blink_events":        self.blink_events,
            "blink_durations":     self.blink_durations,
            "rapid_blink_events":  self.rapid_blink_events,
            "long_blink_events":   self.long_blink_events,
            "burst_events":        self.burst_events,
        }