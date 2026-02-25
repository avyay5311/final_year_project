# channels/blink_channel.py

import time


class BlinkChannel:
    """
    Behavior Channel: Blink Aggregation Layer

    Responsibilities:
    - Track blink events
    - Measure blink durations
    - Store rapid blink timestamps
    - Store long blink count

    Does NOT:
    - Compute cheating
    - Interpret suspicious patterns
    """

    def __init__(self):

        # Active blink tracking
        self.current_blink_start = None

        # Metrics
        self.total_blink_count = 0
        self.long_blink_count = 0

        # Stored data
        self.blink_events = []
        self.blink_durations = []
        self.rapid_blink_events = []

        # State memory
        self.last_blink_state = None

        # Thresholds (pure classification, not suspicion)
        self.long_blink_threshold = 0.6
        self.rapid_blink_threshold = 0.25

    # -------------------------------------------------
    # Update
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
        duration = end_time - start_time

        self.total_blink_count += 1

        self.blink_events.append(end_time)
        self.blink_durations.append(duration)

        # Long blink
        if duration >= self.long_blink_threshold:
            self.long_blink_count += 1

        # Rapid blink
        if duration <= self.rapid_blink_threshold:
            self.rapid_blink_events.append(end_time)

        self.current_blink_start = None

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
            "total_blink_count": self.total_blink_count,
            "long_blink_count": self.long_blink_count,
            "blink_events": self.blink_events,
            "blink_durations": self.blink_durations,
            "rapid_blink_events": self.rapid_blink_events,
        }