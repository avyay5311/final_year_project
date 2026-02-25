import time


class GazeChannel:
    """
    Behavior Channel: Gaze Aggregation Layer

    Responsibilities:
    - Aggregate LOOKING_AWAY intervals
    - Track duration metrics
    - Store rapid shift timestamps
    - Store intervals for future reasoning

    Does NOT:
    - Compute cheating
    - Interpret density
    - Know identity internals
    """

    def __init__(self):

        # Active interval tracking
        self.current_away_start = None

        # Metrics
        self.total_away_duration = 0.0
        self.longest_continuous_away = 0.0
        self.rapid_shift_count = 0
        self.sustained_away_count = 0

        # Stored intervals
        self.away_intervals = []

        # NEW: Rapid shift timestamps
        self.rapid_shift_events = []

        # Previous gaze state
        self.last_gaze_state = None

    # -------------------------------------------------
    # Update
    # -------------------------------------------------

    def update(self, gaze_state, identity_valid):

        now = time.time()

        # Identity invalid → close interval
        if not identity_valid:
            if self.current_away_start is not None:
                self._close_interval(now)

            self.last_gaze_state = None
            return

        # NO_PUPILS → close interval
        if gaze_state == "NO_PUPILS":
            if self.current_away_start is not None:
                self._close_interval(now)

            self.last_gaze_state = None
            return

        # First valid frame
        if self.last_gaze_state is None:
            self.last_gaze_state = gaze_state

            if gaze_state == "LOOKING_AWAY":
                self.current_away_start = now

            return

        # CENTER → AWAY
        if (
            self.last_gaze_state == "LOOKING_CENTER"
            and gaze_state == "LOOKING_AWAY"
        ):
            self.current_away_start = now

        # AWAY → CENTER
        if (
            self.last_gaze_state == "LOOKING_AWAY"
            and gaze_state == "LOOKING_CENTER"
        ):
            if self.current_away_start is not None:
                self._close_interval(now)

        self.last_gaze_state = gaze_state

    # -------------------------------------------------
    # Close interval
    # -------------------------------------------------

    def _close_interval(self, end_time):

        start_time = self.current_away_start
        duration = end_time - start_time

        self.away_intervals.append((start_time, end_time))

        self.total_away_duration += duration

        if duration > self.longest_continuous_away:
            self.longest_continuous_away = duration

        # Rapid shift (< 1.5 sec)
        if duration < 1.5:
            self.rapid_shift_count += 1

            # NEW: store timestamp of rapid shift
            self.rapid_shift_events.append(end_time)

        # Sustained away (≥ 5 sec)
        if duration >= 5.0:
            self.sustained_away_count += 1

        self.current_away_start = None

    # -------------------------------------------------
    # Finalize
    # -------------------------------------------------

    def finalize(self):
        if self.current_away_start is not None:
            self._close_interval(time.time())

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    def get_summary(self):
        return {
            "total_away_duration": self.total_away_duration,
            "longest_continuous_away": self.longest_continuous_away,
            "rapid_shift_count": self.rapid_shift_count,
            "sustained_away_count": self.sustained_away_count,
            "rapid_shift_events": self.rapid_shift_events,
            "away_intervals": self.away_intervals,
        }