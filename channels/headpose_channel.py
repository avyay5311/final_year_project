import time


class HeadPoseChannel:
    """
    Behavior Channel: Head Pose Aggregation Layer

    Tracks HEAD_AWAY intervals.
    Mirrors GazeChannel structure.
    """

    def __init__(self):

        # Active interval
        self.current_away_start = None

        # Metrics
        self.total_away_duration = 0.0
        self.longest_continuous_away = 0.0
        self.rapid_away_count = 0
        self.sustained_away_count = 0

        # Stored data
        self.away_intervals = []
        self.rapid_away_events = []

        # State memory
        self.last_state = None

        # Thresholds (classification only)
        self.rapid_threshold = 1.5
        self.sustained_threshold = 5.0

    # -------------------------------------------------
    # Update
    # -------------------------------------------------

    def update(self, headpose_state, identity_valid):

        now = time.time()
        state_name = headpose_state.name

        # Identity invalid → close interval
        if not identity_valid:
            if self.current_away_start is not None:
                self._close_interval(now)

            self.last_state = None
            return

        # First valid frame
        if self.last_state is None:
            self.last_state = state_name

            if state_name == "HEAD_AWAY":
                self.current_away_start = now

            return

        # CENTER → AWAY
        if (
            self.last_state == "HEAD_CENTER"
            and state_name == "HEAD_AWAY"
        ):
            self.current_away_start = now

        # AWAY → CENTER
        if (
            self.last_state == "HEAD_AWAY"
            and state_name == "HEAD_CENTER"
        ):
            if self.current_away_start is not None:
                self._close_interval(now)

        self.last_state = state_name

    # -------------------------------------------------
    # Close Interval
    # -------------------------------------------------

    def _close_interval(self, end_time):

        start_time = self.current_away_start
        duration = end_time - start_time

        self.away_intervals.append((start_time, end_time))

        self.total_away_duration += duration

        if duration > self.longest_continuous_away:
            self.longest_continuous_away = duration

        if duration < self.rapid_threshold:
            self.rapid_away_count += 1
            self.rapid_away_events.append(end_time)

        if duration >= self.sustained_threshold:
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
            "rapid_away_count": self.rapid_away_count,
            "sustained_away_count": self.sustained_away_count,
            "rapid_away_events": self.rapid_away_events,
            "away_intervals": self.away_intervals,
        }