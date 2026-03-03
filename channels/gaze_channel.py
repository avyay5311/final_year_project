# channels/gaze_channel.py

import time


class GazeChannel:
    """
    Behavior Channel: Gaze Aggregation Layer

    Responsibilities:
    - Aggregate LOOKING_AWAY intervals
    - Classify intervals into three duration buckets:
        rapid     < rapid_threshold        (quick glance)
        moderate  rapid → sustained        (deliberate look)
        sustained ≥ sustained_threshold    (prolonged absence)
    - Store timestamps for cross-channel correlation

    Does NOT:
    - Compute cheating scores
    - Interpret suspicious patterns
    - Know identity internals
    """

    def __init__(
        self,
        rapid_threshold=1.5,
        moderate_threshold=5.0,
    ):
        self.rapid_threshold    = rapid_threshold
        self.moderate_threshold = moderate_threshold

        # Active interval tracking
        self.current_away_start = None

        # Metrics
        self.total_away_duration       = 0.0
        self.longest_continuous_away   = 0.0

        # Three bucket counts
        self.rapid_shift_count    = 0   # < 1.5 sec
        self.moderate_away_count  = 0   # 1.5 → 5.0 sec
        self.sustained_away_count = 0   # ≥ 5.0 sec

        # Stored data
        self.away_intervals       = []  # all (start, end) pairs
        self.rapid_shift_events   = []  # timestamps
        self.moderate_away_events = []  # timestamps
        self.sustained_events     = []  # timestamps

        # State memory
        self.last_gaze_state = None

    # -------------------------------------------------
    # Update — called every frame
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
    # Close Interval
    # -------------------------------------------------

    def _close_interval(self, end_time):

        start_time = self.current_away_start
        duration   = end_time - start_time

        # Ignore ghost events under 50ms
        if duration < 0.05:
            self.current_away_start = None
            return

        self.away_intervals.append((start_time, end_time))
        self.total_away_duration += duration

        if duration > self.longest_continuous_away:
            self.longest_continuous_away = duration

        # --- Three bucket classification ---

        # Rapid — quick involuntary glance
        if duration < self.rapid_threshold:
            self.rapid_shift_count += 1
            self.rapid_shift_events.append(end_time)

        # Sustained — prolonged attention loss
        elif duration >= self.moderate_threshold:
            self.sustained_away_count += 1
            self.sustained_events.append(end_time)

        # Moderate — deliberate look (most suspicious bucket)
        else:
            self.moderate_away_count += 1
            self.moderate_away_events.append(end_time)

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
            "total_away_duration":     self.total_away_duration,
            "longest_continuous_away": self.longest_continuous_away,
            "rapid_shift_count":       self.rapid_shift_count,
            "moderate_away_count":     self.moderate_away_count,
            "sustained_away_count":    self.sustained_away_count,
            "rapid_shift_events":      self.rapid_shift_events,
            "moderate_away_events":    self.moderate_away_events,
            "sustained_events":        self.sustained_events,
            "away_intervals":          self.away_intervals,
        }