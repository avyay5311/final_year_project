# channels/face_identity_channel.py

import time
from core_ai.primary_identity_sm import IdentityState


class FaceIdentityChannel:
    """
    Behavior Channel: Face Identity Aggregation Layer

    Responsibilities:
    - Track primary candidate absence intervals
    - Track multiple face presence intervals
    - Track impersonation suspect intervals
    - Track identity uncertain intervals
    - Store timestamps for cross-channel correlation

    Unlike other channels this channel:
    - Does NOT take identity_valid as a parameter
    - It IS the identity source — it tracks identity
      state directly
    - Does NOT need a face width or landmark input
    - Receives raw IdentityState enum every frame

    Does NOT:
    - Compute cheating scores
    - Make identity decisions
    - Know anything about face embeddings or landmarks
    """

    def __init__(self):

        # -------------------------------------------------
        # Absence tracking
        # PRIMARY_TEMP_ABSENT
        # -------------------------------------------------
        self.absence_duration       = 0.0
        self.longest_absence        = 0.0
        self.absence_count          = 0
        self.absence_intervals      = []   # (start, end)
        self.absence_events         = []   # end timestamps
        self._absence_start         = None

        # -------------------------------------------------
        # Multiple faces tracking
        # MULTIPLE_FACES_PRESENT
        # -------------------------------------------------
        self.multiple_face_duration  = 0.0
        self.longest_multiple_face   = 0.0
        self.multiple_face_count     = 0
        self.multiple_face_intervals = []   # (start, end)
        self.multiple_face_events    = []   # end timestamps
        self._multiple_face_start    = None

        # -------------------------------------------------
        # Impersonation tracking
        # IMPERSONATION_SUSPECT
        # -------------------------------------------------
        self.impersonation_duration  = 0.0
        self.longest_impersonation   = 0.0
        self.impersonation_count     = 0
        self.impersonation_intervals = []   # (start, end)
        self.impersonation_events    = []   # end timestamps
        self._impersonation_start    = None

        # -------------------------------------------------
        # Uncertain tracking
        # PRIMARY_TEMP_UNCERTAIN
        # -------------------------------------------------
        self.uncertain_duration      = 0.0
        self.uncertain_count         = 0
        self.uncertain_intervals     = []   # (start, end)
        self._uncertain_start        = None

        # State memory
        self.last_state = None

    # -------------------------------------------------
    # Update — called every frame
    # -------------------------------------------------

    def update(self, identity_state):

        now = time.time()

        # First frame
        if self.last_state is None:
            self.last_state = identity_state
            self._open_interval(identity_state, now)
            return

        # State has not changed — nothing to close or open
        if identity_state == self.last_state:
            return

        # State changed — close previous interval
        self._close_interval(self.last_state, now)

        # Open new interval
        self._open_interval(identity_state, now)

        self.last_state = identity_state

    # -------------------------------------------------
    # Open Interval
    # -------------------------------------------------

    def _open_interval(self, state, now):

        if state == IdentityState.PRIMARY_TEMP_ABSENT:
            self._absence_start = now

        elif state == IdentityState.MULTIPLE_FACES_PRESENT:
            self._multiple_face_start = now

        elif state == IdentityState.IMPERSONATION_SUSPECT:
            self._impersonation_start = now

        elif state == IdentityState.PRIMARY_TEMP_UNCERTAIN:
            self._uncertain_start = now

    # -------------------------------------------------
    # Close Interval
    # -------------------------------------------------

    def _close_interval(self, state, end_time):

        if state == IdentityState.PRIMARY_TEMP_ABSENT:
            if self._absence_start is not None:
                duration = end_time - self._absence_start

                # Ignore ghost events under 100ms
                if duration >= 0.1:
                    self.absence_duration += duration
                    self.absence_count    += 1
                    self.absence_events.append(end_time)
                    self.absence_intervals.append(
                        (self._absence_start, end_time)
                    )
                    if duration > self.longest_absence:
                        self.longest_absence = duration

                self._absence_start = None

        elif state == IdentityState.MULTIPLE_FACES_PRESENT:
            if self._multiple_face_start is not None:
                duration = end_time - self._multiple_face_start

                if duration >= 0.1:
                    self.multiple_face_duration += duration
                    self.multiple_face_count    += 1
                    self.multiple_face_events.append(end_time)
                    self.multiple_face_intervals.append(
                        (self._multiple_face_start, end_time)
                    )
                    if duration > self.longest_multiple_face:
                        self.longest_multiple_face = duration

                self._multiple_face_start = None

        elif state == IdentityState.IMPERSONATION_SUSPECT:
            if self._impersonation_start is not None:
                duration = end_time - self._impersonation_start

                if duration >= 0.1:
                    self.impersonation_duration += duration
                    self.impersonation_count    += 1
                    self.impersonation_events.append(end_time)
                    self.impersonation_intervals.append(
                        (self._impersonation_start, end_time)
                    )
                    if duration > self.longest_impersonation:
                        self.longest_impersonation = duration

                self._impersonation_start = None

        elif state == IdentityState.PRIMARY_TEMP_UNCERTAIN:
            if self._uncertain_start is not None:
                duration = end_time - self._uncertain_start

                if duration >= 0.1:
                    self.uncertain_duration += duration
                    self.uncertain_count    += 1
                    self.uncertain_intervals.append(
                        (self._uncertain_start, end_time)
                    )

                self._uncertain_start = None

    # -------------------------------------------------
    # Finalize
    # -------------------------------------------------

    def finalize(self):
        """
        Close any intervals that were still open
        when the session ended.
        """
        now = time.time()
        if self.last_state is not None:
            self._close_interval(self.last_state, now)

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    def get_summary(self):
        return {
            # Absence
            "absence_duration":        self.absence_duration,
            "longest_absence":         self.longest_absence,
            "absence_count":           self.absence_count,
            "absence_intervals":       self.absence_intervals,
            "absence_events":          self.absence_events,

            # Multiple faces
            "multiple_face_duration":  self.multiple_face_duration,
            "longest_multiple_face":   self.longest_multiple_face,
            "multiple_face_count":     self.multiple_face_count,
            "multiple_face_intervals": self.multiple_face_intervals,
            "multiple_face_events":    self.multiple_face_events,

            # Impersonation
            "impersonation_duration":  self.impersonation_duration,
            "longest_impersonation":   self.longest_impersonation,
            "impersonation_count":     self.impersonation_count,
            "impersonation_intervals": self.impersonation_intervals,
            "impersonation_events":    self.impersonation_events,

            # Uncertain
            "uncertain_duration":      self.uncertain_duration,
            "uncertain_count":         self.uncertain_count,
            "uncertain_intervals":     self.uncertain_intervals,
        }