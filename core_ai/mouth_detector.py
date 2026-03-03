import numpy as np
from enum import Enum
from collections import deque


class MouthState(Enum):
    MOUTH_STILL = "MOUTH_STILL"
    MOUTH_MOVING = "MOUTH_MOVING"


class MouthDetector:

    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_CORNER = 78
    RIGHT_CORNER = 308

    def __init__(
        self,
        movement_threshold=0.015,
        smoothing_window=4,
        delta_window=5,
        min_stable_frames=3,
        min_face_width=120,
    ):

        self.movement_threshold = movement_threshold
        self.min_face_width = min_face_width

        self._mar_buffer = deque(maxlen=smoothing_window)
        self._delta_buffer = deque(maxlen=delta_window)

        self._previous_mar = None

        self._current_state = MouthState.MOUTH_STILL
        self._candidate_state = None
        self._candidate_count = 0
        self._min_stable_frames = min_stable_frames

        self._mar = 0.0

    @property
    def mar(self):
        return self._mar

    def _compute_mar(self, landmarks):

        if landmarks is None or len(landmarks) < 309:
            return None

        top = np.array(landmarks[self.UPPER_LIP])
        bottom = np.array(landmarks[self.LOWER_LIP])
        left = np.array(landmarks[self.LEFT_CORNER])
        right = np.array(landmarks[self.RIGHT_CORNER])

        vertical = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(left - right)

        if horizontal == 0:
            return None

        return vertical / horizontal

    def _smooth_mar(self, mar):
        self._mar_buffer.append(mar)
        self._mar = sum(self._mar_buffer) / len(self._mar_buffer)

    def _detect_movement(self):

        if self._previous_mar is None:
            self._previous_mar = self._mar
            return MouthState.MOUTH_STILL

        delta = abs(self._mar - self._previous_mar)
        self._delta_buffer.append(delta)
        self._previous_mar = self._mar

        if len(self._delta_buffer) < self._delta_buffer.maxlen:
            return MouthState.MOUTH_STILL

        avg_delta = sum(self._delta_buffer) / len(self._delta_buffer)

        if avg_delta > self.movement_threshold:
            return MouthState.MOUTH_MOVING
        else:
            return MouthState.MOUTH_STILL

    def _stability_filter(self, new_state):

        if new_state != self._current_state:
            if self._candidate_state == new_state:
                self._candidate_count += 1
            else:
                self._candidate_state = new_state
                self._candidate_count = 1

            if self._candidate_count >= self._min_stable_frames:
                self._current_state = new_state
                self._candidate_state = None
                self._candidate_count = 0
        else:
            self._candidate_state = None
            self._candidate_count = 0

        return self._current_state

    def update(self, landmarks, face_width):

        if face_width < self.min_face_width:
            return self._current_state

        mar = self._compute_mar(landmarks)
        if mar is None:
            return self._current_state

        self._smooth_mar(mar)

        raw_state = self._detect_movement()
        return self._stability_filter(raw_state)