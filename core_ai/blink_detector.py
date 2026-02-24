# core_ai/blink_detector.py

import numpy as np
from enum import Enum
from collections import deque


class BlinkState(Enum):
    BLINK = "BLINK"
    NO_BLINK = "NO_BLINK"


class BlinkDetector:
    """
    Robust EAR-based blink detector using MediaPipe Face Mesh landmarks.

    Fixes:
    - EAR jitter when face is far
    - False positives due to squint / gaze
    - Pixel noise instability
    """

    # MediaPipe eye landmark indices
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(
        self,
        ear_threshold=0.21,
        consec_frames=2,
        ear_window=3,
        reopen_margin=0.03
    ):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.reopen_margin = reopen_margin

        # Temporal state
        self._closed_counter = 0
        self._blink_active = False

        # EAR smoothing buffer
        self._ear_buffer = deque(maxlen=ear_window)

    @staticmethod
    def _euclidean(p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.linalg.norm(p1 - p2)

    def _eye_ear(self, landmarks, eye_idx):
        A = self._euclidean(landmarks[eye_idx[1]], landmarks[eye_idx[5]])
        B = self._euclidean(landmarks[eye_idx[2]], landmarks[eye_idx[4]])
        C = self._euclidean(landmarks[eye_idx[0]], landmarks[eye_idx[3]])

        if C == 0:
            return 0.0

        return (A + B) / (2.0 * C)

    def compute_ear(self, landmarks):
        """
        landmarks: list of 468 (x, y) MediaPipe landmarks
        """
        if landmarks is None or len(landmarks) < 468:
            return None

        left_ear = self._eye_ear(landmarks, self.LEFT_EYE)
        right_ear = self._eye_ear(landmarks, self.RIGHT_EYE)

        ear = (left_ear + right_ear) / 2.0

        # Smooth EAR (reduce pixel noise)
        self._ear_buffer.append(ear)
        return sum(self._ear_buffer) / len(self._ear_buffer)

    def update(self, landmarks):
        """
        Returns: BlinkState
        """
        ear = self.compute_ear(landmarks)
        if ear is None:
            return BlinkState.NO_BLINK

        # Eye closed
        if ear < self.ear_threshold:
            self._closed_counter += 1

            if self._closed_counter >= self.consec_frames and not self._blink_active:
                self._blink_active = True
                return BlinkState.BLINK

            return BlinkState.NO_BLINK

        # Eye reopened clearly (anti-squint)
        if ear > self.ear_threshold + self.reopen_margin:
            self._closed_counter = 0
            self._blink_active = False

        return BlinkState.NO_BLINK
