# core_ai/blink_detector.py

import numpy as np
from enum import Enum


class BlinkState(Enum):
    BLINK = "BLINK"
    NO_BLINK = "NO_BLINK"


class BlinkDetector:
    """
    EAR-based blink detector using MediaPipe Face Mesh landmarks.
    Deterministic + temporal debouncing.
    """

    # MediaPipe eye landmark indices (stable)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self, ear_threshold=0.21, consec_frames=2):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames

        self._closed_counter = 0
        self._blink_active = False

    @staticmethod
    def _euclidean(p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.linalg.norm(p1 - p2)

    def _eye_ear(self, landmarks, eye_idx):
        """
        landmarks: list[(x, y)]
        eye_idx: list of 6 indices
        """
        A = self._euclidean(landmarks[eye_idx[1]], landmarks[eye_idx[5]])
        B = self._euclidean(landmarks[eye_idx[2]], landmarks[eye_idx[4]])
        C = self._euclidean(landmarks[eye_idx[0]], landmarks[eye_idx[3]])

        if C == 0:
            return 0.0

        return (A + B) / (2.0 * C)

    def compute_ear(self, landmarks):
        """
        landmarks: MediaPipe landmarks (list of 468 (x,y))
        """
        if landmarks is None or len(landmarks) < 468:
            return None

        left_ear = self._eye_ear(landmarks, self.LEFT_EYE)
        right_ear = self._eye_ear(landmarks, self.RIGHT_EYE)

        return (left_ear + right_ear) / 2.0

    def update(self, landmarks):
        """
        Returns: BlinkState
        """
        ear = self.compute_ear(landmarks)
        if ear is None:
            return BlinkState.NO_BLINK

        if ear < self.ear_threshold:
            self._closed_counter += 1

            if self._closed_counter >= self.consec_frames and not self._blink_active:
                self._blink_active = True
                return BlinkState.BLINK

            return BlinkState.NO_BLINK

        # Eyes open again
        self._closed_counter = 0
        self._blink_active = False
        return BlinkState.NO_BLINK
