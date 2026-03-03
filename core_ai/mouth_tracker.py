# core_ai/mouth_tracker.py

import numpy as np
from enum import Enum
from collections import deque


class MouthState(Enum):
    MOUTH_CLOSED = "MOUTH_CLOSED"
    MOUTH_OPEN   = "MOUTH_OPEN"
    MOUTH_MOVING = "MOUTH_MOVING"


class MouthTracker:
    """
    MAR-based mouth tracker using MediaPipe Face Mesh 478 landmarks.

    Detects:
    - MOUTH_CLOSED  : resting / neutral mouth
    - MOUTH_OPEN    : mouth held open statically
    - MOUTH_MOVING  : dynamic speech-like movement

    Three-zone MAR classification:

        Zone 1: mar < closed_threshold                     → MOUTH_CLOSED
        Zone 2: closed_threshold ≤ mar < moving_threshold  → MOUTH_CLOSED (noise gap)
        Zone 3: moving_threshold ≤ mar < open_threshold    → MOUTH_MOVING (MAR alone)
        Zone 4: mar ≥ open_threshold + low delta           → MOUTH_OPEN
                mar ≥ open_threshold + high delta          → MOUTH_MOVING

    Delta used ONLY in Zone 4 to separate static open from active speech.
    Zone 3 uses MAR elevation alone — safe because Zone 3 MAR (0.018���0.070)
    never overlaps with rest MAR (max observed: 0.014).

    Design principles:
    - Perception layer only
    - No camera, no identity, no channel, no scoring logic
    - CPU-light, deterministic, FPS-independent
    """

    LEFT_CORNER   = 61
    RIGHT_CORNER  = 291
    TOP_CENTER    = 13
    BOTTOM_CENTER = 14
    TOP_LEFT      = 82
    BOTTOM_LEFT   = 87
    TOP_RIGHT     = 312
    BOTTOM_RIGHT  = 317

    def __init__(
        self,
        closed_threshold=0.014,   # below this → MOUTH_CLOSED
        moving_threshold=0.018,   # Zone 3 lower bound → MOUTH_MOVING, no delta needed
        open_threshold=0.070,     # Zone 4 lower bound → delta decides OPEN vs MOVING
        movement_delta=0.012,     # delta threshold — only used in Zone 4
        smoothing_window=5,
        delta_window=4,
        min_stable_frames=4,
        baseline_alpha=0.04,
        min_baseline=0.005,
        min_face_width=120,
    ):
        self.closed_threshold  = closed_threshold
        self.moving_threshold  = moving_threshold
        self.open_threshold    = open_threshold
        self.movement_delta    = movement_delta
        self.min_stable_frames = min_stable_frames
        self.baseline_alpha    = baseline_alpha
        self.min_baseline      = min_baseline
        self.min_face_width    = min_face_width

        self._mar_buffer   = deque(maxlen=smoothing_window)
        self._delta_buffer = deque(maxlen=delta_window)

        self._baseline_mar      = None
        self._last_raw_mar      = None
        self._last_smoothed_mar = None
        self._prev_smoothed_mar = None

        self._current_state   = MouthState.MOUTH_CLOSED
        self._candidate_state = None
        self._candidate_count = 0

    # ==================================================
    # PUBLIC API
    # ==================================================

    def update(self, landmarks, face_width):
        """
        Primary call. Returns MouthState.

        Args:
            landmarks  : list of 478 (x, y) tuples from LandmarkDetector
            face_width : int — face bounding box width in pixels

        Returns:
            MouthState
        """
        if face_width < self.min_face_width:
            return self._current_state

        if landmarks is None or len(landmarks) < 478:
            return self._current_state

        raw_mar = self._compute_mar(landmarks)
        if raw_mar is None:
            return self._current_state

        self._last_raw_mar = raw_mar

        self._mar_buffer.append(raw_mar)
        smoothed_mar = sum(self._mar_buffer) / len(self._mar_buffer)
        self._last_smoothed_mar = smoothed_mar

        delta = 0.0
        if self._prev_smoothed_mar is not None:
            delta = abs(smoothed_mar - self._prev_smoothed_mar)
        self._delta_buffer.append(delta)
        self._prev_smoothed_mar = smoothed_mar

        if self._current_state == MouthState.MOUTH_CLOSED:
            self._update_baseline(smoothed_mar)

        raw_state = self._classify_state(smoothed_mar)
        self._apply_stability(raw_state)

        return self._current_state

    @property
    def last_mar(self):
        return self._last_raw_mar

    @property
    def smoothed_mar(self):
        return self._last_smoothed_mar

    @property
    def state(self):
        return self._current_state

    @property
    def baseline_mar(self):
        return self._baseline_mar

    # ==================================================
    # PRIVATE — GEOMETRY
    # ==================================================

    @staticmethod
    def _euclidean(p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def _midpoint(p1, p2):
        return (
            (p1[0] + p2[0]) / 2.0,
            (p1[1] + p2[1]) / 2.0,
        )

    def _compute_mar(self, landmarks):
        """
        MAR = (A + B) / (2 * C)

        A = dist(top_center[13],  bottom_center[14])
        B = dist(midpoint(top_left[82], top_right[312]),
                 midpoint(bottom_left[87], bottom_right[317]))
        C = dist(left_corner[61], right_corner[291])
        """
        try:
            A = self._euclidean(
                landmarks[self.TOP_CENTER],
                landmarks[self.BOTTOM_CENTER]
            )
            B = self._euclidean(
                self._midpoint(landmarks[self.TOP_LEFT],
                               landmarks[self.TOP_RIGHT]),
                self._midpoint(landmarks[self.BOTTOM_LEFT],
                               landmarks[self.BOTTOM_RIGHT])
            )
            C = self._euclidean(
                landmarks[self.LEFT_CORNER],
                landmarks[self.RIGHT_CORNER]
            )
            if C == 0:
                return None
            return (A + B) / (2.0 * C)
        except (IndexError, TypeError):
            return None

    # ==================================================
    # PRIVATE — BASELINE
    # ==================================================

    def _update_baseline(self, smoothed_mar):
        if self._baseline_mar is None:
            self._baseline_mar = smoothed_mar
            return
        self._baseline_mar = (
            (1.0 - self.baseline_alpha) * self._baseline_mar
            + self.baseline_alpha * smoothed_mar
        )
        if self._baseline_mar < self.min_baseline:
            self._baseline_mar = self.min_baseline

    # ==================================================
    # PRIVATE — CLASSIFICATION
    # ==================================================

    def _classify_state(self, smoothed_mar):
        """
        Three-zone classification.

        Zone boundaries (from observed data):
            rest MAR max        : 0.014  → closed_threshold = 0.014
            soft speech MAR mean: 0.025  → moving_threshold = 0.018
            open wide MAR mean  : 0.400+ → open_threshold   = 0.070

        Baseline adaptation:
            effective_closed = max(closed_threshold, baseline + 0.004)
            effective_moving = max(moving_threshold, baseline + 0.008)

        These use fixed offsets — not ratios — for predictable, stable behaviour.
        """
        mean_delta = (
            sum(self._delta_buffer) / len(self._delta_buffer)
            if self._delta_buffer else 0.0
        )

        # Fixed-offset adaptive thresholds
        effective_closed = self.closed_threshold
        effective_moving = self.moving_threshold

        if self._baseline_mar is not None:
            effective_closed = max(
                self.closed_threshold,
                self._baseline_mar + 0.004
            )
            effective_moving = max(
                self.moving_threshold,
                self._baseline_mar + 0.008
            )

        # Zone 1 + 2: Below closed threshold → CLOSED
        if smoothed_mar < effective_closed:
            return MouthState.MOUTH_CLOSED

        # Zone 4: Above open threshold → delta decides OPEN vs MOVING
        if smoothed_mar >= self.open_threshold:
            if mean_delta < self.movement_delta:
                return MouthState.MOUTH_OPEN
            else:
                return MouthState.MOUTH_MOVING

        # Zone 3: Between closed and open threshold
        # MAR elevation alone — no delta check needed
        # Soft speech lives here (MAR 0.018–0.070)
        if smoothed_mar >= effective_moving:
            return MouthState.MOUTH_MOVING

        # Default fallback
        return MouthState.MOUTH_CLOSED

    # ==================================================
    # PRIVATE — STABILITY FILTER
    # ==================================================

    def _apply_stability(self, raw_state):
        if raw_state == self._current_state:
            self._candidate_state = None
            self._candidate_count = 0
            return

        if raw_state == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = raw_state
            self._candidate_count = 1

        if self._candidate_count >= self.min_stable_frames:
            self._current_state   = self._candidate_state
            self._candidate_state = None
            self._candidate_count = 0