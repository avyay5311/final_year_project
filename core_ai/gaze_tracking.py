from __future__ import division
from collections import deque
import time
import cv2
import numpy as np

from .eye import Eye
from .calibration import Calibration


class GazeTracking(object):

    def __init__(
        self,
        smoothing_window=4,
        min_stable_frames=3,
        min_face_width=120,
        horizontal_thresh=0.05,
        vertical_thresh=0.01,   # Updated default
    ):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        self.min_face_width = min_face_width

        # Smoothing buffers
        self._hr_history = deque(maxlen=smoothing_window)
        self._vr_history = deque(maxlen=smoothing_window)

        # Stability
        self._current_state = "LOOKING_CENTER"
        self._candidate_state = None
        self._candidate_count = 0
        self._min_stable_frames = min_stable_frames

        # Direction thresholds
        self._h_thresh = horizontal_thresh
        self._v_thresh = vertical_thresh

        # Calibration
        self._calibrating = False
        self._calib_start = 0.0
        self._calib_duration = 1.0
        self._calib_samples = []

        self.center_hr = 0.5
        self.center_vr = 0.5
        self._calibrated = False

    # -------------------------
    # PROPERTIES
    # -------------------------

    @property
    def pupils_located(self):
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    @property
    def is_calibrating(self):
        return self._calibrating

    @property
    def calibration_progress(self):
        if not self._calibrating:
            return 0.0
        return min(1.0, (time.time() - self._calib_start) / self._calib_duration)

    # -------------------------
    # CALIBRATION
    # -------------------------

    def start_calibration(self, duration_sec=1.0):
        self._calibrating = True
        self._calib_start = time.time()
        self._calib_duration = duration_sec
        self._calib_samples = []

    # -------------------------
    # CORE ANALYSIS
    # -------------------------

    def _analyze(self, landmarks):
        if self.frame is None or landmarks is None:
            self.eye_left = None
            self.eye_right = None
            return

        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        try:
            self.eye_left = Eye(gray, landmarks, side=0, calibration=self.calibration)
            self.eye_right = Eye(gray, landmarks, side=1, calibration=self.calibration)
        except Exception:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame, landmarks, face_width):

        if face_width < self.min_face_width:
            return

        self.frame = frame
        self._analyze(landmarks)
        self._update_ratios()

    # -------------------------
    # RATIO LOGIC
    # -------------------------

    def _raw_ratios(self):
        if not self.pupils_located:
            return None, None

        # Horizontal
        hr_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2)
        hr_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2)
        hr = (hr_left + hr_right) / 2

        # Vertical
        vr_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2)
        vr_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2)
        vr = (vr_left + vr_right) / 2

        return hr, vr

    def _update_ratios(self):
        hr, vr = self._raw_ratios()

        if hr is None:
            return

        self._hr_history.append(hr)
        self._vr_history.append(vr)

        # Calibration
        if self._calibrating:
            if abs(hr - 0.5) < 0.08 and abs(vr - 0.5) < 0.08:
                self._calib_samples.append((hr, vr))

            if (time.time() - self._calib_start) >= self._calib_duration:
                if len(self._calib_samples) >= 10:
                    hrs = [s[0] for s in self._calib_samples]
                    vrs = [s[1] for s in self._calib_samples]
                    self.center_hr = sum(hrs) / len(hrs)
                    self.center_vr = sum(vrs) / len(vrs)
                    self._calibrated = True
                self._calibrating = False

    def horizontal_ratio(self):
        if not self._hr_history:
            return None
        return sum(self._hr_history) / len(self._hr_history)

    def vertical_ratio(self):
        if not self._vr_history:
            return None
        return sum(self._vr_history) / len(self._vr_history)

    # -------------------------
    # STABLE STATE
    # -------------------------

    def stable_state(self):

        if not self.pupils_located or not self._calibrated:
            return "NO_PUPILS"

        hr = self.horizontal_ratio()
        vr = self.vertical_ratio()

        if hr is None:
            return self._current_state

        dx = hr - self.center_hr
        dy = vr - self.center_vr

        # -----------------------------
        # Binary Center vs Away with Hysteresis
        # -----------------------------

        ENTER_H = 0.035   # must exceed to enter AWAY
        ENTER_V = 0.02

        EXIT_H = 0.025     # must fall below to return to CENTER
        EXIT_V = 0.015

        if self._current_state == "LOOKING_AWAY":
            # Stay AWAY until clearly inside center zone
            if abs(dx) < EXIT_H and abs(dy) < EXIT_V:
                new_state = "LOOKING_CENTER"
            else:
                new_state = "LOOKING_AWAY"
        else:
            # Currently CENTER
            if abs(dx) > ENTER_H or abs(dy) > ENTER_V:
                new_state = "LOOKING_AWAY"
            else:
                new_state = "LOOKING_CENTER"

        # -----------------------------
        # Stability Filtering
        # -----------------------------

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

        print(f"dx: {dx:.4f}, dy: {dy:.4f}, STATE: {self._current_state}")

        return self._current_state
