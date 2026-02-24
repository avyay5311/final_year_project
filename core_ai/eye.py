import math
import numpy as np
import cv2
from types import SimpleNamespace
from .pupil import Pupil


class Eye(object):
    """
    Isolates an eye region and initializes pupil detection.
    Ported for MediaPipe FaceMesh landmarks (pixel-space list[(x,y)]).
    """

    # Eye contour points (FaceMesh)
    LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_POINTS = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

    # Iris points (FaceMesh with refine_landmarks=True)
    LEFT_IRIS_POINTS = [474, 475, 476, 477]
    RIGHT_IRIS_POINTS = [469, 470, 471, 472]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None
        self.blinking = None

        self._analyze(original_frame, landmarks, side, calibration)

    def _isolate(self, frame, landmarks, points):
        region = np.array([landmarks[i] for i in points], dtype=np.int32)
        self.landmark_points = region

        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)

        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        margin = 5
        min_x = int(np.min(region[:, 0]) - margin)
        max_x = int(np.max(region[:, 0]) + margin)
        min_y = int(np.min(region[:, 1]) - margin)
        max_y = int(np.max(region[:, 1]) + margin)

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(width - 1, max_x)
        max_y = min(height - 1, max_y)

        if max_x <= min_x or max_y <= min_y:
            self.frame = None
            self.origin = (0, 0)
            self.center = (1, 1)
            return

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        h2, w2 = self.frame.shape[:2]
        self.center = (w2 / 2, h2 / 2)

    def _blinking_ratio(self, landmarks, points):
        p_left = landmarks[points[0]]
        p_right = landmarks[points[8]]
        p_top = landmarks[points[12]]
        p_bottom = landmarks[points[4]]

        eye_width = math.hypot(p_left[0] - p_right[0], p_left[1] - p_right[1])
        eye_height = math.hypot(p_top[0] - p_bottom[0], p_top[1] - p_bottom[1])

        try:
            return eye_width / eye_height
        except ZeroDivisionError:
            return None

    def _iris_center(self, landmarks, side):
        iris_points = self.LEFT_IRIS_POINTS if side == 0 else self.RIGHT_IRIS_POINTS

        # If landmarks list is too short (refine_landmarks=False), skip
        if max(iris_points) >= len(landmarks):
            return None

        pts = np.array([landmarks[i] for i in iris_points], dtype=np.float32)
        cx, cy = np.mean(pts, axis=0)
        return (int(cx), int(cy))

    def _analyze(self, original_frame, landmarks, side, calibration):
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if self.frame is None or self.frame.size == 0:
            self.pupil = Pupil(np.zeros((10, 10), dtype=np.uint8), 50)
            return

        # Prefer iris landmarks if available
        iris_center = self._iris_center(landmarks, side)
        if iris_center is not None:
            px = iris_center[0] - self.origin[0]
            py = iris_center[1] - self.origin[1]
            self.pupil = SimpleNamespace(x=int(px), y=int(py))
            return

        # Fallback to threshold-based pupil detection
        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)