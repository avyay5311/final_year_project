import numpy as np
import cv2
from enum import Enum
from collections import deque


class HeadPoseState(Enum):
    HEAD_CENTER = "HEAD_CENTER"
    HEAD_AWAY = "HEAD_AWAY"


class HeadPoseDetector:

    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),           # Nose
        (0.0, -330.0, -65.0),      # Chin
        (-225.0, 170.0, -135.0),   # Left eye
        (225.0, 170.0, -135.0),    # Right eye
        (-150.0, -150.0, -125.0),  # Left mouth
        (150.0, -150.0, -125.0)    # Right mouth
    ], dtype=np.float64)

    LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    def __init__(
        self,
        yaw_thresh=8.0,
        pitch_thresh=8.0,
        smoothing_window=4,
        min_stable_frames=3,
        hysteresis=3.0,
        min_face_width=120
    ):

        self.yaw_thresh = yaw_thresh
        self.pitch_thresh = pitch_thresh
        self.hysteresis = hysteresis
        self.min_face_width = min_face_width

        self._yaw_buffer = deque(maxlen=smoothing_window)
        self._pitch_buffer = deque(maxlen=smoothing_window)
        self._roll_buffer = deque(maxlen=smoothing_window)

        self._current_state = HeadPoseState.HEAD_CENTER
        self._candidate_state = None
        self._candidate_count = 0
        self._min_stable_frames = min_stable_frames

        self._yaw = 0.0
        self._pitch = 0.0
        self._roll = 0.0

        self._camera_matrix = None
        self._dist_coeffs = np.zeros((4, 1))

    @property
    def yaw(self):
        return self._yaw

    @property
    def pitch(self):
        return self._pitch

    @property
    def roll(self):
        return self._roll

    def _init_camera_matrix(self, w, h):
        focal_length = w
        center = (w / 2, h / 2)
        self._camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

    def _extract_points(self, landmarks):
        if landmarks is None or len(landmarks) < 468:
            return None
        pts = []
        for idx in self.LANDMARK_INDICES:
            x, y = landmarks[idx]
            pts.append([x, y])
        return np.array(pts, dtype=np.float64)

    def _compute_angles(self, points_2d, frame_shape):

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS,
            points_2d,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        # Projection-based yaw & pitch
        nose_end_3d = np.array([[0.0, 0.0, 1000.0]])
        nose_end_2d, _ = cv2.projectPoints(
            nose_end_3d,
            rvec,
            tvec,
            self._camera_matrix,
            self._dist_coeffs
        )

        nose_tip = points_2d[0]
        projected = nose_end_2d[0][0]

        dx = projected[0] - nose_tip[0]
        dy = projected[1] - nose_tip[1]

        h, w = frame_shape[:2]

        yaw = np.degrees(np.arctan2(dx, w))
        pitch = np.degrees(np.arctan2(-dy, h))

        # Roll (optional, for visualization only)
        rmat, _ = cv2.Rodrigues(rvec)
        roll = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))

        return yaw, pitch, roll

    def _smooth(self, yaw, pitch, roll):
        self._yaw_buffer.append(yaw)
        self._pitch_buffer.append(pitch)
        self._roll_buffer.append(roll)

        self._yaw = sum(self._yaw_buffer) / len(self._yaw_buffer)
        self._pitch = sum(self._pitch_buffer) / len(self._pitch_buffer)
        self._roll = sum(self._roll_buffer) / len(self._roll_buffer)

    def _classify(self):
        yaw = abs(self._yaw)
        pitch = abs(self._pitch)

        if self._current_state == HeadPoseState.HEAD_AWAY:
            yaw_thresh = self.yaw_thresh - self.hysteresis
            pitch_thresh = self.pitch_thresh - self.hysteresis
        else:
            yaw_thresh = self.yaw_thresh
            pitch_thresh = self.pitch_thresh

        if yaw < yaw_thresh and pitch < pitch_thresh:
            return HeadPoseState.HEAD_CENTER
        else:
            return HeadPoseState.HEAD_AWAY

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

    def update(self, landmarks, face_width, frame_shape):

        if face_width < self.min_face_width:
            return self._current_state

        if self._camera_matrix is None:
            h, w = frame_shape[:2]
            self._init_camera_matrix(w, h)

        points_2d = self._extract_points(landmarks)
        if points_2d is None:
            return self._current_state

        result = self._compute_angles(points_2d, frame_shape)
        if result is None:
            return self._current_state

        yaw, pitch, roll = result

        self._smooth(yaw, pitch, roll)

        raw_state = self._classify()
        return self._stability_filter(raw_state)