import cv2
import mediapipe as mp


class LandmarkDetector:
    """
    MediaPipe-based facial landmark detector.
    Returns pixel-space landmarks for the given face bbox.
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # iris landmarks enabled
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame, bbox):
        """
        frame: BGR image
        bbox: (x, y, w, h)
        returns: list[(x, y)] of 478 landmarks in image coords
        """

        x, y, w, h = bbox
        h_img, w_img = frame.shape[:2]

        # Expand bbox slightly for better landmark quality
        pad_x = int(0.25 * w)
        pad_y = int(0.35 * h)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            px = int(lm.x * (x2 - x1)) + x1
            py = int(lm.y * (y2 - y1)) + y1
            landmarks.append((px, py))

        return landmarks