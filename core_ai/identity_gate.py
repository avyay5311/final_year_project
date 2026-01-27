import cv2
import numpy as np
import time
import os


class IdentityGate:
    """
    Identity verification using OpenFace embeddings.
    Returns tolerance-aware verdicts:
    - "STRONG"
    - "WEAK"
    - "MISMATCH"
    """

    def __init__(
        self,
        model_path="models/face_recognition/openface.nn4.small2.v1.t7",
        registered_face_path=None,
        strong_threshold=0.55,
        weak_threshold=0.75,
        cooldown_sec=1.0
    ):
        self.net = cv2.dnn.readNetFromTorch(model_path)

        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
        self.cooldown_sec = cooldown_sec

        self._last_check_time = 0.0
        self._last_result = None

        self.registered_embedding = None
        if registered_face_path and os.path.exists(registered_face_path):
            self.registered_embedding = np.load(registered_face_path)

    def _get_embedding(self, face_img):
        blob = cv2.dnn.blobFromImage(
            face_img,
            scalefactor=1.0 / 255.0,
            size=(96, 96),
            mean=(0, 0, 0),
            swapRB=True,
            crop=True
        )
        self.net.setInput(blob)
        return self.net.forward().flatten()

    def verify(self, frame, bbox):
        """
        Returns: "STRONG", "WEAK", or "MISMATCH"
        """
        if self.registered_embedding is None:
            return "MISMATCH"

        now = time.time()
        if now - self._last_check_time < self.cooldown_sec:
            return self._last_result

        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            self._last_result = "MISMATCH"
            self._last_check_time = now
            return self._last_result

        face = cv2.resize(face, (96, 96))
        embedding = self._get_embedding(face)

        dist = np.linalg.norm(embedding - self.registered_embedding)

        # -----------------------------
        # Tolerance-based decision
        # -----------------------------
        if dist < self.strong_threshold:
            verdict = "STRONG"
        elif dist < self.weak_threshold:
            verdict = "WEAK"
        else:
            verdict = "MISMATCH"

        self._last_result = verdict
        self._last_check_time = now
        return verdict
