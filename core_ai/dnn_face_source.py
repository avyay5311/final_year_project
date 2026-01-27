# core_ai/dnn_face_source.py

import cv2
import numpy as np
from core_ai.face_source import FaceSource
from core_ai.config import FACE_CONFIDENCE_THRESHOLD


class DNNFaceSource(FaceSource):
    def __init__(self, proto_path, model_path):
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    def get_faces(self, frame):
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])

            if confidence < FACE_CONFIDENCE_THRESHOLD:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            faces.append({
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "confidence": confidence
            })

        return faces
