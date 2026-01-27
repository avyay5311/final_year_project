# identity/register_candidate.py

import cv2
import numpy as np
import os

from core_ai.dnn_face_source import DNNFaceSource
from core_ai.identity_gate import IdentityGate

os.makedirs("identity", exist_ok=True)

# SAME detector as runtime
face_source = DNNFaceSource(
    "models/dnn_face/deploy.prototxt",
    "models/dnn_face/res10_300x300_ssd_iter_140000.caffemodel"
)

# Load embedding model only (no registered face yet)
gate = IdentityGate(
    model_path="models/face_recognition/openface.nn4.small2.v1.t7"
)

cap = cv2.VideoCapture(0)
print("Press 'c' to register face (exactly ONE face). Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = face_source.get_faces(frame)

    # Draw detected faces
    for det in detections:
        x, y, w, h = det["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Register Candidate", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        if len(detections) != 1:
            print("❌ Ensure exactly ONE face is visible.")
            continue

        det = detections[0]
        x, y, w, h = det["bbox"]

        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            print("❌ Invalid face crop.")
            continue

        face = cv2.resize(face, (96, 96))
        embedding = gate._get_embedding(face)

        np.save("identity/registered_face.npy", embedding)
        print("✅ Candidate face registered successfully.")
        break

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
