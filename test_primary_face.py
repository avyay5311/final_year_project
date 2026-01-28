# test_primary_face.py

import cv2

from core_ai.dnn_face_source import DNNFaceSource
from core_ai.face_tracker import FaceTracker
from core_ai.duplicate_suppression import suppress_duplicates
from core_ai.identity_gate import IdentityGate
from core_ai.primary_identity_sm import (
    PrimaryIdentityStateMachine,
    IdentityState,
)
from core_ai.landmark_detector import LandmarkDetector
from core_ai.blink_detector import BlinkDetector, BlinkState


# -------------------------------------------------
# Initialize components
# -------------------------------------------------

# Face detector (OpenCV DNN)
face_source = DNNFaceSource(
    "models/dnn_face/deploy.prototxt",
    "models/dnn_face/res10_300x300_ssd_iter_140000.caffemodel"
)

# Tracker
tracker = FaceTracker()

# Identity verification
identity_gate = IdentityGate(
    registered_face_path="identity/registered_face.npy",
    strong_threshold=0.55,
    weak_threshold=0.75,
    cooldown_sec=1.0
)

identity_sm = PrimaryIdentityStateMachine(identity_gate)

# MediaPipe landmark detector
landmark_detector = LandmarkDetector()

# Blink detector (MediaPipe-aware)
blink_detector = BlinkDetector(
    ear_threshold=0.21,
    consec_frames=2
)

# Camera
cap = cv2.VideoCapture(0)


# MediaPipe eye landmark indices (for debug draw)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# -------------------------------------------------
# Main loop
# -------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Face detection
    detections = face_source.get_faces(frame)

    # 2️⃣ Face tracking
    face_states = tracker.update(detections)

    # 3️⃣ Duplicate suppression
    face_states = suppress_duplicates(face_states)

    # 4️⃣ Identity State Machine
    identity_state, primary_face = identity_sm.update(frame, face_states)

    landmarks = None
    ear = None
    blink_state = BlinkState.NO_BLINK

    # -------------------------------------------------
    # Landmark + Blink detection (PRIMARY FACE ONLY)
    # -------------------------------------------------
    if primary_face is not None and identity_state in (
        IdentityState.PRIMARY_CONFIRMED,
        IdentityState.PRIMARY_REACQUIRED,
    ):
        landmarks = landmark_detector.detect(frame, primary_face["bbox"])

        if landmarks is not None:
            ear = blink_detector.compute_ear(landmarks)

            if ear is not None:
                print(f"EAR: {ear:.3f}")

                blink_state = blink_detector.update(landmarks)
                if blink_state == BlinkState.BLINK:
                    print("👁️ Blink detected!")

    # -------------------------------------------------
    # Status mapping
    # -------------------------------------------------
    if identity_state in (
        IdentityState.PRIMARY_CONFIRMED,
        IdentityState.PRIMARY_REACQUIRED,
    ):
        status = "PRIMARY_PRESENT"

    elif identity_state == IdentityState.PRIMARY_TEMP_ABSENT:
        status = "NO_FACE"

    elif identity_state == IdentityState.PRIMARY_TEMP_UNCERTAIN:
        status = "PRIMARY_UNCERTAIN"

    elif identity_state == IdentityState.IMPERSONATION_SUSPECT:
        status = "IMPERSONATION_SUSPECT"

    else:
        status = identity_state.value

    # -------------------------------------------------
    # Visualization
    # -------------------------------------------------

    # Draw face boxes
    for face in face_states:
        x, y, w, h = face["bbox"]
        fid = face["id"]

        color = (0, 255, 0)

        # Primary face → BLUE
        if (
            primary_face
            and fid == primary_face["id"]
            and identity_state in (
                IdentityState.PRIMARY_CONFIRMED,
                IdentityState.PRIMARY_REACQUIRED,
            )
        ):
            color = (255, 0, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"ID:{fid}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    # Draw eye landmarks (debug)
    if landmarks is not None:
        for i in LEFT_EYE + RIGHT_EYE:
            lx, ly = landmarks[i]
            cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)

    # Blink label
    if blink_state == BlinkState.BLINK:
        cv2.putText(
            frame,
            "BLINK",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )

    # Status text
    cv2.putText(
        frame,
        f"STATUS: {status}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("Primary Identity + MediaPipe Blink Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
