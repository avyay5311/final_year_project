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

# -------------------------------------------------
# Initialize components
# -------------------------------------------------
face_source = DNNFaceSource(
    "models/dnn_face/deploy.prototxt",
    "models/dnn_face/res10_300x300_ssd_iter_140000.caffemodel"
)

tracker = FaceTracker()

identity_gate = IdentityGate(
    registered_face_path="identity/registered_face.npy",
    strong_threshold=0.55,
    weak_threshold=0.75,
    cooldown_sec=1.0
)


identity_sm = PrimaryIdentityStateMachine(identity_gate)

cap = cv2.VideoCapture(0)

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

    # 3️⃣ Duplicate suppression (important for fast motion)
    face_states = suppress_duplicates(face_states)

    # 4️⃣ Identity State Machine (ONLY authority)
    identity_state, primary_face = identity_sm.update(frame, face_states)

    # -------------------------------------------------
    # Status mapping (NO extra logic)
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
    for face in face_states:
        x, y, w, h = face["bbox"]
        fid = face["id"]

        color = (0, 255, 0)  # default GREEN

        # Verified / Reacquired primary → BLUE
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

    cv2.imshow("Primary Identity Test (Recovery Enabled)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
