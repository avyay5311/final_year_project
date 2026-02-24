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
from core_ai.gaze_tracking import GazeTracking

from identity.calibration_prompt import run_initial_calibration


# -------------------------------------------------
# Configuration
# -------------------------------------------------
MIN_FACE_WIDTH = 120


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
    strong_threshold=0.71,
    weak_threshold=0.77,
    cooldown_sec=1.5
)

identity_sm = PrimaryIdentityStateMachine(identity_gate)

landmark_detector = LandmarkDetector()

blink_detector = BlinkDetector(
    ear_threshold=0.21,
    consec_frames=2
)

# Balanced gaze detector
gaze = GazeTracking(
    smoothing_window=5,
    min_stable_frames=3,
    min_face_width=MIN_FACE_WIDTH,
    horizontal_thresh=0.06,
    vertical_thresh=0.06,
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not opened")
    raise SystemExit


# -------------------------------------------------
# Identity-Gated Calibration
# -------------------------------------------------

calibrated = run_initial_calibration(
    cap=cap,
    gaze=gaze,
    face_source=face_source,
    tracker=tracker,
    identity_sm=identity_sm,
    landmark_detector=landmark_detector,
    suppress_duplicates=suppress_duplicates,
    window_name="Primary Identity + Gaze Test",
)

if not calibrated:
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit


# -------------------------------------------------
# Main loop
# -------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection + tracking
    detections = face_source.get_faces(frame)
    face_states = tracker.update(detections)
    face_states = suppress_duplicates(face_states)

    identity_state, primary_face = identity_sm.update(frame, face_states)

    gaze_label = "NO_PUPILS"
    blink_state = BlinkState.NO_BLINK

    # -------------------------------------------------
    # Run detection only when identity confirmed
    # -------------------------------------------------

    if primary_face is not None and identity_state in (
        IdentityState.PRIMARY_CONFIRMED,
        IdentityState.PRIMARY_REACQUIRED,
    ):

        landmarks = landmark_detector.detect(frame, primary_face["bbox"])

        if landmarks is not None:
            x, y, w, h = primary_face["bbox"]

            # Gaze
            gaze.refresh(frame, landmarks, face_width=w)
            gaze_label = gaze.stable_state()

            # Blink (distance-aware)
            if w >= MIN_FACE_WIDTH:
                ear = blink_detector.compute_ear(landmarks)
                if ear is not None:
                    blink_state = blink_detector.update(landmarks)

    # -------------------------------------------------
    # Status Mapping
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

    cv2.putText(frame, f"STATUS: {status}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"GAZE: {gaze_label}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    if blink_state == BlinkState.BLINK:
        cv2.putText(frame, "BLINK", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Primary Identity + Gaze Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()