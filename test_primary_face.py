import cv2
import json
import os

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
from core_ai.headpose_detector import HeadPoseDetector, HeadPoseState

from identity.calibration_prompt import run_initial_calibration
from channels.gaze_channel import GazeChannel
from channels.blink_channel import BlinkChannel
from channels.headpose_channel import HeadPoseChannel
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

gaze = GazeTracking(
    smoothing_window=5,
    min_stable_frames=3,
    min_face_width=MIN_FACE_WIDTH,
    horizontal_thresh=0.06,
    vertical_thresh=0.06,
)

gaze_channel = GazeChannel()  # ✅ Behavior Channel
blink_channel = BlinkChannel()
headpose_channel = HeadPoseChannel()

headpose_detector = HeadPoseDetector(
    yaw_thresh=8.0,
    pitch_thresh=8.0,
    hysteresis=2.0,
    smoothing_window=4,
    min_stable_frames=3,
    min_face_width=MIN_FACE_WIDTH
)

cap = cv2.VideoCapture(0)
# print("Frame width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print("Frame height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
    window_name="Primary Identity + Gaze + HeadPose Test",
)

if not calibrated:
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit


# -------------------------------------------------
# Main Loop (Graceful Shutdown Enabled)
# -------------------------------------------------

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = face_source.get_faces(frame)
        face_states = tracker.update(detections)
        face_states = suppress_duplicates(face_states)

        identity_state, primary_face = identity_sm.update(frame, face_states)

        gaze_label = "NO_PUPILS"
        blink_state = BlinkState.NO_BLINK
        headpose_state = HeadPoseState.HEAD_CENTER

        # -------------------------------------------------
        # Run detection when behavior-valid identity
        # -------------------------------------------------

        if (
            primary_face is not None
            and identity_state not in (
                IdentityState.PRIMARY_TEMP_ABSENT,
                IdentityState.IMPERSONATION_SUSPECT,
                IdentityState.MULTIPLE_FACES_PRESENT,
            )
        ):

            landmarks = landmark_detector.detect(frame, primary_face["bbox"])

            if landmarks is not None:
                x, y, w, h = primary_face["bbox"]

                # -------------------------
                # Gaze
                # -------------------------
                gaze.refresh(frame, landmarks, face_width=w)
                gaze_label = gaze.stable_state()

                # -------------------------
                # Blink
                # -------------------------
                if w >= MIN_FACE_WIDTH:
                    ear = blink_detector.compute_ear(landmarks)
                    if ear is not None:
                        blink_state = blink_detector.update(landmarks)

                # -------------------------
                # Head Pose
                # -------------------------
                headpose_state = headpose_detector.update(
                    landmarks=landmarks,
                    face_width=w,
                    frame_shape=frame.shape
                )

        # -------------------------------------------------
        # Identity validity for behavior channel
        # -------------------------------------------------

        identity_valid_for_behavior = (
            primary_face is not None
            and identity_state not in (
                IdentityState.PRIMARY_TEMP_ABSENT,
                IdentityState.MULTIPLE_FACES_PRESENT,
                IdentityState.IMPERSONATION_SUSPECT,
            )
        )

        # ✅ Update Gaze Channel Every Frame
        gaze_channel.update(
            gaze_state=gaze_label,
            identity_valid=identity_valid_for_behavior
        )

        #update blink channel every frame as well, with same identity validity check
        blink_channel.update(
            blink_state=blink_state,
            identity_valid=identity_valid_for_behavior
        )


        # -------------------------------------------------
        # Status Mapping (Preserved)
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
        # Visualization (ALL PRESERVED)
        # -------------------------------------------------

        cv2.putText(frame, f"STATUS: {status}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"GAZE: {gaze_label}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if blink_state == BlinkState.BLINK:
            cv2.putText(frame, "BLINK", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"HEAD: {headpose_state.value}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if headpose_state == HeadPoseState.HEAD_CENTER else (0, 0, 255),
                    2)

        cv2.putText(frame, f"Yaw: {headpose_detector.yaw:+.2f}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"Pitch: {headpose_detector.pitch:+.2f}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Primary Identity + Gaze + HeadPose Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user (Ctrl+C).")

finally:
    print("Finalizing gaze behavior summary...")

    gaze_channel.finalize()

    blink_channel.finalize()
    headpose_channel.finalize()
    summary = gaze_channel.get_summary()

    os.makedirs("summaries", exist_ok=True)

    with open("summaries/gaze_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    blink_summary = blink_channel.get_summary()
    with open("summaries/blink_summary.json", "w") as f:
        json.dump(blink_summary, f, indent=4)
    
    headpose_summary = headpose_channel.get_summary()
    with open("summaries/headpose_summary.json", "w") as f:
        json.dump(headpose_summary, f, indent=4)
    
    cap.release()
    cv2.destroyAllWindows()

    print("Gaze summary saved to summaries/gaze_summary.json")
    print("Blink summary saved to summaries/blink_summary.json")
    print("Headpose summary saved to summaries/headpose_summary.json")