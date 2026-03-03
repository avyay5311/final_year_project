import cv2
import json
import os
import time
import uuid

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
from core_ai.mouth_tracker import MouthTracker, MouthState
from identity.calibration_prompt import run_initial_calibration
from channels.gaze_channel import GazeChannel
from channels.blink_channel import BlinkChannel
from channels.headpose_channel import HeadPoseChannel
from channels.mouth_channel import MouthChannel
from channels.face_identity_channel import FaceIdentityChannel
from integrity_engine import IntegrityEngine

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

mouth_tracker = MouthTracker(
    closed_threshold=0.014,
    moving_threshold=0.018,
    open_threshold=0.070,
    movement_delta=0.012,
    smoothing_window=5,
    delta_window=4,
    min_stable_frames=4,
    min_face_width=MIN_FACE_WIDTH,
)

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

gaze_channel         = GazeChannel()
blink_channel        = BlinkChannel()
headpose_channel     = HeadPoseChannel()
mouth_channel        = MouthChannel(
    sustained_threshold=3.0,
    rapid_threshold=1.5,
    burst_window=15.0,
    burst_min_events=3,
)
face_identity_channel = FaceIdentityChannel()

headpose_detector = HeadPoseDetector(
    yaw_thresh=8.0,
    pitch_thresh=8.0,
    hysteresis=1.0,
    smoothing_window=4,
    min_stable_frames=3,
    min_face_width=MIN_FACE_WIDTH
)

# -------------------------------------------------
# Generate a unique session ID for this run
# -------------------------------------------------
session_id = f"session_{uuid.uuid4().hex[:8]}"

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
    window_name="Primary Identity + Gaze + HeadPose Test",
)

if not calibrated:
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

# -------------------------------------------------
# Session timer starts after calibration
# -------------------------------------------------
session_start = time.time()
print(f"\nSession started  →  ID: {session_id}\n")

# -------------------------------------------------
# Main Loop
# -------------------------------------------------

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections  = face_source.get_faces(frame)
        face_states = tracker.update(detections)
        face_states = suppress_duplicates(face_states)

        identity_state, primary_face = identity_sm.update(
            frame, face_states
        )

        # -------------------------------------------------
        # FaceIdentityChannel — updated every frame
        # directly from identity_state
        # no identity_valid gate needed here
        # -------------------------------------------------
        face_identity_channel.update(identity_state)

        # Default states
        gaze_label     = "NO_PUPILS"
        blink_state    = BlinkState.NO_BLINK
        headpose_state = HeadPoseState.HEAD_CENTER
        mouth_state    = MouthState.MOUTH_CLOSED

        # -------------------------------------------------
        # Run detection only when identity is behavior-valid
        # -------------------------------------------------

        if (
            primary_face is not None
            and identity_state not in (
                IdentityState.PRIMARY_TEMP_ABSENT,
                IdentityState.IMPERSONATION_SUSPECT,
                IdentityState.MULTIPLE_FACES_PRESENT,
            )
        ):
            landmarks = landmark_detector.detect(
                frame, primary_face["bbox"]
            )

            if landmarks is not None:
                x, y, w, h = primary_face["bbox"]

                # Gaze
                gaze.refresh(frame, landmarks, face_width=w)
                gaze_label = gaze.stable_state()

                # Blink
                if w >= MIN_FACE_WIDTH:
                    blink_state = blink_detector.update(landmarks)

                # Head Pose
                headpose_state = headpose_detector.update(
                    landmarks=landmarks,
                    face_width=w,
                    frame_shape=frame.shape
                )

                # Mouth
                mouth_state = mouth_tracker.update(
                    landmarks=landmarks,
                    face_width=w
                )

        # -------------------------------------------------
        # Identity validity for behavior channels
        # -------------------------------------------------

        identity_valid_for_behavior = (
            primary_face is not None
            and identity_state not in (
                IdentityState.PRIMARY_TEMP_ABSENT,
                IdentityState.MULTIPLE_FACES_PRESENT,
                IdentityState.IMPERSONATION_SUSPECT,
            )
        )

        # -------------------------------------------------
        # Update behavior channels
        # -------------------------------------------------

        gaze_channel.update(
            gaze_state=gaze_label,
            identity_valid=identity_valid_for_behavior
        )

        blink_channel.update(
            blink_state=blink_state,
            identity_valid=identity_valid_for_behavior
        )

        headpose_channel.update(
            headpose_state=headpose_state,
            identity_valid=identity_valid_for_behavior
        )

        mouth_channel.update(
            mouth_state=mouth_state,
            identity_valid=identity_valid_for_behavior
        )

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
        elif identity_state == IdentityState.MULTIPLE_FACES_PRESENT:
            status = "MULTIPLE_FACES"
        else:
            status = identity_state.value

        # -------------------------------------------------
        # Visualization
        # -------------------------------------------------

        cv2.putText(frame, f"STATUS: {status}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"GAZE: {gaze_label}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2)

        if blink_state == BlinkState.BLINK:
            cv2.putText(frame, "BLINK",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"HEAD: {headpose_state.value}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0)
                    if headpose_state == HeadPoseState.HEAD_CENTER
                    else (0, 0, 255),
                    2)

        cv2.putText(frame, f"Yaw: {headpose_detector.yaw:+.2f}",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"Pitch: {headpose_detector.pitch:+.2f}",
                    (20, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        mouth_color = {
            MouthState.MOUTH_CLOSED: (0,   200,   0),
            MouthState.MOUTH_OPEN:   (0,   200, 255),
            MouthState.MOUTH_MOVING: (0,    60, 255),
        }.get(mouth_state, (200, 200, 200))

        cv2.putText(frame, f"MOUTH: {mouth_state.value}",
                    (20, 210), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, mouth_color, 2)

        cv2.imshow("Primary Identity + Gaze + HeadPose Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nSession interrupted by user (Ctrl+C).")

finally:

    # -------------------------------------------------
    # STEP 1 — Compute session duration
    # -------------------------------------------------
    session_duration = time.time() - session_start

    print(f"\n{'='*55}")
    print(f"  SESSION COMPLETE")
    print(f"  ID       : {session_id}")
    print(f"  Duration : {session_duration:.1f} seconds")
    print(f"{'='*55}\n")

    # -------------------------------------------------
    # STEP 2 — Finalize all channels
    # -------------------------------------------------
    print("Finalizing behavior channels...")

    gaze_channel.finalize()
    blink_channel.finalize()
    headpose_channel.finalize()
    mouth_channel.finalize()
    face_identity_channel.finalize()

    # -------------------------------------------------
    # STEP 3 — Get summaries
    # -------------------------------------------------
    gaze_summary          = gaze_channel.get_summary()
    blink_summary         = blink_channel.get_summary()
    headpose_summary      = headpose_channel.get_summary()
    mouth_summary         = mouth_channel.get_summary()
    face_identity_summary = face_identity_channel.get_summary()

    # -------------------------------------------------
    # STEP 4 — Save raw summaries to disk
    # -------------------------------------------------
    os.makedirs("summaries", exist_ok=True)

    with open("summaries/gaze_summary.json", "w") as f:
        json.dump(gaze_summary, f, indent=4)

    with open("summaries/blink_summary.json", "w") as f:
        json.dump(blink_summary, f, indent=4)

    with open("summaries/headpose_summary.json", "w") as f:
        json.dump(headpose_summary, f, indent=4)

    with open("summaries/mouth_summary.json", "w") as f:
        json.dump(mouth_summary, f, indent=4)

    with open("summaries/face_identity_summary.json", "w") as f:
        json.dump(face_identity_summary, f, indent=4)

    print("Raw summaries saved:")
    print("  summaries/gaze_summary.json")
    print("  summaries/blink_summary.json")
    print("  summaries/headpose_summary.json")
    print("  summaries/mouth_summary.json")
    print("  summaries/face_identity_summary.json")

    # -------------------------------------------------
    # STEP 5 — Run IntegrityEngine
    # -------------------------------------------------
    print("\nRunning IntegrityEngine...")

    engine = IntegrityEngine()
    report = engine.run(
        session_id            = session_id,
        session_duration      = session_duration,   # ← added
        gaze_summary          = gaze_summary,
        headpose_summary      = headpose_summary,
        blink_summary         = blink_summary,
        mouth_summary         = mouth_summary,
        face_identity_summary = face_identity_summary,
    )

    # -------------------------------------------------
    # STEP 6 — Save integrity report to disk
    # -------------------------------------------------
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/{session_id}_report.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    # -------------------------------------------------
    # STEP 7 — Print final result to terminal
    # -------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  INTEGRITY REPORT")
    print(f"{'='*55}")
    print(f"  Session ID      : {report['session_id']}")
    print(f"  Duration        : {report['session_duration']}s")
    print(f"  Integrity Score : {report['integrity_score']}")
    print(f"  Risk Level      : {report['risk_level']}")
    print(f"  Hard Cap        : {report['hard_cap_applied']}")
    print(f"  Total Flags     : {report['flag_summary']['total_flags']}")
    print(f"  By Severity     : {report['flag_summary']['by_severity']}")
    print(f"{'='*55}")
    print(f"\n  Channel Penalties:")
    for ch, val in report["channel_penalties"].items():
        print(f"    {ch:<20} {val:.4f}")
    print(f"\n  Top Flags:")
    for flag in report["flags"][:5]:
        ts  = f"t={flag['timestamp']}" if flag["timestamp"] else "t=N/A"
        dur = f"dur={flag['duration']}s" if flag["duration"] else ""
        print(f"    [{flag['severity']:<8}] {flag['type']}  {ts}  {dur}")
    print(f"\n  Full report saved → {report_path}")
    print(f"{'='*55}\n")

    cap.release()
    cv2.destroyAllWindows()