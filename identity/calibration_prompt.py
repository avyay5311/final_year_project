import cv2
from core_ai.primary_identity_sm import IdentityState


def draw_calibration_prompt(frame, message):
    prompt = frame.copy()
    cv2.putText(
        prompt,
        message,
        (20, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        prompt,
        "Press 'c' to calibrate",
        (20, 80),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        prompt,
        "Press 'q' to quit",
        (20, 120),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    return prompt


def run_initial_calibration(
    cap,
    gaze,
    face_source,
    tracker,
    identity_sm,
    landmark_detector,
    suppress_duplicates,
    window_name="Proctoring System",
):

    while True:
        ret, frame = cap.read()
        if not ret:
            return False

        # --- Identity Pipeline ---
        detections = face_source.get_faces(frame)
        face_states = tracker.update(detections)
        face_states = suppress_duplicates(face_states)
        identity_state, primary_face = identity_sm.update(frame, face_states)

        # --- Only allow calibration when identity confirmed ---
        if identity_state != IdentityState.PRIMARY_CONFIRMED:
            view = draw_calibration_prompt(
                frame,
                "Face not verified. Confirm identity first."
            )
            cv2.imshow(window_name, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False

            continue

        # --- Show calibration ready screen ---
        view = draw_calibration_prompt(
            frame,
            "Keep eyes steady and look straight"
        )
        cv2.imshow(window_name, view)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):

            gaze.start_calibration(duration_sec=1.0)

            # --- Calibration Loop ---
            while gaze.is_calibrating:
                ok, frame = cap.read()
                if not ok:
                    return False

                # Force time check even if no face
                if gaze.calibration_progress >= 1.0:
                    gaze._calibrating = False
                    break

                ok, frame = cap.read()
                if not ok:
                    return False

                detections = face_source.get_faces(frame)
                face_states = tracker.update(detections)
                face_states = suppress_duplicates(face_states)
                identity_state, primary_face = identity_sm.update(frame, face_states)

                if primary_face is None:
                    continue

                landmarks = landmark_detector.detect(frame, primary_face["bbox"])
                if landmarks is None:
                    continue

                x, y, w, h = primary_face["bbox"]
                gaze.refresh(frame, landmarks, face_width=w)

                progress = int(gaze.calibration_progress * 100)

                cv2.putText(
                    frame,
                    "Keep eyes steady and look straight",
                    (20, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"CALIBRATING... {progress}%",
                    (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )

                cv2.imshow(window_name, frame)
                cv2.waitKey(1)

            # --- After Calibration Ends ---
            if gaze._calibrated:
                return True

            # --- If calibration failed ---
            failure_view = frame.copy()
            cv2.putText(
                failure_view,
                "Calibration unstable. Press C to retry.",
                (20, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            cv2.imshow(window_name, failure_view)
            cv2.waitKey(1500)

            # IMPORTANT: Do NOT return True
            # Go back to top of loop and wait for 'c' again
            continue

        if key == ord("q"):
            return False