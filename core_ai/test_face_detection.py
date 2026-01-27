import cv2
from face_detection import detect_faces
from face_tracker import StableFaceTracker

cap = cv2.VideoCapture(0)
tracker = StableFaceTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    _, detected_faces = detect_faces(
        frame,
        draw_box=False,
        draw_landmarks=False
    )

    tracked_faces = tracker.update(detected_faces)

    for face_id, face in tracked_faces.items():
        x, y, w, h = face["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {face_id}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    dominant = tracker.get_dominant_face()
    if dominant:
        fid, _ = dominant
        cv2.putText(
            frame,
            f"Dominant Face: ID {fid}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Stable Face Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
