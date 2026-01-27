# test_face_tracker.py

import cv2
from core_ai.dnn_face_source import DNNFaceSource
from core_ai.face_tracker import FaceTracker

face_source = DNNFaceSource(
    "models/dnn_face/deploy.prototxt",
    "models/dnn_face/res10_300x300_ssd_iter_140000.caffemodel"
)

tracker = FaceTracker()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    detections = face_source.get_faces(frame)
    face_states = tracker.update(detections)
    # for face in face_states:
    #     print(face) 
    for face in face_states:
        x, y, w, h = face["bbox"]
        fid = face["id"]
        vx, vy = face["velocity"]

        color = (0, 255, 0) if face["visible"] else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            f"ID:{fid} v=({vx},{vy})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    cv2.imshow("FaceTracker Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()