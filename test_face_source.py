# test_face_source.py

import cv2
from core_ai.dnn_face_source import DNNFaceSource

face_source = DNNFaceSource(
    "models/dnn_face/deploy.prototxt",
    "models/dnn_face/res10_300x300_ssd_iter_140000.caffemodel"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = face_source.get_faces(frame)

    for f in faces:
        x, y, w, h = f["bbox"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("FaceSource Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
