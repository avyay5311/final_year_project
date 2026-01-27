import cv2
import dlib
from imutils import face_utils
import os

# -------------------------------
# Load models ONCE (important)
# -------------------------------
_FACE_DETECTOR = dlib.get_frontal_face_detector()



# -------------------------------
# Resolve paths safely
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHAPE_PREDICTOR_PATH = os.path.join(
    BASE_DIR,
    "shape_predictor_model",
    "shape_predictor_68_face_landmarks.dat"
)

if not os.path.exists(SHAPE_PREDICTOR_PATH):
    raise FileNotFoundError(
        f"Shape predictor not found at {SHAPE_PREDICTOR_PATH}"
    )

# -------------------------------
# Load models ONCE
# -------------------------------
_FACE_DETECTOR = dlib.get_frontal_face_detector()
_SHAPE_PREDICTOR = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)


def detect_faces(
    frame,
    draw_box: bool = True,
    draw_landmarks: bool = False
):
    """
    Face detection using dlib HOG detector.

    Args:
        frame: BGR frame from OpenCV
        draw_box: draw face bounding box
        draw_landmarks: draw 68 facial landmarks

    Returns:
        face_count (int)
        faces (list of dicts):
            {
              "rect": dlib.rectangle,
              "box": (x, y, w, h),
              "landmarks": np.ndarray | None
            }
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = _FACE_DETECTOR(gray, 0)

    faces = []

    for face in detections:
        x = face.left()
        y = face.top()
        w = face.width()
        h = face.height()

        landmarks_np = None

        if draw_landmarks:
            landmarks = _SHAPE_PREDICTOR(gray, face)
            landmarks_np = face_utils.shape_to_np(landmarks)

            for (lx, ly) in landmarks_np:
                cv2.circle(frame, (lx, ly), 1, (255, 255, 0), -1)

        if draw_box:
            _draw_fancy_box(frame, x, y, w, h)

        faces.append({
            "rect": face,
            "box": (x, y, w, h),
            "landmarks": landmarks_np
        })

    return len(faces), faces


# -------------------------------
# Helper: fancy bounding box
# -------------------------------
def _draw_fancy_box(frame, x, y, w, h):
    c = (0, 255, 255)
    t = 2
    l = 20

    # top-left
    cv2.line(frame, (x, y), (x + l, y), c, t)
    cv2.line(frame, (x, y), (x, y + l), c, t)

    # top-right
    cv2.line(frame, (x + w, y), (x + w - l, y), c, t)
    cv2.line(frame, (x + w, y), (x + w, y + l), c, t)

    # bottom-left
    cv2.line(frame, (x, y + h), (x + l, y + h), c, t)
    cv2.line(frame, (x, y + h), (x, y + h - l), c, t)

    # bottom-right
    cv2.line(frame, (x + w, y + h), (x + w - l, y + h), c, t)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - l), c, t)
