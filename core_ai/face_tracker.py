import math
from core_ai.config import MAX_CENTROID_DISTANCE, MAX_MISSING_FRAMES


class FaceTracker:
    def __init__(self):
        self.next_face_id = 0
        self.faces = {}              # face_id -> FaceState
        self.primary_face_id = None  # <-- NEW (for stability)

    def _centroid(self, bbox):
        x, y, w, h = bbox
        return (int(x + w / 2), int(y + h / 2))

    def _distance(self, c1, c2):
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def update(self, detections):
        """
        detections: list of {bbox, confidence}
        returns: list of FaceState
        """

        # -----------------------------
        # CASE 1: No faces detected
        # -----------------------------
        if len(detections) == 0:
            for face in self.faces.values():
                face["visible"] = False
                face["missing_frames"] += 1
                face["age"] += 1

            self.faces = {
                fid: f for fid, f in self.faces.items()
                if f["missing_frames"] <= MAX_MISSING_FRAMES
            }
            return list(self.faces.values())

        # -----------------------------
        # CASE 2: Exactly ONE face detected
        # 👉 enforce ID stickiness
        # -----------------------------
        if len(detections) == 1 and len(self.faces) >= 1:
            det = detections[0]
            cx, cy = self._centroid(det["bbox"])

            # Prefer PRIMARY face if it exists
            if self.primary_face_id in self.faces:
                face = self.faces[self.primary_face_id]
                prev_center = face["center"]

                face.update({
                    "bbox": det["bbox"],
                    "center": (cx, cy),
                    "velocity": (cx - prev_center[0], cy - prev_center[1]),
                    "confidence": det["confidence"],
                    "visible": True,
                    "missing_frames": 0,
                    "age": face["age"] + 1
                })

                self.faces = {self.primary_face_id: face}
                return [face]

        # -----------------------------
        # CASE 3: General matching
        # -----------------------------
        updated_faces = {}
        used_detections = set()

        for fid, face in self.faces.items():
            face["visible"] = False
            best_match = None
            best_distance = float("inf")

            for i, det in enumerate(detections):
                if i in used_detections:
                    continue

                cx, cy = self._centroid(det["bbox"])
                dist = self._distance(face["center"], (cx, cy))

                if dist < best_distance and dist < MAX_CENTROID_DISTANCE:
                    best_distance = dist
                    best_match = i

            if best_match is not None:
                det = detections[best_match]
                used_detections.add(best_match)

                cx, cy = self._centroid(det["bbox"])
                prev_center = face["center"]

                face.update({
                    "bbox": det["bbox"],
                    "center": (cx, cy),
                    "velocity": (cx - prev_center[0], cy - prev_center[1]),
                    "confidence": det["confidence"],
                    "visible": True,
                    "missing_frames": 0,
                    "age": face["age"] + 1
                })

                updated_faces[fid] = face
            else:
                face["missing_frames"] += 1
                face["age"] += 1

                if face["missing_frames"] <= MAX_MISSING_FRAMES:
                    updated_faces[fid] = face

        # -----------------------------
        # Create new faces for unmatched detections
        # -----------------------------
        for i, det in enumerate(detections):
            if i in used_detections:
                continue

            cx, cy = self._centroid(det["bbox"])
            face_id = self.next_face_id
            self.next_face_id += 1

            updated_faces[face_id] = {
                "id": face_id,
                "bbox": det["bbox"],
                "center": (cx, cy),
                "velocity": (0, 0),
                "confidence": det["confidence"],
                "visible": True,
                "missing_frames": 0,
                "age": 1
            }

            # If this is the first face ever, set as primary
            if self.primary_face_id is None:
                self.primary_face_id = face_id

        self.faces = updated_faces
        return list(self.faces.values())
