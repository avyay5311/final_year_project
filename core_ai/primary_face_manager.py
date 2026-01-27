class PrimaryFaceManager:
    def __init__(self, max_absence_frames=30):
        self.primary_face_id = None
        self.absence_counter = 0
        self.max_absence_frames = max_absence_frames

    def update(self, face_states):
        """
        face_states: List[FaceState]

        Returns:
        {
            "status": str,
            "primary_face": FaceState or None
        }
        """

        num_faces = len(face_states)

        # -------------------------------
        # HARD OVERRIDE: MULTIPLE FACES
        # -------------------------------
        if num_faces > 1:
            return {
                "status": "MULTIPLE_FACES_PRESENT",
                "primary_face": None
            }

        # -------------------------------
        # NO FACE
        # -------------------------------
        if num_faces == 0:
            self.absence_counter += 1

            if self.absence_counter > self.max_absence_frames:
                return {
                    "status": "PRIMARY_ABSENT_TOO_LONG",
                    "primary_face": None
                }

            return {
                "status": "NO_FACE",
                "primary_face": None
            }

        # -------------------------------
        # SINGLE FACE ONLY
        # -------------------------------
        face = face_states[0]

        # First time assignment
        if self.primary_face_id is None:
            self.primary_face_id = face["id"]
            self.absence_counter = 0
            return {
                "status": "PRIMARY_SET",
                "primary_face": face
            }

        # Same primary face continues
        if face["id"] == self.primary_face_id:
            self.absence_counter = 0
            return {
                "status": "PRIMARY_PRESENT",
                "primary_face": face
            }

        # Different face detected while primary missing
        self.absence_counter += 1

        if self.absence_counter > self.max_absence_frames:
            return {
                "status": "FACE_SWITCH",
                "primary_face": None
            }

        return {
            "status": "PRIMARY_TEMP_ABSENT",
            "primary_face": None
        }
