import time
from enum import Enum


class IdentityState(Enum):
    PRIMARY_CONFIRMED = "PRIMARY_CONFIRMED"
    PRIMARY_TEMP_UNCERTAIN = "PRIMARY_TEMP_UNCERTAIN"
    PRIMARY_TEMP_ABSENT = "PRIMARY_TEMP_ABSENT"
    PRIMARY_REACQUIRED = "PRIMARY_REACQUIRED"
    MULTIPLE_FACES_PRESENT = "MULTIPLE_FACES_PRESENT"
    IMPERSONATION_SUSPECT = "IMPERSONATION_SUSPECT"


class PrimaryIdentityStateMachine:
    """
    Final identity control logic with persistence.

    Guarantees:
    - Absence ≠ impersonation
    - Identity tolerance (STRONG / WEAK / MISMATCH)
    - Primary can be re-acquired
    - Multi-face & replacement escalate only with time
    """

    def __init__(
        self,
        identity_gate,
        suspect_grace=10.0,
        multi_face_grace=3.0,
        replacement_grace=5.0,
    ):
        self.identity_gate = identity_gate

        self.suspect_grace = suspect_grace
        self.multi_face_grace = multi_face_grace
        self.replacement_grace = replacement_grace

        self.state = IdentityState.PRIMARY_TEMP_ABSENT

        self.last_mismatch_time = None
        self.multi_face_start_time = None
        self.replacement_start_time = None

        self.ever_confirmed_primary = False

    def reset(self):
        self.state = IdentityState.PRIMARY_TEMP_ABSENT
        self.last_mismatch_time = None
        self.multi_face_start_time = None
        self.replacement_start_time = None
        self.ever_confirmed_primary = False

    def update(self, frame, face_states):
        now = time.time()

        # -------------------------------------------------
        # CASE 1: NO FACE
        # -------------------------------------------------
        if len(face_states) == 0:
            self.state = IdentityState.PRIMARY_TEMP_ABSENT
            self.multi_face_start_time = None
            self.replacement_start_time = None
            return self.state, None

        # -------------------------------------------------
        # CASE 2: MULTIPLE FACES
        # -------------------------------------------------
        if len(face_states) > 1:
            if self.multi_face_start_time is None:
                self.multi_face_start_time = now

            if now - self.multi_face_start_time >= self.multi_face_grace:
                self.state = IdentityState.MULTIPLE_FACES_PRESENT
                return self.state, None

            self.state = IdentityState.PRIMARY_TEMP_UNCERTAIN
            return self.state, None

        # -------------------------------------------------
        # CASE 3: EXACTLY ONE FACE
        # -------------------------------------------------
        face = face_states[0]
        bbox = face["bbox"]

        verdict = self.identity_gate.verify(frame, bbox)

        # Reset multi-face timer
        self.multi_face_start_time = None

        # ---------------- STRONG MATCH ----------------
        if verdict == "STRONG":
            self.last_mismatch_time = None
            self.replacement_start_time = None
            self.ever_confirmed_primary = True

            if self.state in (
                IdentityState.PRIMARY_TEMP_ABSENT,
                IdentityState.PRIMARY_TEMP_UNCERTAIN,
            ):
                self.state = IdentityState.PRIMARY_REACQUIRED
            else:
                self.state = IdentityState.PRIMARY_CONFIRMED

            return self.state, face

        # ---------------- WEAK MATCH ----------------
        if verdict == "WEAK":
            self.last_mismatch_time = None
            self.replacement_start_time = None
            self.ever_confirmed_primary = True

            if self.state == IdentityState.PRIMARY_TEMP_ABSENT:
                self.state = IdentityState.PRIMARY_REACQUIRED
            else:
                self.state = IdentityState.PRIMARY_CONFIRMED

            return self.state, face

        # ---------------- CLEAR MISMATCH ----------------
        if verdict == "MISMATCH":

            # Face replacement logic
            if self.ever_confirmed_primary:
                if self.replacement_start_time is None:
                    self.replacement_start_time = now

                if now - self.replacement_start_time >= self.replacement_grace:
                    self.state = IdentityState.IMPERSONATION_SUSPECT
                    return self.state, None

            # Generic mismatch logic
            if self.last_mismatch_time is None:
                self.last_mismatch_time = now

            if now - self.last_mismatch_time >= self.suspect_grace:
                self.state = IdentityState.IMPERSONATION_SUSPECT
            else:
                self.state = IdentityState.PRIMARY_TEMP_UNCERTAIN

            return self.state, None
