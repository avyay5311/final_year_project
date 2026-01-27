# core_ai/primary_identity_sm.py

import time
from enum import Enum


class IdentityState(Enum):
    PRIMARY_CONFIRMED = "PRIMARY_CONFIRMED"
    PRIMARY_TEMP_UNCERTAIN = "PRIMARY_TEMP_UNCERTAIN"
    PRIMARY_TEMP_ABSENT = "PRIMARY_TEMP_ABSENT"
    PRIMARY_REACQUIRED = "PRIMARY_REACQUIRED"
    IMPERSONATION_SUSPECT = "IMPERSONATION_SUSPECT"


class PrimaryIdentityStateMachine:
    """
    Identity-driven primary authority.

    Guarantees:
    - Absence ≠ impersonation
    - Primary can be re-acquired after absence
    - WEAK matches are tolerated
    - Impersonation requires persistent mismatch
    """

    def __init__(self, identity_gate, suspect_grace=10.0):
        self.identity_gate = identity_gate
        self.suspect_grace = suspect_grace

        self.state = IdentityState.PRIMARY_TEMP_ABSENT
        self.last_mismatch_time = None

    def reset(self):
        self.state = IdentityState.PRIMARY_TEMP_ABSENT
        self.last_mismatch_time = None

    def update(self, frame, face_states):
        now = time.time()

        # -------------------------------------------------
        # CASE 1: NO FACE
        # -------------------------------------------------
        if len(face_states) == 0:
            # Absence never causes impersonation
            self.state = IdentityState.PRIMARY_TEMP_ABSENT
            self.last_mismatch_time = None
            return self.state, None

        # -------------------------------------------------
        # CASE 2: EXACTLY ONE FACE
        # -------------------------------------------------
        if len(face_states) == 1:
            face = face_states[0]
            bbox = face["bbox"]

            verdict = self.identity_gate.verify(frame, bbox)

            # ✅ STRONG MATCH
            if verdict == "STRONG":
                self.last_mismatch_time = None

                if self.state in (
                    IdentityState.PRIMARY_TEMP_ABSENT,
                    IdentityState.PRIMARY_TEMP_UNCERTAIN,
                ):
                    self.state = IdentityState.PRIMARY_REACQUIRED
                else:
                    self.state = IdentityState.PRIMARY_CONFIRMED

                return self.state, face

            # 🟡 WEAK MATCH → stay confirmed (NO downgrade)
            if verdict == "WEAK":
                self.last_mismatch_time = None

                if self.state == IdentityState.PRIMARY_TEMP_ABSENT:
                    self.state = IdentityState.PRIMARY_REACQUIRED
                else:
                    self.state = IdentityState.PRIMARY_CONFIRMED

                return self.state, face

            # 🔴 CLEAR MISMATCH → start suspicion timer
            if verdict == "MISMATCH":
                if self.last_mismatch_time is None:
                    self.last_mismatch_time = now

                if now - self.last_mismatch_time >= self.suspect_grace:
                    self.state = IdentityState.IMPERSONATION_SUSPECT
                else:
                    self.state = IdentityState.PRIMARY_TEMP_UNCERTAIN

                return self.state, None

        # -------------------------------------------------
        # CASE 3: MULTIPLE FACES
        # -------------------------------------------------
        # Multiple faces alone ≠ impersonation
        # Keep uncertainty but DO NOT reset recovery context
        self.state = IdentityState.PRIMARY_TEMP_UNCERTAIN
        return self.state, None
