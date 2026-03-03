# =============================================================
# integrity_engine/integrity_engine.py
#
# Main orchestrator for the IntegrityEngine.
#
# Calls every scorer in the correct order and assembles
# the final report via report_builder.
#
# This is the only file the outside world needs to call.
# Everything else is internal.
#
# Usage:
#   from integrity_engine import IntegrityEngine
#   engine = IntegrityEngine()
#   report = engine.run(
#       session_id            = "session_001",
#       gaze_summary          = {...},
#       headpose_summary      = {...},
#       blink_summary         = {...},
#       mouth_summary         = {...},
#       face_identity_summary = {...},
#   )
# =============================================================

from integrity_engine.constants import (
    SINGLE_CHANNEL_PENALTY_CAP,
    IMPERSONATION_SCORE_CAP,
    PENALTY_SCALE,
)

from integrity_engine.channel_scorers import (
    score_gaze,
    score_headpose,
    score_blink,
    score_mouth,
    score_face_identity,
)

from integrity_engine.combination_scorer import (
    score_combinations,
)

from integrity_engine.report_builder import (
    build_report,
)


# =============================================================
# INTEGRITY ENGINE
# =============================================================

class IntegrityEngine:
    """
    Main entry point for the integrity scoring system.

    Instantiate once and call .run() for each session.

    Example
    -------
    engine = IntegrityEngine()
    report = engine.run(
        session_id            = "exam_001",
        gaze_summary          = gaze_data,
        headpose_summary      = headpose_data,
        blink_summary         = blink_data,
        mouth_summary         = mouth_data,
        face_identity_summary = face_data,
    )
    print(report["integrity_score"])
    print(report["risk_level"])
    """

    def run(
        self,
        session_id,
        session_duration,             # ← added as direct parameter
        gaze_summary,
        headpose_summary,
        blink_summary,
        mouth_summary,
        face_identity_summary,
    ):
        """
        Run the full integrity scoring pipeline for one session.

        Parameters
        ----------
        session_id            : str  — unique session identifier
        gaze_summary          : dict — from GazeChannel
        headpose_summary      : dict — from HeadPoseChannel
        blink_summary         : dict — from BlinkChannel
        mouth_summary         : dict — from MouthChannel
        face_identity_summary : dict — from FaceIdentityChannel

        Returns
        -------
        dict — complete integrity report
             see report_builder.build_report for full structure
        """

        # ----------------------------------------------------------
        # STEP 1 — Read session duration
        # All channel summaries carry session_duration.
        # Use gaze as the source of truth.
        # ----------------------------------------------------------

        # session_duration = gaze_summary.get("session_duration", 0.0)

        # ----------------------------------------------------------
        # STEP 2 — Score each channel individually
        # Each scorer returns (penalty, flags)
        # Face identity also returns hard_cap bool
        # ----------------------------------------------------------

        gaze_penalty, gaze_flags = score_gaze(
            gaze_summary, session_duration
        )

        head_penalty, head_flags = score_headpose(
            headpose_summary, session_duration
        )

        blink_penalty, blink_flags = score_blink(
            blink_summary, session_duration
        )

        mouth_penalty, mouth_flags = score_mouth(
            mouth_summary, session_duration
        )

        face_penalty, face_flags, hard_cap = score_face_identity(
            face_identity_summary, session_duration
        )

        # ----------------------------------------------------------
        # STEP 3 — Apply single channel penalty cap
        # No single channel can contribute more than
        # SINGLE_CHANNEL_PENALTY_CAP (40.0) to the total.
        # This prevents one extreme channel from making
        # all other channels irrelevant.
        # ----------------------------------------------------------

        gaze_penalty  = min(gaze_penalty,  SINGLE_CHANNEL_PENALTY_CAP)
        head_penalty  = min(head_penalty,  SINGLE_CHANNEL_PENALTY_CAP)
        blink_penalty = min(blink_penalty, SINGLE_CHANNEL_PENALTY_CAP)
        mouth_penalty = min(mouth_penalty, SINGLE_CHANNEL_PENALTY_CAP)
        face_penalty  = min(face_penalty,  SINGLE_CHANNEL_PENALTY_CAP)

        # ----------------------------------------------------------
        # STEP 4 — Score combinations
        # Must run AFTER individual channels so all
        # interval data is available from the summaries.
        # ----------------------------------------------------------

        combo_penalties, combo_flags = score_combinations(
            gaze_summary,
            headpose_summary,
            mouth_summary,
            face_identity_summary,
            session_duration,
        )

        # ----------------------------------------------------------
        # STEP 5 — Assemble channel penalties dict
        # ----------------------------------------------------------

        channel_penalties = {
            "gaze"          : gaze_penalty,
            "headpose"      : head_penalty,
            "blink"         : blink_penalty,
            "mouth"         : mouth_penalty,
            "face_identity" : face_penalty,
        }

        # ----------------------------------------------------------
        # STEP 6 — Compute total penalty
        # ----------------------------------------------------------

        total_channel_penalty = sum(channel_penalties.values())
        total_combo_penalty   = sum(combo_penalties.values())
        total_penalty         = total_channel_penalty + total_combo_penalty

        # ----------------------------------------------------------
        # STEP 7 — Compute integrity score
        # Score = 100 - total_penalty
        # Clamped to [0, 100]
        # ----------------------------------------------------------

        integrity_score = max(0.0, min(100.0, 100.0 - total_penalty))

        # ----------------------------------------------------------
        # STEP 8 — Apply hard cap for impersonation
        # If impersonation was detected, score cannot exceed
        # IMPERSONATION_SCORE_CAP (35) regardless of
        # what the formula produced.
        # This ensures impersonation always results in
        # at least a HIGH risk level.
        # ----------------------------------------------------------

        if hard_cap and integrity_score > IMPERSONATION_SCORE_CAP:
            integrity_score = float(IMPERSONATION_SCORE_CAP)

        # ----------------------------------------------------------
        # STEP 9 — Collect all flags
        # Order: channel flags first, combination flags last
        # report_builder will sort by severity
        # ----------------------------------------------------------

        all_flags = (
            gaze_flags
            + head_flags
            + blink_flags
            + mouth_flags
            + face_flags
            + combo_flags
        )

        # ----------------------------------------------------------
        # STEP 10 — Collect channel summaries for the report
        # ----------------------------------------------------------

        channel_summaries = {
            "gaze"          : gaze_summary,
            "headpose"      : headpose_summary,
            "blink"         : blink_summary,
            "mouth"         : mouth_summary,
            "face_identity" : face_identity_summary,
        }

        # ----------------------------------------------------------
        # STEP 11 — Build and return the final report
        # ----------------------------------------------------------

        report = build_report(
            session_id            = session_id,
            session_duration      = session_duration,
            integrity_score       = integrity_score,
            hard_cap_applied      = hard_cap,
            channel_penalties     = channel_penalties,
            combination_penalties = combo_penalties,
            all_flags             = all_flags,
            channel_summaries     = channel_summaries,
        )

        return report