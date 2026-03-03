# =============================================================
# integrity_engine/combination_scorer.py
#
# Combination scoring for the IntegrityEngine.
#
# Detects when two or more behavior channels fire
# simultaneously and amplifies the penalty to reflect
# the stronger evidence that corroboration provides.
#
# Combinations handled (most severe first):
#   TRIPLE          — gaze + headpose + mouth      × 2.0
#   GAZE + HEADPOSE — simultaneous                 × 1.5
#   GAZE + MOUTH    — simultaneous                 × 1.6
#   HEADPOSE + MOUTH— simultaneous                 × 1.5
#   ABSENCE + ANY   — suspicious activity ±25 sec
#                     around each absence interval × 1.8
#
# Intervals already counted in a TRIPLE are not
# double-counted in pairwise combinations.
# =============================================================

from integrity_engine.constants import (
    AMP_GAZE_HEAD,
    AMP_GAZE_MOUTH,
    AMP_HEAD_MOUTH,
    AMP_TRIPLE,
    AMP_ABSENCE,
    ABSENCE_PROXIMITY_WINDOW,
    PENALTY_SCALE,
    W_GAZE_SUSTAINED,
    W_GAZE_MODERATE,
    W_HEAD_SUSTAINED,
    W_HEAD_MODERATE,
    W_MOUTH_SUSTAINED,
    W_MOUTH_BURST,
)

from integrity_engine.overlap_detector import (
    detect_overlaps,
    detect_triple_overlaps,
)


# =============================================================
# COMBINATION SCORER — MAIN ENTRY POINT
# =============================================================

def score_combinations(
    gaze_summary,
    headpose_summary,
    mouth_summary,
    face_identity_summary,
    session_duration,
):
    """
    Detect and score all combination events.

    Parameters
    ----------
    gaze_summary          : dict — from GazeChannel.get_summary()
    headpose_summary      : dict — from HeadPoseChannel.get_summary()
    mouth_summary         : dict — from MouthChannel.get_summary()
    face_identity_summary : dict — from FaceIdentityChannel.get_summary()
    session_duration      : float — session length in seconds

    Returns
    -------
    combo_penalties : dict — penalty per combination type
    combo_flags     : list — flag dicts for the report
    """

    # Retrieve interval lists from summaries
    gaze_intervals     = gaze_summary.get("away_intervals",    [])
    headpose_intervals = headpose_summary.get("away_intervals", [])
    mouth_intervals    = mouth_summary.get("active_intervals",  [])
    absence_intervals  = face_identity_summary.get(
                            "absence_intervals", []
                         )

    combo_penalties = {
        "gaze_headpose"    : 0.0,
        "gaze_mouth"       : 0.0,
        "headpose_mouth"   : 0.0,
        "triple"           : 0.0,
        "absence_combined" : 0.0,
    }
    combo_flags = []

    # Track which interval pairs are already counted
    # in a TRIPLE so we do not double-count them
    triple_gaze_intervals     = set()
    triple_headpose_intervals = set()
    triple_mouth_intervals    = set()

    # ==========================================================
    # STEP 1 — TRIPLE (most severe — processed first)
    # Gaze away + Head away + Mouth active simultaneously
    # ==========================================================

    triple_events = detect_triple_overlaps(
        gaze_intervals,
        headpose_intervals,
        mouth_intervals,
    )

    for event in triple_events:

        overlap_dur   = event["overlap_dur"]
        overlap_start = event["overlap_start"]
        overlap_end   = event["overlap_end"]

        # Base penalty from all three channels combined
        # normalized by session duration
        base_penalty = _base_interval_penalty(
            event["interval_a"], W_GAZE_SUSTAINED, session_duration
        ) + _base_interval_penalty(
            event["interval_b"], W_HEAD_SUSTAINED, session_duration
        ) + _base_interval_penalty(
            event["interval_c"], W_MOUTH_SUSTAINED, session_duration
        )

        amplified = base_penalty * AMP_TRIPLE
        combo_penalties["triple"] += amplified

        combo_flags.append({
            "type"      : "TRIPLE_OVERLAP",
            "timestamp" : overlap_start,
            "duration"  : round(overlap_dur, 2),
            "severity"  : "CRITICAL",
        })

        # Mark these intervals as used in a TRIPLE
        triple_gaze_intervals.add(event["interval_a"])
        triple_headpose_intervals.add(event["interval_b"])
        triple_mouth_intervals.add(event["interval_c"])

    # ==========================================================
    # STEP 2 — GAZE + HEADPOSE
    # Skip pairs already counted in a TRIPLE
    # ==========================================================

    gh_events = detect_overlaps(gaze_intervals, headpose_intervals)

    for event in gh_events:

        # Already counted in TRIPLE — skip
        if (
            event["interval_a"] in triple_gaze_intervals
            and event["interval_b"] in triple_headpose_intervals
        ):
            continue

        overlap_dur   = event["overlap_dur"]
        overlap_start = event["overlap_start"]

        base_penalty = _base_interval_penalty(
            event["interval_a"], W_GAZE_MODERATE, session_duration
        ) + _base_interval_penalty(
            event["interval_b"], W_HEAD_MODERATE, session_duration
        )

        amplified = base_penalty * AMP_GAZE_HEAD
        combo_penalties["gaze_headpose"] += amplified

        combo_flags.append({
            "type"      : "GAZE_HEADPOSE_OVERLAP",
            "timestamp" : overlap_start,
            "duration"  : round(overlap_dur, 2),
            "severity"  : "HIGH",
        })

    # ==========================================================
    # STEP 3 — GAZE + MOUTH
    # Skip pairs already counted in a TRIPLE
    # ==========================================================

    gm_events = detect_overlaps(gaze_intervals, mouth_intervals)

    for event in gm_events:

        # Already counted in TRIPLE — skip
        if (
            event["interval_a"] in triple_gaze_intervals
            and event["interval_b"] in triple_mouth_intervals
        ):
            continue

        overlap_dur   = event["overlap_dur"]
        overlap_start = event["overlap_start"]

        base_penalty = _base_interval_penalty(
            event["interval_a"], W_GAZE_MODERATE, session_duration
        ) + _base_interval_penalty(
            event["interval_b"], W_MOUTH_SUSTAINED, session_duration
        )

        amplified = base_penalty * AMP_GAZE_MOUTH
        combo_penalties["gaze_mouth"] += amplified

        combo_flags.append({
            "type"      : "GAZE_MOUTH_OVERLAP",
            "timestamp" : overlap_start,
            "duration"  : round(overlap_dur, 2),
            "severity"  : "HIGH",
        })

    # ==========================================================
    # STEP 4 — HEADPOSE + MOUTH
    # Skip pairs already counted in a TRIPLE
    # ==========================================================

    hm_events = detect_overlaps(headpose_intervals, mouth_intervals)

    for event in hm_events:

        # Already counted in TRIPLE — skip
        if (
            event["interval_a"] in triple_headpose_intervals
            and event["interval_b"] in triple_mouth_intervals
        ):
            continue

        overlap_dur   = event["overlap_dur"]
        overlap_start = event["overlap_start"]

        base_penalty = _base_interval_penalty(
            event["interval_a"], W_HEAD_MODERATE, session_duration
        ) + _base_interval_penalty(
            event["interval_b"], W_MOUTH_SUSTAINED, session_duration
        )

        amplified = base_penalty * AMP_HEAD_MOUTH
        combo_penalties["headpose_mouth"] += amplified

        combo_flags.append({
            "type"      : "HEADPOSE_MOUTH_OVERLAP",
            "timestamp" : overlap_start,
            "duration"  : round(overlap_dur, 2),
            "severity"  : "HIGH",
        })

    # ==========================================================
    # STEP 5 — ABSENCE + ANY
    #
    # Behavior channels go SILENT during absence so direct
    # interval overlap is not possible.
    #
    # Instead: look for any suspicious event timestamp
    # within ABSENCE_PROXIMITY_WINDOW seconds before or
    # after each absence interval.
    # ==========================================================

    if absence_intervals:

        # Collect all suspicious event timestamps
        suspicious_timestamps = []

        suspicious_timestamps += gaze_summary.get(
            "moderate_away_events", []
        )
        suspicious_timestamps += gaze_summary.get(
            "sustained_events", []
        )
        suspicious_timestamps += headpose_summary.get(
            "moderate_away_events", []
        )
        suspicious_timestamps += headpose_summary.get(
            "sustained_events", []
        )
        suspicious_timestamps += mouth_summary.get(
            "sustained_events", []
        )
        suspicious_timestamps += mouth_summary.get(
            "burst_events", []
        )

        for absence in absence_intervals:
            absence_start = absence[0]
            absence_end   = absence[1]
            absence_dur   = absence_end - absence_start

            # Define proximity window around this absence
            window_start = absence_start - ABSENCE_PROXIMITY_WINDOW
            window_end   = absence_end   + ABSENCE_PROXIMITY_WINDOW

            # Find any suspicious event in the window
            nearby = [
                t for t in suspicious_timestamps
                if window_start <= t <= window_end
            ]

            if nearby:
                # Base penalty normalized by session duration
                base_penalty = (
                    absence_dur / max(session_duration, 1.0)
                ) * PENALTY_SCALE

                amplified = base_penalty * AMP_ABSENCE
                combo_penalties["absence_combined"] += amplified

                combo_flags.append({
                    "type"      : "ABSENCE_COMBINED",
                    "timestamp" : absence_start,
                    "duration"  : round(absence_dur, 2),
                    "severity"  : "HIGH",
                })

    return combo_penalties, combo_flags


# =============================================================
# INTERNAL UTILITY
# =============================================================

def _base_interval_penalty(interval, weight, session_duration):
    """
    Compute a normalized base penalty from a single interval.

    Uses the interval duration as a ratio of session length,
    multiplied by the channel weight and global scale.

    Normalizing by session duration ensures combination
    penalties stay proportional to individual channel
    penalties and do not dominate the final score.

    Parameters
    ----------
    interval         : (float, float) — (start, end) tuple
    weight           : float          — channel weight
    session_duration : float          — session length in seconds

    Returns
    -------
    float — normalized base penalty contribution
    """
    start, end = interval
    duration   = max(0.0, end - start)
    ratio      = duration / max(session_duration, 1.0)
    return ratio * weight * PENALTY_SCALE