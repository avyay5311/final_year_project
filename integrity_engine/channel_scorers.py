# =============================================================
# integrity_engine/channel_scorers.py
#
# Individual channel scoring functions for the IntegrityEngine.
#
# Each function:
#   - Reads a channel summary dict
#   - Normalizes signals using session_duration
#   - Applies weights from constants.py
#   - Returns (penalty: float, flags: list)
#
# No combination logic lives here.
# No final score assembly lives here.
# No channel caps are applied here — that happens in
# integrity_engine.py after all scorers have run.
# =============================================================

from integrity_engine.constants import (

    # Gaze
    W_GAZE_SUSTAINED,
    W_GAZE_MODERATE,
    W_GAZE_RAPID_DENS,
    RAPID_GAZE_INNOCENCE_RATE,

    # Head pose
    W_HEAD_SUSTAINED,
    W_HEAD_MODERATE,
    W_HEAD_RAPID,
    RAPID_HEAD_INNOCENCE_RATE,

    # Blink
    W_BLINK_BURST,
    W_BLINK_LONG,
    W_BLINK_RATE_DEV,
    NORMAL_BLINK_RATE_MIN,
    NORMAL_BLINK_RATE_MAX,

    # Mouth
    W_MOUTH_SUSTAINED,
    W_MOUTH_BURST,
    W_MOUTH_TOTAL_DUR,

    # Face identity
    W_IMPERSONATION,
    W_MULTIPLE_FACE,
    W_ABSENCE,
    W_UNCERTAIN_LOW,
    W_UNCERTAIN_HIGH,
    UNCERTAIN_IGNORE_BELOW,
    UNCERTAIN_LOW_BELOW,
    IMPERSONATION_SCORE_CAP,

    # Scale
    PENALTY_SCALE,
)

from integrity_engine.normalizers import (
    normalize_count,
    normalize_duration,
)


# =============================================================
# GAZE CHANNEL SCORER
# =============================================================

def score_gaze(gaze, session_duration):
    """
    Score the gaze channel summary.

    Signals evaluated (medium-high weight channel):
      - sustained_away_count : each occurrence penalized directly
      - moderate_away_count  : normalized by rate/min
      - rapid_shift_count    : only penalized above innocence rate

    Parameters
    ----------
    gaze             : dict — from GazeChannel.get_summary()
    session_duration : float — session length in seconds

    Returns
    -------
    (penalty: float, flags: list of dict)
    """
    penalty = 0.0
    flags   = []

    sustained_count = gaze.get("sustained_away_count", 0)
    moderate_count  = gaze.get("moderate_away_count",  0)
    rapid_count     = gaze.get("rapid_shift_count",    0)

    sustained_events = gaze.get("sustained_events",     [])
    moderate_events  = gaze.get("moderate_away_events", [])

    # ----------------------------------------------------------
    # Sustained away — each occurrence penalized directly
    # Not normalized — even one sustained away in a short session
    # is meaningful
    # ----------------------------------------------------------
    for ts in sustained_events:
        penalty += W_GAZE_SUSTAINED * PENALTY_SCALE
        flags.append({
            "type"      : "SUSTAINED_GAZE_AWAY",
            "timestamp" : ts,
            "duration"  : None,
            "severity"  : "HIGH",
        })

    # ----------------------------------------------------------
    # Moderate away — normalized by rate per minute
    # Most suspicious gaze bucket
    # ----------------------------------------------------------
    moderate_rate    = normalize_count(moderate_count, session_duration)
    moderate_penalty = moderate_rate * W_GAZE_MODERATE * PENALTY_SCALE
    penalty += moderate_penalty

    for ts in moderate_events:
        flags.append({
            "type"      : "MODERATE_GAZE_AWAY",
            "timestamp" : ts,
            "duration"  : None,
            "severity"  : "MEDIUM",
        })

    # ----------------------------------------------------------
    # Rapid shift density — only penalized above innocence rate
    # Below 8/min is completely innocent
    # ----------------------------------------------------------
    rapid_rate = normalize_count(rapid_count, session_duration)
    excess_rate = max(0.0, rapid_rate - RAPID_GAZE_INNOCENCE_RATE)

    if excess_rate > 0:
        rapid_penalty = excess_rate * W_GAZE_RAPID_DENS * PENALTY_SCALE
        penalty += rapid_penalty
        flags.append({
            "type"      : "RAPID_GAZE_EXCESS",
            "timestamp" : None,
            "duration"  : None,
            "severity"  : "LOW",
        })

    return penalty, flags


# =============================================================
# HEAD POSE CHANNEL SCORER
# =============================================================

def score_headpose(headpose, session_duration):
    """
    Score the head pose channel summary.

    Signals evaluated (medium weight channel):
      - sustained_away_count : each occurrence penalized directly
      - moderate_away_count  : normalized by rate/min
      - rapid_away_count     : only penalized above innocence rate

    Parameters
    ----------
    headpose         : dict — from HeadPoseChannel.get_summary()
    session_duration : float — session length in seconds

    Returns
    -------
    (penalty: float, flags: list of dict)
    """
    penalty = 0.0
    flags   = []

    sustained_count = headpose.get("sustained_away_count", 0)
    moderate_count  = headpose.get("moderate_away_count",  0)
    rapid_count     = headpose.get("rapid_away_count",     0)

    sustained_events = headpose.get("sustained_events",    [])
    moderate_events  = headpose.get("moderate_away_events",[])

    # ----------------------------------------------------------
    # Sustained away — each occurrence penalized directly
    # ----------------------------------------------------------
    for ts in sustained_events:
        penalty += W_HEAD_SUSTAINED * PENALTY_SCALE
        flags.append({
            "type"      : "SUSTAINED_HEAD_AWAY",
            "timestamp" : ts,
            "duration"  : None,
            "severity"  : "HIGH",
        })

    # ----------------------------------------------------------
    # Moderate away — normalized by rate per minute
    # ----------------------------------------------------------
    moderate_rate    = normalize_count(moderate_count, session_duration)
    moderate_penalty = moderate_rate * W_HEAD_MODERATE * PENALTY_SCALE
    penalty += moderate_penalty

    for ts in moderate_events:
        flags.append({
            "type"      : "MODERATE_HEAD_AWAY",
            "timestamp" : ts,
            "duration"  : None,
            "severity"  : "MEDIUM",
        })

    # ----------------------------------------------------------
    # Rapid away density — only penalized above innocence rate
    # Below 10/min is completely innocent
    # ----------------------------------------------------------
    rapid_rate  = normalize_count(rapid_count, session_duration)
    excess_rate = max(0.0, rapid_rate - RAPID_HEAD_INNOCENCE_RATE)

    if excess_rate > 0:
        rapid_penalty = excess_rate * W_HEAD_RAPID * PENALTY_SCALE
        penalty += rapid_penalty
        flags.append({
            "type"      : "RAPID_HEAD_EXCESS",
            "timestamp" : None,
            "duration"  : None,
            "severity"  : "LOW",
        })

    return penalty, flags


# =============================================================
# BLINK CHANNEL SCORER
# =============================================================

def score_blink(blink, session_duration):
    """
    Score the blink channel summary.

    Signals evaluated (lowest weight — corroborating only):
      - burst_count       : stress/anxiety signal
      - long_blink_count  : deliberate eye closure
      - blink rate        : deviation from normal 15-20/min band

    Parameters
    ----------
    blink            : dict — from BlinkChannel.get_summary()
    session_duration : float — session length in seconds

    Returns
    -------
    (penalty: float, flags: list of dict)
    """
    penalty = 0.0
    flags   = []

    burst_count      = blink.get("burst_count",      0)
    long_blink_count = blink.get("long_blink_count",  0)
    total_blinks     = blink.get("total_blink_count", 0)

    burst_events      = blink.get("burst_events",      [])
    long_blink_events = blink.get("long_blink_events", [])

    # ----------------------------------------------------------
    # Burst count — normalized by rate per minute
    # Stress and anxiety signal
    # ----------------------------------------------------------
    burst_rate    = normalize_count(burst_count, session_duration)
    burst_penalty = burst_rate * W_BLINK_BURST * PENALTY_SCALE
    penalty += burst_penalty

    for ts in burst_events:
        flags.append({
            "type"      : "BLINK_BURST",
            "timestamp" : ts,
            "duration"  : None,
            "severity"  : "LOW",
        })

    # ----------------------------------------------------------
    # Long blink count — raw count matters more here
    # Deliberate eye closure is not rate-dependent
    # ----------------------------------------------------------
    long_penalty = long_blink_count * W_BLINK_LONG * PENALTY_SCALE
    penalty += long_penalty

    for ts in long_blink_events:
        flags.append({
            "type"      : "LONG_BLINK",
            "timestamp" : ts,
            "duration"  : None,
            "severity"  : "LOW",
        })

    # ----------------------------------------------------------
    # Blink rate deviation
    # Normal band = 15-20 blinks/min
    # Deviation in either direction is penalized
    # Too low = suppressing blinks (stress/concentration)
    # Too high = stress/anxiety spike
    # ----------------------------------------------------------
    actual_rate = normalize_count(total_blinks, session_duration)

    if actual_rate < NORMAL_BLINK_RATE_MIN:
        deviation = (NORMAL_BLINK_RATE_MIN - actual_rate) / NORMAL_BLINK_RATE_MIN
    elif actual_rate > NORMAL_BLINK_RATE_MAX:
        deviation = (actual_rate - NORMAL_BLINK_RATE_MAX) / NORMAL_BLINK_RATE_MAX
    else:
        deviation = 0.0

    if deviation > 0:
        rate_penalty = deviation * W_BLINK_RATE_DEV * PENALTY_SCALE
        penalty += rate_penalty
        flags.append({
            "type"      : "BLINK_RATE_DEVIATION",
            "timestamp" : None,
            "duration"  : None,
            "severity"  : "LOW",
        })

    return penalty, flags


# =============================================================
# MOUTH CHANNEL SCORER
# =============================================================

def score_mouth(mouth, session_duration):
    """
    Score the mouth channel summary.

    Signals evaluated (medium weight channel):
      - sustained_active_count : deliberate speech episodes
      - burst_count            : whispering pattern
      - total_active_duration  : overall mouth activity ratio

    Parameters
    ----------
    mouth            : dict — from MouthChannel.get_summary()
    session_duration : float — session length in seconds

    Returns
    -------
    (penalty: float, flags: list of dict)
    """
    penalty = 0.0
    flags   = []

    sustained_count    = mouth.get("sustained_active_count", 0)
    burst_count        = mouth.get("burst_count",            0)
    total_duration     = mouth.get("total_active_duration",  0.0)

    sustained_events   = mouth.get("sustained_events",       [])
    burst_events       = mouth.get("burst_events",           [])

    # ----------------------------------------------------------
    # Sustained active — each occurrence penalized directly
    # Deliberate speech episodes are the primary mouth signal
    # ----------------------------------------------------------
    for ts in sustained_events:
        penalty += W_MOUTH_SUSTAINED * PENALTY_SCALE
        flags.append({
            "type"      : "MOUTH_SUSTAINED_ACTIVE",
            "timestamp" : ts,
            "duration"  : None,
            "severity"  : "MEDIUM",
        })

    # ----------------------------------------------------------
    # Burst count — normalized by rate per minute
    # Whispering pattern — multiple rapid mouth movements
    # ----------------------------------------------------------
    burst_rate    = normalize_count(burst_count, session_duration)
    burst_penalty = burst_rate * W_MOUTH_BURST * PENALTY_SCALE
    penalty += burst_penalty

    for ts in burst_events:
        flags.append({
            "type"      : "MOUTH_BURST_PATTERN",
            "timestamp" : ts,
            "duration"  : None,
            "severity"  : "MEDIUM",
        })

    # ----------------------------------------------------------
    # Total active duration ratio
    # Overall mouth activity as a proportion of session
    # ----------------------------------------------------------
    duration_ratio    = normalize_duration(total_duration, session_duration)
    duration_penalty  = duration_ratio * W_MOUTH_TOTAL_DUR * PENALTY_SCALE
    penalty += duration_penalty

    return penalty, flags


# =============================================================
# FACE IDENTITY CHANNEL SCORER
# =============================================================

def score_face_identity(face_identity, session_duration):
    """
    Score the face identity channel summary.

    Signals evaluated (highest weight channel):
      - impersonation_count : critical — triggers hard cap
      - multiple_face_count : high weight
      - absence_count       : high weight
      - uncertain_duration  : tiered low weight — mostly noise

    Parameters
    ----------
    face_identity    : dict — from FaceIdentityChannel.get_summary()
    session_duration : float — session length in seconds

    Returns
    -------
    (penalty: float, flags: list of dict, hard_cap: bool)

    Note: returns three values unlike other scorers.
    hard_cap=True means integrity_score must be capped at
    IMPERSONATION_SCORE_CAP regardless of other channels.
    """
    penalty  = 0.0
    flags    = []
    hard_cap = False

    impersonation_count = face_identity.get("impersonation_count", 0)
    multiple_face_count = face_identity.get("multiple_face_count", 0)
    absence_count       = face_identity.get("absence_count",       0)
    uncertain_duration  = face_identity.get("uncertain_duration",  0.0)

    impersonation_events = face_identity.get("impersonation_events", [])
    multiple_face_events = face_identity.get("multiple_face_events", [])
    absence_events       = face_identity.get("absence_events",       [])

    # ----------------------------------------------------------
    # Impersonation — critical event
    # Even one occurrence triggers the hard cap
    # Penalty is set high enough to guarantee score <= 35
    # regardless of what other channels report
    # ----------------------------------------------------------
    if impersonation_count >= 1:
        hard_cap = True
        # Penalty large enough to push score to 0 before cap
        penalty += impersonation_count * W_IMPERSONATION * PENALTY_SCALE * 10

        for ts in impersonation_events:
            flags.append({
                "type"      : "IMPERSONATION_SUSPECT",
                "timestamp" : ts,
                "duration"  : None,
                "severity"  : "CRITICAL",
            })

    # ----------------------------------------------------------
    # Multiple faces — high weight
    # Raw count matters — each occurrence is significant
    # ----------------------------------------------------------
    if multiple_face_count >= 1:
        multiple_penalty = (
            multiple_face_count * W_MULTIPLE_FACE * PENALTY_SCALE
        )
        penalty += multiple_penalty

        for ts in multiple_face_events:
            flags.append({
                "type"      : "MULTIPLE_FACES",
                "timestamp" : ts,
                "duration"  : None,
                "severity"  : "HIGH",
            })

    # ----------------------------------------------------------
    # Absence — high weight
    # Normalized by rate — frequent absences in a long session
    # are more suspicious than rare ones
    # ----------------------------------------------------------
    if absence_count >= 1:
        absence_rate    = normalize_count(absence_count, session_duration)
        absence_penalty = absence_rate * W_ABSENCE * PENALTY_SCALE
        penalty += absence_penalty

        for ts in absence_events:
            flags.append({
                "type"      : "FACE_ABSENT",
                "timestamp" : ts,
                "duration"  : None,
                "severity"  : "MEDIUM",
            })

    # ----------------------------------------------------------
    # Uncertain duration — tiered, very low weight
    # Caused by lighting, distance, jitter — mostly innocent
    # Never drives risk level alone
    # ----------------------------------------------------------
    if uncertain_duration >= UNCERTAIN_IGNORE_BELOW:
        if uncertain_duration < UNCERTAIN_LOW_BELOW:
            uncertain_weight = W_UNCERTAIN_LOW
        else:
            uncertain_weight = W_UNCERTAIN_HIGH

        uncertain_penalty = uncertain_duration * uncertain_weight
        penalty += uncertain_penalty
        # No flag generated for uncertain — too noisy

    return penalty, flags, hard_cap