# =============================================================
# integrity_engine/normalizers.py
#
# Normalization utilities for the IntegrityEngine.
#
# All scoring functions must normalize their inputs before
# computing penalties so that a 10-minute session and a
# 2-hour session produce comparable scores.
#
# No weights, no scoring logic, no imports live here.
# Pure calculation functions only.
# =============================================================


def normalize_count(count, session_duration):
    """
    Convert a raw event count into a per-minute rate.

    Use this for anything you are counting:
        rapid shifts, moderate aways, blink bursts, etc.

    Parameters
    ----------
    count            : int or float — raw event count
    session_duration : float        — session length in seconds

    Returns
    -------
    float — events per minute
            returns 0.0 if session_duration is zero or negative

    Example
    -------
    rapid_shift_count = 24, session_duration = 300 sec (5 min)
    → normalize_count(24, 300) = 24 / 5.0 = 4.8 shifts/min
    """
    if session_duration <= 0:
        return 0.0

    minutes = session_duration / 60.0
    return count / minutes


def normalize_duration(duration, session_duration):
    """
    Convert a raw duration into a ratio of total session time.

    Use this for anything you are measuring in seconds:
        total away duration, total mouth active duration, etc.

    Parameters
    ----------
    duration         : float — raw duration in seconds
    session_duration : float — session length in seconds

    Returns
    -------
    float — ratio between 0.0 and 1.0
            0.0 means none of the session was spent in this state
            1.0 means the entire session was spent in this state
            returns 0.0 if session_duration is zero or negative

    Example
    -------
    total_away_duration = 29.93, session_duration = 847.3 sec
    → normalize_duration(29.93, 847.3) = 0.0353  (3.5% of session)
    """
    if session_duration <= 0:
        return 0.0

    # Clamp to 1.0 — duration cannot exceed session length
    ratio = duration / session_duration
    return min(ratio, 1.0)