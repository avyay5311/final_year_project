# =============================================================
# integrity_engine/report_builder.py
#
# Assembles the final integrity report dict from all
# channel penalties, flags, combination results, and
# the computed integrity score.
#
# No scoring logic lives here.
# No penalty computation lives here.
# Only assembly, formatting, and structure.
#
# Output format:
# {
#     "session_id"           : str,
#     "session_duration"     : float,
#     "generated_at"         : str  (ISO 8601 UTC),
#     "integrity_score"      : float,
#     "risk_level"           : str,
#     "hard_cap_applied"     : bool,
#     "channel_penalties"    : dict,
#     "combination_penalties": dict,
#     "total_penalty"        : float,
#     "flags"                : list,
#     "flag_summary"         : dict,
#     "channel_summaries"    : dict,
# }
# =============================================================

import datetime

from integrity_engine.constants import (
    RISK_LOW,
    RISK_MEDIUM,
    RISK_HIGH,
    IMPERSONATION_SCORE_CAP,
)


# =============================================================
# MAIN ENTRY POINT
# =============================================================

def build_report(
    session_id,
    session_duration,
    integrity_score,
    hard_cap_applied,
    channel_penalties,
    combination_penalties,
    all_flags,
    channel_summaries,
):
    """
    Build the final integrity report dict.

    Parameters
    ----------
    session_id            : str   — unique session identifier
    session_duration      : float — session length in seconds
    integrity_score       : float — final score 0-100
    hard_cap_applied      : bool  — True if impersonation cap hit
    channel_penalties     : dict  — penalty per channel
    combination_penalties : dict  — penalty per combination type
    all_flags             : list  — all flag dicts from all scorers
    channel_summaries     : dict  — raw summaries from each channel

    Returns
    -------
    dict — complete integrity report
    """

    risk_level    = _compute_risk_level(integrity_score)
    flag_summary  = _build_flag_summary(all_flags)
    sorted_flags  = _sort_flags(all_flags)
    total_penalty = (
        sum(channel_penalties.values())
        + sum(combination_penalties.values())
    )

    report = {
        "session_id"            : session_id,
        "session_duration"      : round(session_duration, 2),
        "generated_at"          : _utc_now(),
        "integrity_score"       : round(integrity_score, 2),
        "risk_level"            : risk_level,
        "hard_cap_applied"      : hard_cap_applied,
        "channel_penalties"     : _round_dict(channel_penalties),
        "combination_penalties" : _round_dict(combination_penalties),
        "total_penalty"         : round(total_penalty, 4),
        "flags"                 : sorted_flags,
        "flag_summary"          : flag_summary,
        "channel_summaries"     : channel_summaries,
    }

    return report


# =============================================================
# RISK LEVEL
# =============================================================

def _compute_risk_level(integrity_score):
    """
    Map integrity score to a risk level string.

    Thresholds (from constants.py):
        >= RISK_LOW    (80)  → LOW
        >= RISK_MEDIUM (60)  → MEDIUM
        >= RISK_HIGH   (40)  → HIGH
        <  RISK_HIGH   (40)  → CRITICAL

    Parameters
    ----------
    integrity_score : float — score between 0 and 100

    Returns
    -------
    str — one of LOW / MEDIUM / HIGH / CRITICAL
    """
    if integrity_score >= RISK_LOW:
        return "LOW"
    elif integrity_score >= RISK_MEDIUM:
        return "MEDIUM"
    elif integrity_score >= RISK_HIGH:
        return "HIGH"
    else:
        return "CRITICAL"


# =============================================================
# FLAG SUMMARY
# =============================================================

def _build_flag_summary(all_flags):
    """
    Build a summary dict counting flags by type and severity.

    Returns
    -------
    dict with structure:
    {
        "total_flags" : int,
        "by_severity" : {
            "CRITICAL" : int,
            "HIGH"     : int,
            "MEDIUM"   : int,
            "LOW"      : int,
        },
        "by_type"     : {
            "SUSTAINED_GAZE_AWAY" : int,
            ...
        }
    }
    """

    by_severity = {
        "CRITICAL" : 0,
        "HIGH"     : 0,
        "MEDIUM"   : 0,
        "LOW"      : 0,
    }

    by_type = {}

    for flag in all_flags:

        severity  = flag.get("severity",  "LOW")
        flag_type = flag.get("type",      "UNKNOWN")

        # Count by severity
        if severity in by_severity:
            by_severity[severity] += 1
        else:
            by_severity[severity] = 1

        # Count by type
        if flag_type in by_type:
            by_type[flag_type] += 1
        else:
            by_type[flag_type] = 1

    return {
        "total_flags" : len(all_flags),
        "by_severity" : by_severity,
        "by_type"     : by_type,
    }


# =============================================================
# FLAG SORTING
# =============================================================

def _sort_flags(all_flags):
    """
    Sort flags for the report.

    Sort order:
        1. Severity   — CRITICAL first, LOW last
        2. Timestamp  — earliest first within same severity
                        flags with None timestamp go last

    Parameters
    ----------
    all_flags : list of flag dicts

    Returns
    -------
    list of flag dicts — sorted
    """

    severity_order = {
        "CRITICAL" : 0,
        "HIGH"     : 1,
        "MEDIUM"   : 2,
        "LOW"      : 3,
    }

    def sort_key(flag):
        severity_rank  = severity_order.get(
            flag.get("severity", "LOW"), 3
        )
        # None timestamps sort to the end
        ts             = flag.get("timestamp")
        timestamp_rank = ts if ts is not None else float("inf")
        return (severity_rank, timestamp_rank)

    return sorted(all_flags, key=sort_key)


# =============================================================
# UTILITIES
# =============================================================

def _utc_now():
    """
    Return current UTC time as ISO 8601 string.

    Example: "2026-03-03T14:22:11.447Z"
    """
    now = datetime.datetime.utcnow()
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + \
           f"{now.microsecond // 1000:03d}Z"


def _round_dict(d, decimals=4):
    """
    Return a copy of dict d with all float values
    rounded to `decimals` places.
    """
    return {k: round(v, decimals) for k, v in d.items()}