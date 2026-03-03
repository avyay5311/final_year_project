# =============================================================
# integrity_engine/constants.py
#
# Central configuration for the IntegrityEngine.
# All weights, thresholds, amplifiers, and caps live here.
#
# To tune the scoring system — change numbers here only.
# No other file needs to be touched for calibration.
# =============================================================


# -------------------------------------------------------------
# Risk Level Thresholds
# Final integrity_score is compared against these to
# determine the human-readable risk label.
# -------------------------------------------------------------
RISK_LOW    = 80   # 80 → 100  : LOW RISK
RISK_MEDIUM = 60   # 60 → 79   : MEDIUM RISK
RISK_HIGH   = 40   # 40 → 59   : HIGH RISK
                   #  0 → 39   : CRITICAL


# -------------------------------------------------------------
# Impersonation Hard Cap
# If impersonation_count >= 1, the final integrity_score
# is hard-capped at this value regardless of other channels.
# This is the only event that overrides the scoring formula.
# -------------------------------------------------------------
IMPERSONATION_SCORE_CAP = 35


# -------------------------------------------------------------
# Per-Channel Penalty Cap
# No single channel can contribute more than this many
# penalty points on its own.
# Enforces the philosophy: combination evidence is required
# for HIGH RISK. One channel alone cannot destroy a score.
# -------------------------------------------------------------
SINGLE_CHANNEL_PENALTY_CAP = 40.0


# -------------------------------------------------------------
# Face Identity Channel — Weights
# Highest weight channel. Tracks who is in the frame.
# -------------------------------------------------------------
W_IMPERSONATION  = 1.0   # impersonation_count  — critical event
W_MULTIPLE_FACE  = 0.8   # multiple_face_count  — high weight
W_ABSENCE        = 0.7   # absence_count        — high weight
W_UNCERTAIN_LOW  = 0.05  # uncertain 5→30 sec   — mostly noise
W_UNCERTAIN_HIGH = 0.1   # uncertain > 30 sec   — low weight

# Uncertain duration tier thresholds (seconds)
UNCERTAIN_IGNORE_BELOW = 5.0    # below this → zero penalty
UNCERTAIN_LOW_BELOW    = 30.0   # below this → W_UNCERTAIN_LOW
                                # above this → W_UNCERTAIN_HIGH


# -------------------------------------------------------------
# Gaze Channel — Weights
# Medium-high weight channel.
# Moderate away is the most suspicious gaze bucket.
# -------------------------------------------------------------
W_GAZE_SUSTAINED  = 0.6   # sustained_away_count  (≥ 5.0 sec)
W_GAZE_MODERATE   = 0.5   # moderate_away_count   (1.5 → 5.0 sec)
W_GAZE_RAPID_DENS = 0.3   # rapid shift rate      (< 1.5 sec)

# Rapid gaze shifts below this rate (per minute) are innocent.
# Above this rate the excess is penalized.
RAPID_GAZE_INNOCENCE_RATE = 8.0   # shifts/min


# -------------------------------------------------------------
# Head Pose Channel — Weights
# Medium weight channel.
# Moderate away is the primary head pose signal.
# -------------------------------------------------------------
W_HEAD_SUSTAINED = 0.6   # sustained_away_count  (≥ 5.0 sec)
W_HEAD_MODERATE  = 0.5   # moderate_away_count   (1.5 → 5.0 sec)
W_HEAD_RAPID     = 0.2   # rapid away rate       (< 1.5 sec)

# Rapid head movements below this rate (per minute) are innocent.
RAPID_HEAD_INNOCENCE_RATE = 10.0   # movements/min


# -------------------------------------------------------------
# Mouth Channel — Weights
# Medium weight channel.
# Sustained active episodes are the primary mouth signal.
# -------------------------------------------------------------
W_MOUTH_SUSTAINED = 0.5   # sustained_active_count  (≥ 3.0 sec)
W_MOUTH_BURST     = 0.4   # burst_count             (whispering)
W_MOUTH_TOTAL_DUR = 0.3   # total_active_duration   (overall ratio)


# -------------------------------------------------------------
# Blink Channel — Weights
# Lowest weight channel. Corroborating signal only.
# -------------------------------------------------------------
W_BLINK_BURST    = 0.15   # burst_count        (stress/anxiety)
W_BLINK_LONG     = 0.1  # long_blink_count   (deliberate closure)
W_BLINK_RATE_DEV = 0.2   # deviation from normal blink rate band

# Normal blink rate band (blinks per minute)
# Deviation in either direction generates a small penalty.
NORMAL_BLINK_RATE_MIN = 15.0
NORMAL_BLINK_RATE_MAX = 20.0


# -------------------------------------------------------------
# Combination Scoring — Amplifiers
# Applied when two or more channel intervals overlap.
# These multiply the base penalty of the overlapping events.
# -------------------------------------------------------------
AMP_GAZE_HEAD  = 1.5   # gaze away + head turned simultaneously
AMP_GAZE_MOUTH = 1.6   # gaze away + talking simultaneously
AMP_HEAD_MOUTH = 1.5   # head turned + talking simultaneously
AMP_TRIPLE     = 2.0   # all three simultaneously — strongest signal
AMP_ABSENCE    = 1.8   # face absent + suspicious activity nearby


# -------------------------------------------------------------
# Combination Overlap Detection
# Two intervals must overlap by at least this ratio
# to trigger a combination penalty.
#
# overlap_ratio = overlap_duration / min(duration_a, duration_b)
# Fire if overlap_ratio >= OVERLAP_RATIO_MIN
# -------------------------------------------------------------
OVERLAP_RATIO_MIN = 0.3


# -------------------------------------------------------------
# Absence + ANY Proximity Window
# Since behavior channels go silent during absence,
# we look for suspicious events within this many seconds
# BEFORE or AFTER each absence interval.
# -------------------------------------------------------------
ABSENCE_PROXIMITY_WINDOW = 25.0   # seconds


# -------------------------------------------------------------
# Penalty Scale Factor
# Global multiplier applied to all penalty calculations.
# Increase to make the engine harsher overall.
# Decrease to make it more lenient overall.
# Touch this only if the entire scoring system feels
# consistently too harsh or too lenient across all sessions.
# -------------------------------------------------------------
PENALTY_SCALE = 10.0