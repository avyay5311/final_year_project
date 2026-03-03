# =============================================================
# integrity_engine/overlap_detector.py
#
# Pure interval overlap detection utility.
#
# Takes two lists of (start, end) time intervals and returns
# every overlapping pair along with the overlap duration
# and overlap ratio.
#
# No weights. No scoring logic. No channel knowledge.
# Used exclusively by combination_scorer.py
# =============================================================

from integrity_engine.constants import OVERLAP_RATIO_MIN


def detect_overlaps(intervals_a, intervals_b):
    """
    Find all overlapping pairs between two interval lists.

    Two intervals overlap if:
        max(start_a, start_b) < min(end_a, end_b)

    Overlap ratio:
        overlap_duration / min(duration_a, duration_b)

    Only returns pairs where overlap_ratio >= OVERLAP_RATIO_MIN
    (defined in constants.py as 0.3)

    Parameters
    ----------
    intervals_a : list of (float, float)
        First list of (start, end) timestamp tuples.

    intervals_b : list of (float, float)
        Second list of (start, end) timestamp tuples.

    Returns
    -------
    list of dict, each containing:
        overlap_start  : float — start of overlap window
        overlap_end    : float — end of overlap window
        overlap_dur    : float — duration of overlap in seconds
        overlap_ratio  : float — overlap_dur / min(dur_a, dur_b)
        interval_a     : (float, float) — the interval from list a
        interval_b     : (float, float) — the interval from list b

    Returns empty list if either input is empty or
    no pairs meet the overlap ratio threshold.

    Example
    -------
    intervals_a = [(100.0, 106.0)]   # 6 sec gaze away
    intervals_b = [(103.0, 108.0)]   # 5 sec head away

    overlap_start = max(100, 103) = 103.0
    overlap_end   = min(106, 108) = 106.0
    overlap_dur   = 106 - 103     = 3.0 sec
    min_dur       = min(6, 5)     = 5.0 sec
    overlap_ratio = 3.0 / 5.0     = 0.6  ← above 0.3 threshold
    → returned
    """

    # Nothing to compare
    if not intervals_a or not intervals_b:
        return []

    results = []

    for a in intervals_a:
        # ← FIX: always work with tuples so results are hashable
        a = tuple(a)
        a_start, a_end = a
        dur_a = a_end - a_start

        # Skip malformed intervals
        if dur_a <= 0:
            continue

        for b in intervals_b:
            # ← FIX: always work with tuples so results are hashable
            b = tuple(b)
            b_start, b_end = b
            dur_b = b_end - b_start

            # Skip malformed intervals
            if dur_b <= 0:
                continue

            # --- Overlap check ---
            overlap_start = max(a_start, b_start)
            overlap_end   = min(a_end,   b_end)

            # No overlap
            if overlap_start >= overlap_end:
                continue

            overlap_dur = overlap_end - overlap_start

            # --- Overlap ratio ---
            min_dur       = min(dur_a, dur_b)
            overlap_ratio = overlap_dur / min_dur

            # Below threshold — not significant enough
            if overlap_ratio < OVERLAP_RATIO_MIN:
                continue

            results.append({
                "overlap_start" : overlap_start,
                "overlap_end"   : overlap_end,
                "overlap_dur"   : overlap_dur,
                "overlap_ratio" : overlap_ratio,
                "interval_a"    : a,
                "interval_b"    : b,
            })

    return results


def detect_triple_overlaps(intervals_a, intervals_b, intervals_c):
    """
    Find all intervals where three channels overlap simultaneously.

    Strategy:
        1. Find all (a, b) overlaps using detect_overlaps()
        2. For each (a, b) overlap window, check if any interval
           in c also overlaps that window
        3. Return only triples where all three overlap

    Parameters
    ----------
    intervals_a : list of (float, float) — e.g. gaze away
    intervals_b : list of (float, float) — e.g. head away
    intervals_c : list of (float, float) — e.g. mouth active

    Returns
    -------
    list of dict, each containing:
        overlap_start  : float — start of three-way overlap
        overlap_end    : float — end of three-way overlap
        overlap_dur    : float — duration of three-way overlap
        interval_a     : (float, float)
        interval_b     : (float, float)
        interval_c     : (float, float)

    Returns empty list if no three-way overlaps are found.
    """

    if not intervals_a or not intervals_b or not intervals_c:
        return []

    # Step 1 — find all (a, b) overlapping pairs
    ab_overlaps = detect_overlaps(intervals_a, intervals_b)

    if not ab_overlaps:
        return []

    results = []

    for ab in ab_overlaps:

        # The (a, b) overlap window is itself an interval
        ab_window = [(ab["overlap_start"], ab["overlap_end"])]

        # Step 2 — check if any c interval overlaps this window
        abc_overlaps = detect_overlaps(ab_window, intervals_c)

        for abc in abc_overlaps:

            # Step 3 — compute the true three-way overlap window
            triple_start = max(ab["overlap_start"], abc["interval_b"][0])
            triple_end   = min(ab["overlap_end"],   abc["interval_b"][1])

            if triple_start >= triple_end:
                continue

            triple_dur = triple_end - triple_start

            results.append({
                "overlap_start" : triple_start,
                "overlap_end"   : triple_end,
                "overlap_dur"   : triple_dur,
                "interval_a"    : ab["interval_a"],
                "interval_b"    : ab["interval_b"],
                "interval_c"    : abc["interval_b"],
            })

    return results