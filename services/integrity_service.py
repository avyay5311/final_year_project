# services/integrity_service.py
# Integrity engine service wrapper

import os
import json
import uuid
import time

from config import Config
from integrity_engine.integrity_engine import IntegrityEngine


# Track session start time (set when exam starts)
_session_start_time = None
_session_id = None


def start_session():
    """Call when exam session starts to track timing."""
    global _session_start_time, _session_id
    _session_start_time = time.time()
    _session_id = f"session_{uuid.uuid4().hex[:8]}"
    return _session_id


def run_integrity_engine():
    """
    Run the integrity engine on saved summaries.
    
    Returns:
        dict: Integrity report with score, risk_level, flags, etc.
    """
    global _session_start_time, _session_id
    
    # Calculate session duration
    if _session_start_time is not None:
        session_duration = time.time() - _session_start_time
    else:
        session_duration = 0.0
    
    # Use stored session_id or generate one
    if _session_id is None:
        _session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    # Load summaries from files
    summaries_dir = Config.SUMMARIES_DIR
    
    with open(os.path.join(summaries_dir, "gaze_summary.json"), "r") as f:
        gaze_summary = json.load(f)
    
    with open(os.path.join(summaries_dir, "blink_summary.json"), "r") as f:
        blink_summary = json.load(f)
    
    with open(os.path.join(summaries_dir, "headpose_summary.json"), "r") as f:
        headpose_summary = json.load(f)
    
    with open(os.path.join(summaries_dir, "mouth_summary.json"), "r") as f:
        mouth_summary = json.load(f)
    
    with open(os.path.join(summaries_dir, "face_identity_summary.json"), "r") as f:
        face_identity_summary = json.load(f)
    
    # Run integrity engine with correct API
    engine = IntegrityEngine()
    report = engine.run(
        session_id=_session_id,
        session_duration=session_duration,
        gaze_summary=gaze_summary,
        headpose_summary=headpose_summary,
        blink_summary=blink_summary,
        mouth_summary=mouth_summary,
        face_identity_summary=face_identity_summary
    )
    
    print(f"\n{'='*55}")
    print(f"  INTEGRITY REPORT")
    print(f"{'='*55}")
    print(f"  Session ID      : {report['session_id']}")
    print(f"  Duration        : {report['session_duration']}s")
    print(f"  Integrity Score : {report['integrity_score']}")
    print(f"  Risk Level      : {report['risk_level']}")
    print(f"  Hard Cap        : {report['hard_cap_applied']}")
    print(f"  Total Flags     : {report['flag_summary']['total_flags']}")
    print(f"{'='*55}")
    
    # Reset session tracking
    _session_start_time = None
    _session_id = None
    
    return report
