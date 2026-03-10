# services/integrity_service.py
# Integrity engine service wrapper

import os
import json
import uuid

from config import Config
from integrity_engine.integrity_engine import IntegrityEngine


def _load_summary(filename):
    """
    Safely load a summary JSON file.
    Returns empty dict if file doesn't exist or is corrupted.
    """
    path = os.path.join(Config.SUMMARIES_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not load {filename}, using empty fallback")
            return {}
    else:
        print(f"Warning: {filename} not found, using empty fallback")
        return {}


def run_integrity_engine():
    """
    Run the integrity engine on saved summaries.
    
    Returns:
        dict: Integrity report with score, risk_level, flags, etc.
    """
    
    # Load all summaries with safe fallback
    gaze_summary = _load_summary("gaze_summary.json")
    blink_summary = _load_summary("blink_summary.json")
    headpose_summary = _load_summary("headpose_summary.json")
    mouth_summary = _load_summary("mouth_summary.json")
    face_identity_summary = _load_summary("face_identity_summary.json")
    
    # Load session duration from file (written by exam_service)
    duration_data = _load_summary("session_duration.json")
    session_duration = duration_data.get("session_duration", Config.EXAM_DURATION)
    
    # Apply floor and cap
    session_duration = max(
        Config.MIN_SESSION_DURATION,
        min(session_duration, Config.EXAM_DURATION)
    )
    
    # Generate session ID
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    # Run integrity engine
    engine = IntegrityEngine()
    report = engine.run(
        session_id=session_id,
        session_duration=session_duration,
        gaze_summary=gaze_summary,
        headpose_summary=headpose_summary,
        blink_summary=blink_summary,
        mouth_summary=mouth_summary,
        face_identity_summary=face_identity_summary
    )
    
    # Print summary to console
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
    
    return report
