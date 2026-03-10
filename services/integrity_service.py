# services/integrity_service.py
# Integrity engine service wrapper

import os
import json
import uuid

from config import Config
from integrity_engine.integrity_engine import IntegrityEngine


def run_integrity_engine():
    """
    Run the integrity engine on saved summaries.
    
    Loads channel summary JSON files from Config.SUMMARIES_DIR
    and passes them to IntegrityEngine.run().
    
    Returns:
        dict: Integrity report with score, risk_level, flags, etc.
    """
    
    summaries_dir = Config.SUMMARIES_DIR
    
    # Load all channel summaries from disk
    def _load_summary(filename):
        path = os.path.join(summaries_dir, filename)
        if not os.path.exists(path):
            return {}
        with open(path, 'r') as f:
            return json.load(f)
    
    gaze_summary = _load_summary("gaze_summary.json")
    headpose_summary = _load_summary("headpose_summary.json")
    blink_summary = _load_summary("blink_summary.json")
    mouth_summary = _load_summary("mouth_summary.json")
    face_identity_summary = _load_summary("face_identity_summary.json")
    
    # Compute session duration from available summaries
    session_duration = (
        gaze_summary.get("session_duration")
        or headpose_summary.get("session_duration")
        or blink_summary.get("session_duration")
        or mouth_summary.get("session_duration")
        or 0.0
    )
    
    # Run the integrity engine with the correct API
    engine = IntegrityEngine()
    report = engine.run(
        session_id=f"session_{uuid.uuid4().hex[:8]}",
        session_duration=session_duration,
        gaze_summary=gaze_summary,
        headpose_summary=headpose_summary,
        blink_summary=blink_summary,
        mouth_summary=mouth_summary,
        face_identity_summary=face_identity_summary,
    )
    
    print("\n========== INTEGRITY REPORT ==========")
    print(f"Integrity Score: {report['integrity_score']}")
    print(f"Risk Level: {report['risk_level']}")
    print(f"Total Flags: {report['flag_summary']['total_flags']}")
    print("======================================")
    
    return report
