# =============================================================
# integrity_engine/__init__.py
#
# Public interface for the integrity_engine package.
#
# Everything the outside world needs is imported here.
# Internal modules (channel_scorers, overlap_detector etc.)
# are not exposed — they are implementation details.
#
# Usage:
#   from integrity_engine import IntegrityEngine
#
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

from integrity_engine.integrity_engine import IntegrityEngine

__all__ = ["IntegrityEngine"]

__version__ = "1.0.0"