# services/__init__.py
# Service layer exports

from services.camera_service import camera, CameraService
from services.registration_service import (
    capture_face_encoding,
    verify_face_exists,
    get_encoding_path,
    verify_identity
)
from services.exam_service import (
    start_exam_proctoring,
    stop_exam_proctoring,
    get_current_warnings,
    get_video_frame,
    get_identity_status,
    ExamProctor
)
from services.integrity_service import run_integrity_engine
from services.calibration_service import (
    calibrate_gaze,
    calibration_exists,
    load_calibration
)

__all__ = [
    'camera',
    'CameraService',
    'capture_face_encoding',
    'verify_face_exists',
    'get_encoding_path',
    'verify_identity',
    'start_exam_proctoring',
    'stop_exam_proctoring',
    'get_current_warnings',
    'get_video_frame',
    'get_identity_status',
    'ExamProctor',
    'run_integrity_engine',
    'calibrate_gaze',
    'calibration_exists',
    'load_calibration'
]
