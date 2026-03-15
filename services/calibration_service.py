# services/calibration_service.py
# Gaze calibration service for pre-exam calibration

import os
import time
import json
import threading

from config import Config
from services.camera_service import camera

# Import required AI components
from core_ai.dnn_face_source import DNNFaceSource
from core_ai.face_tracker import FaceTracker
from core_ai.duplicate_suppression import suppress_duplicates
from core_ai.identity_gate import IdentityGate
from core_ai.primary_identity_sm import PrimaryIdentityStateMachine, IdentityState
from core_ai.landmark_detector import LandmarkDetector
from core_ai.gaze_tracking import GazeTracking


# Lazy-loaded components (shared with registration_service)
_face_source = None
_landmark_detector = None
_gaze_tracker = None


def _get_face_source():
    """Lazy initialization of face source."""
    global _face_source
    if _face_source is None:
        _face_source = DNNFaceSource(
            Config.DNN_PROTOTXT,
            Config.DNN_CAFFEMODEL
        )
    return _face_source


def _get_landmark_detector():
    """Lazy initialization of landmark detector."""
    global _landmark_detector
    if _landmark_detector is None:
        _landmark_detector = LandmarkDetector()
    return _landmark_detector


def _get_gaze_tracker():
    """Get a new gaze tracker for calibration."""
    return GazeTracking(
        smoothing_window=Config.GAZE_SMOOTHING_WINDOW,
        min_stable_frames=Config.GAZE_MIN_STABLE_FRAMES,
        min_face_width=Config.MIN_FACE_WIDTH,
        horizontal_thresh=Config.GAZE_HORIZONTAL_THRESH,
        vertical_thresh=Config.GAZE_VERTICAL_THRESH
    )


def get_calibration_path(candidate_id):
    """Get the calibration data file path for a candidate."""
    return os.path.join(Config.ENCODINGS_DIR, f"{candidate_id}_gaze_calibration.json")


def load_calibration(candidate_id):
    """Load gaze calibration data for a candidate."""
    path = get_calibration_path(candidate_id)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def save_calibration(candidate_id, center_hr, center_vr):
    """Save gaze calibration data for a candidate."""
    Config.init_directories()
    path = get_calibration_path(candidate_id)
    data = {
        'center_hr': center_hr,
        'center_vr': center_vr,
        'timestamp': time.time()
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    return path


def calibrate_gaze(candidate_id, duration_sec=4.0):
    """
    Perform gaze calibration for a candidate.
    User should be looking at the center of the screen.
    
    Args:
        candidate_id: Candidate ID
        duration_sec: Duration of calibration (default: 2 seconds)
        
    Returns:
        tuple: (success: bool, message: str, progress_callback: optional)
    """
    
    # Camera should already be running from video feed
    if not camera.is_active():
        if not camera.start():
            return False, "Could not access camera", None
    
    face_source = _get_face_source()
    landmark_detector = _get_landmark_detector()
    gaze = _get_gaze_tracker()
    
    print(f"Starting gaze calibration for candidate {candidate_id} ({duration_sec}s)...")
    
    # Start calibration
    gaze.start_calibration(duration_sec=duration_sec)
    
    start_time = time.time()
    frames_processed = 0
    
    while gaze.is_calibrating:
        # Check timeout
        if time.time() - start_time > duration_sec + 2.0:  # Extra buffer
            break
        
        # Get frame from camera (use get_current_frame to avoid blocking)
        frame = camera.get_current_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        
        # Detect face
        detections = face_source.get_faces(frame)
        
        if len(detections) == 1:
            det = detections[0]
            x, y, w, h = det["bbox"]
            
            # Get landmarks
            landmarks = landmark_detector.detect(frame, (x, y, w, h))
            
            if landmarks is not None:
                # Update gaze tracker (this collects calibration samples)
                gaze.refresh(frame, landmarks, face_width=w)
                frames_processed += 1
        
        time.sleep(0.033)  # ~30 FPS
    
    # Check calibration result
    if gaze._calibrated:
        # Save calibration data
        save_calibration(candidate_id, gaze.center_hr, gaze.center_vr)
        print(f"Gaze calibration successful: center_hr={gaze.center_hr:.4f}, center_vr={gaze.center_vr:.4f}")
        return True, f"Gaze calibrated successfully (center: {gaze.center_hr:.3f}, {gaze.center_vr:.3f})", None
    else:
        print(f"Gaze calibration failed: only {len(gaze._calib_samples)} samples collected")
        return False, "Calibration failed - please look at the center of the screen and try again", None


def calibration_exists(candidate_id):
    """Check if calibration data exists for a candidate."""
    return os.path.exists(get_calibration_path(candidate_id))
