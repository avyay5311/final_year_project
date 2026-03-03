# services/registration_service.py
# Face registration service for capturing candidate face encodings

import os
import cv2
import numpy as np

from config import Config
from services.camera_service import camera

# Import the DNN face source and identity gate for consistent face detection
from core_ai.dnn_face_source import DNNFaceSource
from core_ai.identity_gate import IdentityGate


# Initialize face detection components (same as register_candidate.py)
_face_source = None
_identity_gate = None


def _get_face_source():
    """Lazy initialization of face source."""
    global _face_source
    if _face_source is None:
        _face_source = DNNFaceSource(
            Config.DNN_PROTOTXT,
            Config.DNN_CAFFEMODEL
        )
    return _face_source


def _get_identity_gate():
    """Lazy initialization of identity gate for embedding extraction."""
    global _identity_gate
    if _identity_gate is None:
        _identity_gate = IdentityGate(
            model_path=Config.OPENFACE_MODEL
        )
    return _identity_gate


def capture_face_encoding(candidate_id):
    """
    Capture face encoding for a candidate.
    Uses SINGLE frame capture - same as original register_candidate.py.
    
    Args:
        candidate_id: Unique ID of the candidate
    
    Returns:
        tuple: (success: bool, encoding_path: str or None, message: str)
    """
    
    if not camera.start():
        return False, None, "Could not access camera"
    
    face_source = _get_face_source()
    gate = _get_identity_gate()
    
    max_attempts = 100  # Maximum frames to try
    attempts = 0
    
    print(f"Capturing face encoding for candidate {candidate_id}...")
    
    while attempts < max_attempts:
        frame = camera.read_frame()
        if frame is None:
            attempts += 1
            continue
        
        # Detect faces using DNN (same as register_candidate.py)
        detections = face_source.get_faces(frame)
        
        if len(detections) == 1:
            # Single face detected - good
            det = detections[0]
            x, y, w, h = det["bbox"]
            
            # Crop and resize face
            face = frame[y:y + h, x:x + w]
            if face.size == 0:
                attempts += 1
                continue
            
            face = cv2.resize(face, (96, 96))
            
            # Get SINGLE embedding (same as register_candidate.py)
            embedding = gate._get_embedding(face)
            
            # Ensure directory exists
            Config.init_directories()
            
            # Save encoding to per-candidate file
            encoding_path = os.path.join(Config.ENCODINGS_DIR, f"{candidate_id}.npy")
            np.save(encoding_path, embedding)
            
            camera.stop()
            print(f"Face encoding saved to: {encoding_path}")
            return True, encoding_path, "Face encoding captured successfully"
            
        elif len(detections) > 1:
            print("Multiple faces detected - please ensure only one face is visible")
        else:
            print("No face detected - please look at the camera")
        
        attempts += 1
        cv2.waitKey(100)  # Small delay between attempts
    
    camera.stop()
    return False, None, "Could not capture face - please try again"


def verify_face_exists(candidate_id):
    """Check if face encoding exists for candidate."""
    encoding_path = os.path.join(Config.ENCODINGS_DIR, f"{candidate_id}.npy")
    return os.path.exists(encoding_path)


def get_encoding_path(candidate_id):
    """Get the encoding path for a candidate."""
    return os.path.join(Config.ENCODINGS_DIR, f"{candidate_id}.npy")


def load_face_encoding(candidate_id):
    """Load face encoding for a candidate."""
    encoding_path = get_encoding_path(candidate_id)
    if os.path.exists(encoding_path):
        return np.load(encoding_path)
    return None


def verify_identity(candidate_id, strong_threshold=0.55, weak_threshold=0.75):
    """
    Verify that the current face matches the registered candidate.
    Uses SINGLE frame comparison - same as original IdentityGate.verify().
    
    Args:
        candidate_id: Candidate ID to verify against
        strong_threshold: Distance for STRONG match (default: 0.55)
        weak_threshold: Distance for WEAK match (default: 0.75)
        
    Returns:
        tuple: (verified: bool, distance: float or None, message: str)
        - verified=True if STRONG or WEAK match
        - verified=False if MISMATCH (distance >= weak_threshold)
    """
    import time
    
    # Load registered encoding
    registered_encoding = load_face_encoding(candidate_id)
    if registered_encoding is None:
        return False, None, "No registered face found for this candidate"
    
    if not camera.start():
        return False, None, "Could not access camera"
    
    face_source = _get_face_source()
    gate = _get_identity_gate()
    
    max_attempts = 50
    attempts = 0
    
    print(f"Verifying identity for candidate {candidate_id}...")
    
    while attempts < max_attempts:
        # Use get_current_frame() to avoid thread contention with video feed
        frame = camera.get_current_frame()
        if frame is None:
            attempts += 1
            time.sleep(0.1)  # Longer delay when waiting for frame
            continue
        
        detections = face_source.get_faces(frame)
        
        if len(detections) == 1:
            det = detections[0]
            x, y, w, h = det["bbox"]
            
            face = frame[y:y + h, x:x + w]
            if face.size == 0:
                attempts += 1
                time.sleep(0.05)
                continue
            
            face = cv2.resize(face, (96, 96))
            current_embedding = gate._get_embedding(face)
            
            # Compute distance (same as IdentityGate)
            distance = np.linalg.norm(registered_encoding - current_embedding)
            
            # Tolerance-based decision (same as original)
            if distance < strong_threshold:
                verdict = "STRONG"
                print(f"Identity verification: {verdict} (distance: {distance:.4f})")
                return True, float(distance), f"Identity verified: {verdict} match"
            elif distance < weak_threshold:
                verdict = "WEAK"
                print(f"Identity verification: {verdict} (distance: {distance:.4f})")
                return True, float(distance), f"Identity verified: {verdict} match"
            else:
                verdict = "MISMATCH"
                print(f"Identity verification: {verdict} (distance: {distance:.4f})")
                return False, float(distance), "Face does not match registered identity"
            
        elif len(detections) > 1:
            return False, None, "Multiple faces detected - ensure only you are visible"
        
        attempts += 1
        time.sleep(0.05)
    
    return False, None, "Could not detect face - please look at the camera"
