# core_ai/pipeline.py
# Shared AI pipeline component initialization
# Used by both test_primary_face.py (standalone) and exam_service.py (Flask)

import os
import sys
import json

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

from core_ai.dnn_face_source import DNNFaceSource
from core_ai.face_tracker import FaceTracker
from core_ai.duplicate_suppression import suppress_duplicates
from core_ai.identity_gate import IdentityGate
from core_ai.primary_identity_sm import PrimaryIdentityStateMachine, IdentityState
from core_ai.landmark_detector import LandmarkDetector
from core_ai.blink_detector import BlinkDetector, BlinkState
from core_ai.gaze_tracking import GazeTracking
from core_ai.headpose_detector import HeadPoseDetector, HeadPoseState
from core_ai.mouth_detector import MouthDetector, MouthState

from channels.gaze_channel import GazeChannel
from channels.blink_channel import BlinkChannel
from channels.headpose_channel import HeadPoseChannel
from channels.mouth_channel import MouthChannel


class ProctoringPipeline:
    """
    Encapsulates all AI components for proctoring.
    
    This class initializes all perception and behavior layer components.
    It can be used by:
    - test_primary_face.py (standalone testing with cv2.imshow)
    - exam_service.py (Flask web service)
    
    The pipeline itself does NOT include any display logic (cv2.imshow).
    """
    
    def __init__(self, registered_face_path=None, candidate_id=None):
        """
        Initialize the proctoring pipeline.
        
        Args:
            registered_face_path: Direct path to face encoding file
            candidate_id: If provided, constructs path from Config.ENCODINGS_DIR
        """
        
        self.candidate_id = candidate_id
        
        # Determine face encoding path
        if registered_face_path:
            self.face_encoding_path = registered_face_path
        elif candidate_id:
            self.face_encoding_path = os.path.join(
                Config.ENCODINGS_DIR, f"{candidate_id}.npy"
            )
        else:
            # Default path for backward compatibility
            self.face_encoding_path = "identity/registered_face.npy"
        
        # Initialize all components
        self._init_perception_layer()
        self._init_behavior_layer()
        
        # Load pre-calibrated gaze data if available
        self._load_gaze_calibration()
    
    def _init_perception_layer(self):
        """Initialize perception layer (core_ai) components."""
        
        # Face detection and tracking
        self.face_source = DNNFaceSource(
            Config.DNN_PROTOTXT,
            Config.DNN_CAFFEMODEL
        )
        self.tracker = FaceTracker()
        
        # Identity verification
        self.identity_gate = IdentityGate(
            registered_face_path=self.face_encoding_path,
            strong_threshold=Config.IDENTITY_STRONG_THRESHOLD,
            weak_threshold=Config.IDENTITY_WEAK_THRESHOLD,
            cooldown_sec=Config.IDENTITY_COOLDOWN_SEC
        )
        self.identity_sm = PrimaryIdentityStateMachine(self.identity_gate)
        
        # Landmark detection
        self.landmark_detector = LandmarkDetector()
        
        # Blink detection
        self.blink_detector = BlinkDetector(
            ear_threshold=Config.BLINK_EAR_THRESHOLD,
            consec_frames=Config.BLINK_CONSEC_FRAMES
        )
        
        # Gaze tracking
        self.gaze = GazeTracking(
            smoothing_window=Config.GAZE_SMOOTHING_WINDOW,
            min_stable_frames=Config.GAZE_MIN_STABLE_FRAMES,
            min_face_width=Config.MIN_FACE_WIDTH,
            horizontal_thresh=Config.GAZE_HORIZONTAL_THRESH,
            vertical_thresh=Config.GAZE_VERTICAL_THRESH
        )
        
        # Head pose detection
        self.headpose_detector = HeadPoseDetector(
            yaw_thresh=Config.HEADPOSE_YAW_THRESH,
            pitch_thresh=Config.HEADPOSE_PITCH_THRESH,
            hysteresis=Config.HEADPOSE_HYSTERESIS,
            smoothing_window=Config.HEADPOSE_SMOOTHING_WINDOW,
            min_stable_frames=Config.HEADPOSE_MIN_STABLE_FRAMES,
            min_face_width=Config.MIN_FACE_WIDTH
        )
        
        # Mouth detection
        self.mouth_detector = MouthDetector(
            movement_threshold=0.015,
            smoothing_window=4,
            delta_window=5,
            min_stable_frames=3,
            min_face_width=Config.MIN_FACE_WIDTH
        )
    
    def _init_behavior_layer(self):
        """Initialize behavior layer (channels) components."""
        self.gaze_channel = GazeChannel()
        self.blink_channel = BlinkChannel()
        self.headpose_channel = HeadPoseChannel()
        self.mouth_channel = MouthChannel()
    
    def _load_gaze_calibration(self):
        """Load pre-calibrated gaze data if available for the candidate."""
        if self.candidate_id is None:
            return
        
        calibration_path = os.path.join(
            Config.ENCODINGS_DIR, f"{self.candidate_id}_gaze_calibration.json"
        )
        
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, 'r') as f:
                    data = json.load(f)
                
                self.gaze.center_hr = data['center_hr']
                self.gaze.center_vr = data['center_vr']
                self.gaze._calibrated = True
                
                print(f"Loaded gaze calibration: center_hr={data['center_hr']:.4f}, center_vr={data['center_vr']:.4f}")
            except Exception as e:
                print(f"Failed to load gaze calibration: {e}")
    
    def process_frame(self, frame):
        """
        Process a single frame through the AI pipeline.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            dict: {
                'identity_state': IdentityState,
                'primary_face': face dict or None,
                'gaze_label': str,
                'blink_state': BlinkState,
                'headpose_state': HeadPoseState,
                'landmarks': landmarks or None,
                'warnings': list of warning strings,
                'identity_valid': bool
            }
        """
        
        warnings = []
        
        # -------------------------------------------------
        # Face Detection and Tracking
        # -------------------------------------------------
        detections = self.face_source.get_faces(frame)
        face_states = self.tracker.update(detections)
        face_states = suppress_duplicates(face_states)
        
        # -------------------------------------------------
        # Identity Verification
        # -------------------------------------------------
        identity_state, primary_face = self.identity_sm.update(frame, face_states)
        
        # Initialize detection states
        gaze_label = "NO_PUPILS"
        blink_state = BlinkState.NO_BLINK
        headpose_state = HeadPoseState.HEAD_CENTER
        mouth_state = MouthState.MOUTH_STILL
        landmarks = None
        
        # -------------------------------------------------
        # Generate Warnings
        # -------------------------------------------------
        if identity_state == IdentityState.MULTIPLE_FACES_PRESENT:
            warnings.append("Multiple faces detected")
        
        if identity_state == IdentityState.PRIMARY_TEMP_ABSENT:
            warnings.append("Face not detected")
        
        if identity_state == IdentityState.IMPERSONATION_SUSPECT:
            warnings.append("Identity verification failed")
        
        # -------------------------------------------------
        # Detection Block (only if primary face confirmed)
        # -------------------------------------------------
        if (
            primary_face is not None
            and identity_state not in (
                IdentityState.PRIMARY_TEMP_ABSENT,
                IdentityState.IMPERSONATION_SUSPECT,
                IdentityState.MULTIPLE_FACES_PRESENT,
            )
        ):
            landmarks = self.landmark_detector.detect(frame, primary_face["bbox"])
            
            if landmarks is not None:
                x, y, w, h = primary_face["bbox"]
                
                # Gaze tracking
                self.gaze.refresh(frame, landmarks, face_width=w)
                gaze_label = self.gaze.stable_state()
                
                # Blink detection
                if w >= Config.MIN_FACE_WIDTH:
                    ear = self.blink_detector.compute_ear(landmarks)
                    if ear is not None:
                        blink_state = self.blink_detector.update(landmarks)
                
                # Head pose detection
                headpose_state = self.headpose_detector.update(
                    landmarks=landmarks,
                    face_width=w,
                    frame_shape=frame.shape
                )
                
                # Mouth detection
                if w >= Config.MIN_FACE_WIDTH:
                    mouth_state = self.mouth_detector.update(landmarks, w)
        
        # -------------------------------------------------
        # Gaze Warning
        # -------------------------------------------------
        # GazeTracking.stable_state() returns "LOOKING_CENTER", "LOOKING_AWAY", or "NO_PUPILS"
        if gaze_label == "LOOKING_AWAY":
            warnings.append("Please look at screen")
        
        # -------------------------------------------------
        # Update Behavior Channels
        # -------------------------------------------------
        identity_valid = (
            primary_face is not None
            and identity_state not in (
                IdentityState.PRIMARY_TEMP_ABSENT,
                IdentityState.MULTIPLE_FACES_PRESENT,
                IdentityState.IMPERSONATION_SUSPECT,
            )
        )
        
        self.gaze_channel.update(gaze_label, identity_valid)
        self.blink_channel.update(blink_state, identity_valid)
        self.headpose_channel.update(headpose_state, identity_valid)
        self.mouth_channel.update(mouth_state, identity_valid)
        
        return {
            'identity_state': identity_state,
            'primary_face': primary_face,
            'gaze_label': gaze_label,
            'blink_state': blink_state,
            'headpose_state': headpose_state,
            'landmarks': landmarks,
            'warnings': warnings,
            'identity_valid': identity_valid
        }
    
    def finalize_channels(self):
        """Finalize all behavior channels."""
        self.gaze_channel.finalize()
        self.blink_channel.finalize()
        self.headpose_channel.finalize()
        self.mouth_channel.finalize()
    
    def get_channel_summaries(self):
        """Get summaries from all channels."""
        return {
            'gaze': self.gaze_channel.get_summary(),
            'blink': self.blink_channel.get_summary(),
            'headpose': self.headpose_channel.get_summary(),
            'mouth': self.mouth_channel.get_summary()
        }
    
    def start_gaze_calibration(self, duration_sec=1.0):
        """Start gaze calibration."""
        self.gaze.start_calibration(duration_sec=duration_sec)
    
    @property
    def is_calibrating(self):
        """Check if gaze calibration is in progress."""
        return self.gaze.is_calibrating
    
    @property
    def calibration_progress(self):
        """Get gaze calibration progress."""
        return self.gaze.calibration_progress


# -------------------------------------------------
# Convenience function for backward compatibility
# -------------------------------------------------

def create_pipeline(registered_face_path=None, candidate_id=None):
    """
    Factory function to create a ProctoringPipeline.
    
    Args:
        registered_face_path: Direct path to face encoding file
        candidate_id: If provided, constructs path from Config.ENCODINGS_DIR
        
    Returns:
        ProctoringPipeline instance
    """
    return ProctoringPipeline(
        registered_face_path=registered_face_path,
        candidate_id=candidate_id
    )
