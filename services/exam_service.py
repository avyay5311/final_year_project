# services/exam_service.py
# Exam proctoring service using the shared pipeline

import threading
import time
import json
import os

from config import Config
from services.camera_service import camera
from core_ai.pipeline import ProctoringPipeline


class ExamProctor:
    """
    Flask-compatible exam proctoring service.
    
    Uses ProctoringPipeline from core_ai/pipeline.py which contains
    the same AI logic as test_primary_face.py but without cv2.imshow.
    """
    
    def __init__(self, candidate_id):
        self.candidate_id = candidate_id
        self.is_running = False
        self.processing_thread = None
        self.warnings = []
        self.warnings_lock = threading.Lock()
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.identity_valid = False
        self.identity_lock = threading.Lock()
        
        # Initialize the shared pipeline
        self.pipeline = ProctoringPipeline(candidate_id=candidate_id)
    
    def start(self):
        """Start exam proctoring in background thread."""
        if self.is_running:
            return False, "Proctoring already running"
        
        if not camera.start():
            return False, "Could not access camera"
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        return True, "Proctoring started"
    
    def stop(self):
        """Stop exam proctoring."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        camera.stop()
        return True, "Proctoring stopped"
    
    def _processing_loop(self):
        """Main processing loop - runs in background thread."""
        
        while self.is_running:
            frame = camera.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Process frame through the shared pipeline
            result = self.pipeline.process_frame(frame)
            
            # Store current frame for video feed
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Store warnings
            if result['warnings']:
                with self.warnings_lock:
                    self.warnings = result['warnings']
            else:
                with self.warnings_lock:
                    self.warnings = []
            
            # Store identity status
            with self.identity_lock:
                self.identity_valid = result['identity_valid']
    
    def get_warnings(self):
        """Get current warnings (thread-safe)."""
        with self.warnings_lock:
            return self.warnings.copy()
    
    def get_current_frame(self):
        """Get current processed frame (thread-safe)."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def get_identity_status(self):
        """Get current identity validation status (thread-safe)."""
        with self.identity_lock:
            return self.identity_valid
    
    def finalize(self):
        """Finalize channels and save summaries."""
        
        # Finalize all channels via pipeline
        self.pipeline.finalize_channels()
        
        # Get summaries
        summaries = self.pipeline.get_channel_summaries()
        
        # Save summaries to files
        Config.init_directories()
        
        with open(os.path.join(Config.SUMMARIES_DIR, "gaze_summary.json"), "w") as f:
            json.dump(summaries['gaze'], f, indent=4)
        
        with open(os.path.join(Config.SUMMARIES_DIR, "blink_summary.json"), "w") as f:
            json.dump(summaries['blink'], f, indent=4)
        
        with open(os.path.join(Config.SUMMARIES_DIR, "headpose_summary.json"), "w") as f:
            json.dump(summaries['headpose'], f, indent=4)
        
        with open(os.path.join(Config.SUMMARIES_DIR, "mouth_summary.json"), "w") as f:
            json.dump(summaries['mouth'], f, indent=4)
        
        print("Behavior summaries saved.")
        return True


# -------------------------------------------------
# Global proctor instance management
# -------------------------------------------------

_active_proctor = None
_proctor_lock = threading.Lock()


def start_exam_proctoring(candidate_id):
    """Start exam proctoring for a candidate."""
    global _active_proctor
    
    with _proctor_lock:
        if _active_proctor is not None:
            return False, "Another exam session is already active"
        
        _active_proctor = ExamProctor(candidate_id)
        success, message = _active_proctor.start()
        
        if not success:
            _active_proctor = None
        
        return success, message


def stop_exam_proctoring():
    """Stop current exam proctoring session."""
    global _active_proctor
    
    with _proctor_lock:
        if _active_proctor is None:
            return False, "No active exam session"
        
        _active_proctor.stop()
        _active_proctor.finalize()
        _active_proctor = None
        
        return True, "Proctoring stopped and finalized"


def get_current_warnings():
    """Get warnings from active proctor."""
    global _active_proctor
    
    with _proctor_lock:
        if _active_proctor is None:
            return []
        return _active_proctor.get_warnings()


def get_video_frame():
    """Get current video frame from active proctor."""
    global _active_proctor
    
    with _proctor_lock:
        if _active_proctor is None:
            return None
        return _active_proctor.get_current_frame()


def get_identity_status():
    """Get identity status from active proctor."""
    global _active_proctor
    
    with _proctor_lock:
        if _active_proctor is None:
            return False
        return _active_proctor.get_identity_status()
