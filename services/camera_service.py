# services/camera_service.py
# Singleton camera service for webcam management

import cv2
import threading


class CameraService:
    """
    Singleton camera service.
    Manages webcam access across the application.
    Only one instance can access the camera at a time.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
    
    def start(self):
        """Start camera capture."""
        if self.cap is not None and self.cap.isOpened():
            print("Camera already running")
            return True
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("ERROR: Could not open camera")
            return False
        
        self.is_running = True
        print("Camera started successfully")
        return True
    
    def stop(self):
        """Stop camera capture."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("Camera stopped")
    
    def read_frame(self):
        """Read a single frame from camera."""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            with self.frame_lock:
                self.current_frame = frame.copy()
            return frame
        return None
    
    def get_current_frame(self):
        """Get the most recent frame (thread-safe)."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def is_active(self):
        """Check if camera is active."""
        return self.cap is not None and self.cap.isOpened() and self.is_running


# Global camera instance
camera = CameraService()
