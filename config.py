# config.py
# Centralized configuration settings for the exam proctoring system

import os


class Config:
    """Application configuration settings."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    
    # Database settings
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_USER = os.environ.get('DB_USER', 'root')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
    DB_NAME = os.environ.get('DB_NAME', 'quizo')
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ENCODINGS_DIR = os.path.join(BASE_DIR, "identity", "encodings")
    SUMMARIES_DIR = os.path.join(BASE_DIR, "summaries")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # AI Model paths
    DNN_PROTOTXT = os.path.join(MODELS_DIR, "dnn_face", "deploy.prototxt")
    DNN_CAFFEMODEL = os.path.join(MODELS_DIR, "dnn_face", "res10_300x300_ssd_iter_140000.caffemodel")
    OPENFACE_MODEL = os.path.join(MODELS_DIR, "face_recognition", "openface.nn4.small2.v1.t7")
    
    # Detection thresholds
    MIN_FACE_WIDTH = 120
    IDENTITY_STRONG_THRESHOLD = 0.71
    IDENTITY_WEAK_THRESHOLD = 0.77
    IDENTITY_COOLDOWN_SEC = 1.5
    
    # Gaze thresholds
    GAZE_HORIZONTAL_THRESH = 0.06
    GAZE_VERTICAL_THRESH = 0.06
    GAZE_SMOOTHING_WINDOW = 5
    GAZE_MIN_STABLE_FRAMES = 3
    
    # HeadPose thresholds
    HEADPOSE_YAW_THRESH = 8.0
    HEADPOSE_PITCH_THRESH = 8.0
    HEADPOSE_HYSTERESIS = 2.0
    HEADPOSE_SMOOTHING_WINDOW = 4
    HEADPOSE_MIN_STABLE_FRAMES = 3
    
    # Blink thresholds
    BLINK_EAR_THRESHOLD = 0.21
    BLINK_CONSEC_FRAMES = 2
    
    # Mouth detection thresholds (matching test_primary_face.py)
    MOUTH_MOVEMENT_THRESHOLD = 0.015
    MOUTH_SMOOTHING_WINDOW = 4
    MOUTH_DELTA_WINDOW = 5
    MOUTH_MIN_STABLE_FRAMES = 3
    
    # Verification threshold (more lenient for single-frame verification)
    IDENTITY_VERIFICATION_THRESHOLD = 0.85
    
    @staticmethod
    def init_directories():
        """Ensure required directories exist."""
        os.makedirs(Config.ENCODINGS_DIR, exist_ok=True)
        os.makedirs(Config.SUMMARIES_DIR, exist_ok=True)
