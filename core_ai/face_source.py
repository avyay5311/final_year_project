# core_ai/face_source.py

from abc import ABC, abstractmethod

class FaceSource(ABC):
    """
    Abstract FaceSource.
    All face providers (DNN, MediaPipe) must follow this contract.
    """

    @abstractmethod
    def get_faces(self, frame):
        """
        Input:
            frame (BGR image)
        Output:
            List of detections:
            [
                {
                    "bbox": (x, y, w, h),
                    "confidence": float
                }
            ]
        """
        pass
