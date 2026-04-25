"""
Image Processor — noise reduction, CLAHE, edge detection, colormaps.
"""
import cv2
import numpy as np


class ImageProcessor:
    COLORMAPS = {
        "Jet":  cv2.COLORMAP_JET,
        "Hot":  cv2.COLORMAP_HOT,
        "Cool": cv2.COLORMAP_COOL,
        "Bone": cv2.COLORMAP_BONE,
        "HSV":  cv2.COLORMAP_HSV,
    }

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def denoise(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def enhance_clahe(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 2:
            return self.clahe.apply(frame)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def detect_edges(self, frame: np.ndarray) -> np.ndarray:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        edges = cv2.Canny(gray, 50, 150)
        out   = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out[edges > 0] = [0, 220, 220]
        return out

    def apply_colormap(self, frame: np.ndarray, name: str) -> np.ndarray:
        cmap_id = self.COLORMAPS.get(name)
        if cmap_id is None:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        return cv2.applyColorMap(gray, cmap_id)