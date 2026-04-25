"""
Illumination Controller — Simulates LED light source intensity and contrast.
"""

import cv2
import numpy as np


class IlluminationController:
    def __init__(self):
        self.brightness = 1.0   # alpha: 0.0 → 2.0 (1.0 = neutral)
        self.contrast = 1.0     # beta offset: maps to -80 … +80

    def set_brightness(self, value: float):
        """value: 0.0 (dark) → 2.0 (bright), 1.0 = no change."""
        self.brightness = max(0.0, min(2.0, value))

    def set_contrast(self, value: float):
        """value: 0.0 → 2.0 mapped to alpha for convertScaleAbs."""
        self.contrast = max(0.0, min(2.0, value))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply brightness and contrast.
        Uses cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness_offset).
        alpha scales pixel values; beta shifts them.
        """
        beta = (self.brightness - 1.0) * 80.0
        result = frame.astype(np.float32) * self.contrast + beta
        return np.clip(result, 0, 255).astype(np.uint8)