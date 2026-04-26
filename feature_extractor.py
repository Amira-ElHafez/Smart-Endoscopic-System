"""
Feature Extractor — shape, color, and texture (LBP) features.
"""
import cv2
import numpy as np


class FeatureExtractor:

    def extract_all(self, frame: np.ndarray) -> dict:
        return {**self._shape(frame), **self._color(frame), **self._texture(frame)}

    def get_feature_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Generates a true Heatmap highlighting areas of high texture/edges (like vessels or polyps).
        """
        gray = self._gray(frame)

        # 1. Calculate texture and edge intensity using Sobel
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(dx, dy)

        # 2. Normalize to 0-255 range for visual mapping
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 3. Apply JET colormap to create a real "Thermal/Medical Heatmap"
        heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)

        return heatmap

    def _shape(self, frame):
        gray = self._gray(frame)
        _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = circularity = 0.0
        if contours:
            lc  = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(lc)
            p   = cv2.arcLength(lc, True)
            if p > 0:
                circularity = (4 * np.pi * largest_area) / (p ** 2)
        return {"contour_count": len(contours),
                "largest_area":  largest_area,
                "circularity":   circularity}

    def _color(self, frame):
        bgr  = frame if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        mean = cv2.mean(bgr)[:3]
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h    = hsv[:, :, 0].flatten()
        dom_hue = float(np.argmax(np.bincount(h, minlength=180))) * 2.0
        sat = float(np.mean(hsv[:, :, 1]))
        return {"mean_color": mean, "dominant_hue": dom_hue, "saturation": sat}

    def _texture(self, frame):
        gray = self._gray(frame).astype(np.float32)
        lbp  = self._lbp(gray)
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256), density=True)
        hist += 1e-10
        dx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return {"lbp_energy":        float(np.sum(hist ** 2)),
                "lbp_entropy":       float(-np.sum(hist * np.log2(hist))),
                "texture_contrast":  float(np.mean(dx ** 2 + dy ** 2) ** 0.5)}

    def _lbp(self, gray, radius=1, n=8):
        lbp    = np.zeros(gray.shape, dtype=np.uint8)
        angles = [2 * np.pi * p / n for p in range(n)]
        for a in angles:
            dy, dx = int(round(radius * np.sin(a))), int(round(radius * np.cos(a)))
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            lbp = (lbp << 1) | (gray >= shifted).astype(np.uint8)
        return lbp

    @staticmethod
    def _gray(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame