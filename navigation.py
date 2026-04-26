"""
Navigation Controller — Simulates insertion tube pan / tilt / zoom movement.
Uses pan_x, pan_y offsets, and zoom level to apply a crop+resize to the live frame.
"""

import cv2
import numpy as np


class NavigationController:
    STEP = 15      # pixels per pan keypress
    MAX_PAN = 120  # maximum offset from center
    TILT_STEP = 5
    MAX_TILT = 30

    # Zoom settings
    ZOOM_STEP = 0.1
    MAX_ZOOM = 3.0   # Maximum zoom 3x
    MIN_ZOOM = 1.0   # Normal view 1x

    def __init__(self):
        self.pan_x = 0
        self.pan_y = 0
        self.tilt = 0
        self.zoom = 1.0

    def move(self, direction: str):
        if direction == "up":
            self.pan_y = max(-self.MAX_PAN, self.pan_y - self.STEP)
        elif direction == "down":
            self.pan_y = min(self.MAX_PAN, self.pan_y + self.STEP)
        elif direction == "left":
            self.pan_x = max(-self.MAX_PAN, self.pan_x - self.STEP)
        elif direction == "right":
            self.pan_x = min(self.MAX_PAN, self.pan_x + self.STEP)
        elif direction == "zoom_in":
            self.zoom = min(self.MAX_ZOOM, self.zoom + self.ZOOM_STEP)
        elif direction == "zoom_out":
            self.zoom = max(self.MIN_ZOOM, self.zoom - self.ZOOM_STEP)
        elif direction == "reset":
            self.pan_x = 0
            self.pan_y = 0
            self.tilt = 0
            self.zoom = 1.0

    def get_state(self):
        # Added zoom to the returned state
        return self.pan_x, self.pan_y, self.tilt, self.zoom

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Simulate navigation by cropping a shifted/scaled window and resizing back.
        """
        h, w = frame.shape[:2]
        margin = self.MAX_PAN + 10

        # Pad frame so we can shift without going out of bounds
        padded = cv2.copyMakeBorder(
            frame, margin, margin, margin, margin,
            cv2.BORDER_REFLECT
        )

        # Calculate new crop dimensions based on zoom level
        new_h = int(h / self.zoom)
        new_w = int(w / self.zoom)

        # Calculate center point considering pan offsets
        center_y = margin + (h // 2) + self.pan_y
        center_x = margin + (w // 2) + self.pan_x

        # Calculate crop start points
        y1 = max(0, center_y - (new_h // 2))
        x1 = max(0, center_x - (new_w // 2))

        # Perform the crop
        cropped = padded[y1:y1 + new_h, x1:x1 + new_w]

        # Resize back to original size to create the zoom effect
        cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # Optional tilt (rotation)
        if self.tilt != 0:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), self.tilt, 1.0)
            cropped = cv2.warpAffine(cropped, M, (w, h))

        return cropped