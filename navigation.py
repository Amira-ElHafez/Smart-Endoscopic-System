"""
Navigation Controller — Simulates insertion tube pan / tilt movement.
Uses pan_x, pan_y offsets and applies a crop+resize to the live frame.
"""

import cv2
import numpy as np


class NavigationController:
    STEP = 15      # pixels per keypress
    MAX_PAN = 120  # maximum offset from center
    TILT_STEP = 5
    MAX_TILT = 30

    def __init__(self):
        self.pan_x = 0
        self.pan_y = 0
        self.tilt = 0

    def move(self, direction: str):
        if direction == "up":
            self.pan_y = max(-self.MAX_PAN, self.pan_y - self.STEP)
        elif direction == "down":
            self.pan_y = min(self.MAX_PAN, self.pan_y + self.STEP)
        elif direction == "left":
            self.pan_x = max(-self.MAX_PAN, self.pan_x - self.STEP)
        elif direction == "right":
            self.pan_x = min(self.MAX_PAN, self.pan_x + self.STEP)
        elif direction == "reset":
            self.pan_x = 0
            self.pan_y = 0
            self.tilt = 0

    def get_state(self):
        return self.pan_x, self.pan_y, self.tilt

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Simulate navigation by cropping a shifted window and resizing back.
        Gives the illusion of camera movement inside the body.
        """
        h, w = frame.shape[:2]
        margin = self.MAX_PAN + 10

        # Pad frame so we can shift without going out of bounds
        padded = cv2.copyMakeBorder(
            frame, margin, margin, margin, margin,
            cv2.BORDER_REFLECT
        )

        # Crop shifted region
        cy = margin + self.pan_y
        cx = margin + self.pan_x
        cropped = padded[cy:cy + h, cx:cx + w]

        # Optional tilt (rotation)
        if self.tilt != 0:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), self.tilt, 1.0)
            cropped = cv2.warpAffine(cropped, M, (w, h))

        return cropped