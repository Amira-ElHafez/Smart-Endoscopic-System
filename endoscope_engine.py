import cv2
import numpy as np

class EndoscopeEngine:
    def __init__(self, media_path=None):
        self.base_image = None
        self.display_image = None
        self.brightness = 1.0  # 1.0 is normal, < 1.0 is darker, > 1.0 is brighter
        
        # Navigation state
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Processing state
        self.processing_mode = "None"
        
        if media_path:
            self.load_media(media_path)
            
    def load_media(self, path):
        # We assume an image for now, but can be extended to VideoCapture
        self.base_image = cv2.imread(path)
        if self.base_image is None:
            print(f"Error: Could not load media from {path}")
            return False
        return True
        
    def set_brightness(self, value):
        """ Set light intensity level (e.g., 0.1 to 3.0) """
        self.brightness = value
        
    def navigate(self, dx, dy, dzoom):
        """ Update panning and zooming """
        self.pan_x += dx
        self.pan_y += dy
        
        # Limit zoom
        self.zoom += dzoom
        if self.zoom < 1.0:
            self.zoom = 1.0
        if self.zoom > 5.0:
            self.zoom = 5.0
            
    def get_current_frame(self):
        if self.base_image is None:
            return None
            
        h, w = self.base_image.shape[:2]
        
        # Center of the image
        cx, cy = w // 2, h // 2
        
        # Apply Zoom and Pan using affine transformation
        # Translation matrix
        M = np.float32([
            [self.zoom, 0, cx * (1 - self.zoom) + self.pan_x],
            [0, self.zoom, cy * (1 - self.zoom) + self.pan_y]
        ])
        
        # Warp the image to simulate camera moving and zooming
        transformed = cv2.warpAffine(self.base_image, M, (w, h))
        
        # Apply Illumination (Brightness)
        # Convert to HSV, adjust V channel
        if self.brightness != 1.0:
            hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV).astype("float32")
            (h_chan, s_chan, v_chan) = cv2.split(hsv)
            v_chan = v_chan * self.brightness
            v_chan = np.clip(v_chan, 0, 255)
            hsv = cv2.merge([h_chan, s_chan, v_chan])
            transformed = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
            
        self.display_image = transformed
        return transformed

