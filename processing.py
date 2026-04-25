import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def apply_noise_reduction(image, method="bilateral"):
    """
    Applies noise reduction to the image.
    Bilateral filter is preferred as it reduces noise while preserving edges.
    """
    if method == "bilateral":
        # d: Diameter of each pixel neighborhood
        # sigmaColor: Value of \sigma in the color space
        # sigmaSpace: Value of \sigma in the coordinate space
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    return image

def apply_contrast_enhancement(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))
    
    # Convert image from LAB Color model to BGR color space
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def extract_color_feature(image):
    """
    Extracts the dominant color / highlights specific color ranges (e.g., redness for bleeding/inflammation).
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for red color
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Threshold the HSV image
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)
    return res

def extract_shape_feature(image):
    """
    Extracts shape features using Canny edge detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    # Convert edges to 3 channel to display alongside original image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_colored

def extract_texture_feature(image):
    """
    Extracts texture feature (GLCM correlation map representation).
    For visualization, we apply a local entropy filter or display Gabor responses.
    Here we use a fast variance filter as a simple texture visualization.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate local variance (a measure of texture)
    mean = cv2.blur(gray, (5,5))
    sq_mean = cv2.blur(gray**2, (5,5))
    variance = sq_mean - mean**2
    
    # Normalize for visualization
    variance = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)
    variance = np.uint8(variance)
    
    # Apply a colormap for better visualization
    texture_map = cv2.applyColorMap(variance, cv2.COLORMAP_JET)
    return texture_map
