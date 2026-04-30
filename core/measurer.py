import cv2
import numpy as np
from scipy.spatial import distance as dist

class MetrologyEngine:
    def __init__(self, pixels_per_mm):
        self.pixels_per_mm = pixels_per_mm

    @staticmethod
    def calibrate(reference_object_pixels, actual_mm=32.0):
        """Calculates pixels-to-metric ratio using the 32mm anchor."""
        return reference_object_pixels / actual_mm

    @staticmethod
    def detect_reference_object(image):
        """
        Detects the 32mm circular reference object in the rectified image.
        Returns the pixel diameter of the reference circle.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Use Hough Circle Transform to detect circular objects
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=100
        )
        
        if circles is None:
            raise ValueError("No circular reference object detected")
        
        circles = np.round(circles[0, :]).astype("int")
        
        # Find the circle closest to the expected size (32mm at our scale)
        # Assuming our rectified image has ~5 pixels per mm
        expected_radius_px = 32 * 5 / 2  # ~80 pixels
        best_circle = None
        min_diff = float('inf')
        
        for (x, y, r) in circles:
            diff = abs(r - expected_radius_px)
            if diff < min_diff:
                min_diff = diff
                best_circle = (x, y, r)
        
        if best_circle is None:
            raise ValueError("No suitable reference circle found")
        
        x, y, radius = best_circle
        diameter_px = radius * 2
        
        return diameter_px

    def measure_contour(self, contour):
        """Calculates real-world dimensions for a tool contour."""
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle = rect
        
        real_w = w / self.pixels_per_mm
        real_h = h / self.pixels_per_mm
        return real_w, real_h