import cv2
import numpy as np
 
 
class MetrologyEngine:
    def __init__(self, pixels_per_mm: float):
        self.pixels_per_mm = pixels_per_mm
 
    # ── Calibration ──────────────────────────────────────────────────────────
 
    @staticmethod
    def calibrate(reference_object_pixels: float, actual_mm: float = 32.0) -> float:
        """
        Calculate pixels-per-mm ratio from a known reference object.
 
        Args:
            reference_object_pixels: measured size of the reference in pixels
            actual_mm:               true size in millimetres (default 32 mm)
        Returns:
            pixels_per_mm ratio (float)
        """
        if reference_object_pixels <= 0:
            raise ValueError("reference_object_pixels must be > 0")
        return reference_object_pixels / actual_mm
 
    @staticmethod
    def detect_reference_object(image, expected_diameter_mm: float = 32.0,
                                 approx_px_per_mm: float = None) -> float:
        """
        Detect a circular reference object and return its pixel diameter.
 
        This should ONLY be called in REF mode (no mat).
        In MAT mode, pixels_per_mm = 5 exactly — do not call this.
 
        Args:
            image:                 BGR image containing the reference circle
            expected_diameter_mm:  true diameter of the reference (default 32)
            approx_px_per_mm:      rough estimate of scale if available
                                   (helps constrain Hough radius bounds)
        Returns:
            diameter in pixels of the best matching circle
        """
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
 
        h, w = image.shape[:2]
        margin = 20
 
        # Set Hough radius bounds
        if approx_px_per_mm is not None:
            expected_r = expected_diameter_mm * approx_px_per_mm / 2
            min_r = max(5,  int(expected_r * 0.5))
            max_r = max(10, int(expected_r * 1.5))
        else:
            min_r, max_r = 10, 0   # 0 = no upper limit
 
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=60,
            param2=30,
            minRadius=min_r,
            maxRadius=max_r,
        )
 
        if circles is None:
            raise ValueError(
                "No circular reference object detected.  "
                "Make sure the 32 mm coin/disk is clearly visible "
                "and contrasts with the background."
            )
 
        circles = np.round(circles[0]).astype(int)
 
        # Score: prefer circles away from the border
        def score(xyr):
            x, y, r = xyr
            near = (x - r < margin or y - r < margin or
                    x + r > w - margin or y + r > h - margin)
            return r if not near else -1
 
        best    = max(circles, key=score)
        cx, cy, cr = best
 
        if score(best) < 0:
            raise ValueError(
                "All detected circles touch the image border — "
                "make sure the reference object is fully in frame."
            )
 
        return float(cr * 2)   # diameter in pixels
 
    # ── Measurement ──────────────────────────────────────────────────────────
 
    def measure_contour(self, contour):
        """
        Return real-world (width_mm, height_mm) of the minimum bounding rectangle
        of a contour.
 
        Note: minAreaRect can orient either way.  The caller should use
        max(w,h) as length and min(w,h) as width for consistent reporting.
        """
        rect            = cv2.minAreaRect(contour)
        (_, _), (w, h), _ = rect
        return w / self.pixels_per_mm, h / self.pixels_per_mm
 
    def contour_area_mm2(self, contour) -> float:
        """Return contour area in mm²."""
        return cv2.contourArea(contour) / (self.pixels_per_mm ** 2)
 
    def pixel_to_mm(self, pixels: float) -> float:
        return pixels / self.pixels_per_mm
 
    def mm_to_pixel(self, mm: float) -> float:
        return mm * self.pixels_per_mm
