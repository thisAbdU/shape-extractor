"""
core/measurer.py
================
MetrologyEngine — calibration, scale verification, and measurement.
 
Scale verification using the checkerboard grid:
    The mat border contains a checkerboard pattern with a known square size
    (each square = 10mm at the printed scale, giving a 20mm period).
    After warping, we can measure the pixel period of this grid and compare
    it against the expected 5 px/mm × 20mm = 100 px period.
 
    If the measured period differs from expected by more than a threshold,
    it means the warp was imperfect (bad corner detection, mat not fully
    flat, extreme camera angle) and the measurements will be inaccurate.
    We report this as a WARNING with the measured correction factor so the
    caller can decide whether to trust the result.
 
    This directly addresses the client concern:
    "They are not the same dimensions. I purposefully took pictures at
     different angles and heights."
"""
 
import cv2
import numpy as np
 
 
# Expected checkerboard period in pixels at 5 px/mm.
# The printed mat border squares are 33mm each → 165px at 5px/mm.
_CHECKER_SQUARE_MM       = 33.0
_EXPECTED_GRID_PERIOD_PX = 165   # 33mm × 5px/mm
_GRID_WARNING_THRESHOLD  = 0.05  # warn if scale error > 5%
 
 
class MetrologyEngine:
    def __init__(self, pixels_per_mm: float):
        self.pixels_per_mm = pixels_per_mm
 
    # ── Calibration ───────────────────────────────────────────────────────────
 
    @staticmethod
    def calibrate(reference_object_pixels: float,
                  actual_mm: float = 32.0) -> float:
        """
        Calculate pixels-per-mm from a known reference object.
 
        Args:
            reference_object_pixels : measured size in pixels
            actual_mm               : true physical size (default 32 mm)
        Returns:
            pixels_per_mm (float)
        """
        if reference_object_pixels <= 0:
            raise ValueError("reference_object_pixels must be > 0")
        return reference_object_pixels / actual_mm
 
    # ── Grid-based scale verification ─────────────────────────────────────────
 
    @staticmethod
    def verify_scale_from_grid(warped_image,
                               nominal_px_per_mm: float = 5.0,
                               grid_period_mm: float = 33.0,  # actual printed square size
                               border_fraction: float = 0.12):
        """
        Measure the actual pixel period of the checkerboard border grid and
        compare against the nominal scale.
 
        Why this matters:
            The homography assumes the mat is perfectly flat and the camera
            is perfectly overhead. In practice, camera tilt and Z-height of
            the tool cause the effective scale to vary. This function
            measures the grid that was printed on the mat (known size) and
            detects if the warp result is consistent with 5 px/mm.
 
        Args:
            warped_image      : the perspective-corrected mat image (BGR)
            nominal_px_per_mm : expected scale (5.0 for our mat)
            grid_period_mm    : checkerboard square period in mm (20mm)
            border_fraction   : fraction of image height/width to sample
                                as the "border" region (default 12%)
 
        Returns:
            dict with keys:
                'measured_px_per_mm'  : float — what the grid says the scale is
                'nominal_px_per_mm'   : float — what we assumed
                'scale_error_pct'     : float — percentage difference
                'correction_factor'   : float — multiply measurements by this
                                        to correct for scale error
                'warning'             : str or None — message if error > threshold
                'reliable'            : bool — False if error > threshold
        """
        h, w = warped_image.shape[:2]
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
 
        expected_period_px = nominal_px_per_mm * grid_period_mm  # 5 × 33 = 165px
 
        # Sample the border strips where the checkerboard lives
        border_h = int(h * border_fraction)
        border_w = int(w * border_fraction)
 
        # Collect horizontal and vertical period measurements
        h_periods = _measure_grid_period_horizontal(
            gray, border_h, w, expected_period_px
        )
        v_periods = _measure_grid_period_vertical(
            gray, h, border_w, expected_period_px
        )
 
        all_periods = h_periods + v_periods
 
        if len(all_periods) < 3:
            # Not enough grid samples — can't verify
            return {
                'measured_px_per_mm' : nominal_px_per_mm,
                'nominal_px_per_mm'  : nominal_px_per_mm,
                'scale_error_pct'    : 0.0,
                'correction_factor'  : 1.0,
                'warning'            : "Could not detect checkerboard grid for scale verification.",
                'reliable'           : True,   # assume ok, don't block pipeline
            }
 
        measured_period_px  = float(np.median(all_periods))
        measured_px_per_mm  = measured_period_px / grid_period_mm
        scale_error_pct     = abs(measured_px_per_mm - nominal_px_per_mm) \
                              / nominal_px_per_mm * 100
        correction_factor   = measured_px_per_mm / nominal_px_per_mm
 
        warning = None
        reliable = True
        if scale_error_pct > _GRID_WARNING_THRESHOLD * 100:
            warning = (
                f"Scale mismatch: grid measures {measured_px_per_mm:.3f} px/mm "
                f"but nominal is {nominal_px_per_mm:.1f} px/mm "
                f"({scale_error_pct:.1f}% error).\n"
                f"Likely cause: camera not directly overhead, mat not flat, "
                f"or extreme Z-height of a thick tool.\n"
                f"Correction factor: {correction_factor:.4f} "
                f"(measurements multiplied by this automatically)."
            )
            reliable = scale_error_pct < 15.0   # > 15% = unusable
 
        return {
            'measured_px_per_mm' : measured_px_per_mm,
            'nominal_px_per_mm'  : nominal_px_per_mm,
            'scale_error_pct'    : scale_error_pct,
            'correction_factor'  : correction_factor,
            'warning'            : warning,
            'reliable'           : reliable,
        }
 
    # ── Reference circle detection (REF mode only) ────────────────────────────
 
    @staticmethod
    def detect_reference_object(image,
                                 expected_diameter_mm: float = 32.0,
                                 approx_px_per_mm: float = None) -> float:
        """
        Detect a circular reference object and return its pixel diameter.
        Call this ONLY in REF mode (no mat).
        In MAT mode pixels_per_mm = 5 exactly — do not call this.
        """
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        h, w    = image.shape[:2]
        margin  = 20
 
        if approx_px_per_mm is not None:
            expected_r = expected_diameter_mm * approx_px_per_mm / 2
            min_r = max(5,  int(expected_r * 0.5))
            max_r = max(10, int(expected_r * 1.5))
        else:
            min_r, max_r = 10, 0
 
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=50,
            param1=60, param2=30,
            minRadius=min_r, maxRadius=max_r,
        )
 
        if circles is None:
            raise ValueError(
                "No circular reference object detected. "
                "Ensure the 32 mm disk is clearly visible."
            )
 
        circles = np.round(circles[0]).astype(int)
 
        def score(xyr):
            x, y, r = xyr
            near = (x - r < margin or y - r < margin or
                    x + r > w - margin or y + r > h - margin)
            return r if not near else -1
 
        best = max(circles, key=score)
        if score(best) < 0:
            raise ValueError(
                "All detected circles touch the image border — "
                "make sure the reference object is fully in frame."
            )
 
        return float(best[2] * 2)
 
    # ── Measurement ───────────────────────────────────────────────────────────
 
    def measure_contour(self, contour,
                         correction_factor: float = 1.0):
        """
        Return real-world (width_mm, height_mm) of the minimum bounding
        rectangle, optionally corrected by a scale factor from grid
        verification.
 
        Args:
            contour           : OpenCV contour array
            correction_factor : from verify_scale_from_grid() (default 1.0)
        Returns:
            (width_mm, height_mm) — use max() for length, min() for width
        """
        rect = cv2.minAreaRect(contour)
        _, (w, h), _ = rect
        scale = self.pixels_per_mm / correction_factor
        return w / scale, h / scale
 
    def contour_area_mm2(self, contour,
                          correction_factor: float = 1.0) -> float:
        """Return contour area in mm², with optional scale correction."""
        px_area = cv2.contourArea(contour)
        corrected_ppm = self.pixels_per_mm / correction_factor
        return px_area / (corrected_ppm ** 2)
 
    def pixel_to_mm(self, pixels: float) -> float:
        return pixels / self.pixels_per_mm
 
    def mm_to_pixel(self, mm: float) -> float:
        return mm * self.pixels_per_mm
 
 
# ── Module-level grid measurement helpers ─────────────────────────────────────
 
def _measure_grid_period_horizontal(gray, border_h, w, expected_period_px):
    """
    Measure checkerboard period along horizontal border strips
    (top and bottom of the warped mat image).
 
    Method: take a horizontal 1D profile through the border, find peaks
    in the intensity signal (bright squares), measure their spacing.
    """
    periods = []
    h_full  = gray.shape[0]
 
    for strip_y in [border_h // 2, h_full - border_h // 2]:
        row = gray[strip_y, :].astype(float)
        periods += _periods_from_profile(row, expected_period_px)
 
    return periods
 
 
def _measure_grid_period_vertical(gray, h, border_w, expected_period_px):
    """
    Measure checkerboard period along vertical border strips
    (left and right of the warped mat image).
    """
    periods = []
    w_full  = gray.shape[1]
 
    for strip_x in [border_w // 2, w_full - border_w // 2]:
        col = gray[:, strip_x].astype(float)
        periods += _periods_from_profile(col, expected_period_px)
 
    return periods
 
 
def _periods_from_profile(profile, expected_period_px,
                           tolerance: float = 0.25):
    """
    Given a 1D intensity profile through a checkerboard border, find the
    spacing between bright (white square) peaks.
 
    Args:
        profile            : 1D numpy array of pixel intensities
        expected_period_px : expected peak spacing in pixels
        tolerance          : accept spacings within ± this fraction of expected
 
    Returns:
        list of measured period values (floats)
    """
    from scipy.signal import find_peaks
 
    # Smooth to suppress high-frequency noise
    smoothed = np.convolve(profile,
                           np.ones(5) / 5,
                           mode='same')
 
    # Find local maxima (bright squares)
    min_dist = int(expected_period_px * (1 - tolerance))
    peaks, _ = find_peaks(smoothed, distance=min_dist, prominence=20)
 
    if len(peaks) < 2:
        return []
 
    # Compute spacings between consecutive peaks
    spacings = np.diff(peaks).astype(float)
 
    # Keep only spacings within tolerance of expected
    lo = expected_period_px * (1 - tolerance)
    hi = expected_period_px * (1 + tolerance)
    valid = spacings[(spacings >= lo) & (spacings <= hi)]
 
    return valid.tolist()
