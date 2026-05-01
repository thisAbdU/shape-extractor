"""
core/segmentor.py
=================
Dual-Stream Fusion segmentation engine.
 
Problem with HSV-only approach:
    Chrome/mirror-finish tools reflect the cyan mat colour back into the
    lens, creating large "holes" in the HSV mask where the tool surface
    looks blue.  Morphological Closing can fill small holes (up to ~kernel
    size) but fails on large specular patches covering several centimetres.
 
Solution — two parallel streams, fused:
 
    Stream A — HSV colour mask
        Finds the "colour blob": everything that is NOT the cyan mat.
        Good at detecting the bulk of the tool but has holes where chrome
        reflects the mat colour.
 
    Stream B — Canny edge map
        Finds all sharp boundaries in the image regardless of colour.
        Good at tracing the tool outline even over chrome reflections,
        but also picks up mat texture, scratches, and background noise.
 
    Fusion strategy:
        1. Dilate the Canny edges slightly to close small gaps in the outline.
        2. Flood-fill from the image corners (known background) on the
           INVERSE of the edge image → this gives a "background blob".
        3. Invert that to get a "foreground blob" from Canny alone.
        4. Union of HSV blob ∪ Canny blob → fills holes left by chrome
           while the HSV stream suppresses background noise that Canny
           would otherwise include.
        5. Morphological cleanup to smooth the final mask.
"""
 
import cv2
import numpy as np
 
 
class ToolSegmentor:
    def __init__(self):
        # Mat blue: printed #00ADEF photographed under real lighting.
        # HSV Hue ≈ 106, allow ±16 for print/lighting variation.
        self.lower_blue = np.array([90, 100,  60])
        self.upper_blue = np.array([122, 255, 255])
 
        # Pixels to step inside the detected blue boundary before cropping,
        # so border-blur pixels don't leak into the ROI.
        self._crop_pad = 15
 
        # Canny thresholds — lower = more edges (more sensitive to noise),
        # higher = fewer edges (may miss faint tool outlines).
        self._canny_low  = 30
        self._canny_high = 90
 
        # Dilation kernel applied to Canny edges before flood-fill,
        # to close small gaps in the edge outline.
        self._edge_close_px = 3
 
    # ── Public API ────────────────────────────────────────────────────────────
 
    def segment_tool(self, image):
        """
        Isolate the tool using Dual-Stream Fusion (HSV + Canny).
 
        Returns a full-size binary mask: 255 = tool, 0 = background.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
        # Find and crop to inner blue mat area (excludes checkerboard border)
        crop = self._find_blue_crop(hsv)
        x1, y1, x2, y2 = crop
 
        roi = image[y1:y2, x1:x2]
 
        # ── Stream A: HSV colour mask ─────────────────────────────────────
        roi_hsv       = hsv[y1:y2, x1:x2]
        blue_mask     = cv2.inRange(roi_hsv, self.lower_blue, self.upper_blue)
        hsv_tool_mask = cv2.bitwise_not(blue_mask)
 
        # ── Stream B: Canny edge → flood-fill foreground ──────────────────
        canny_tool_mask = self._canny_foreground(roi)
 
        # ── Fusion: union of both streams ─────────────────────────────────
        fused = cv2.bitwise_or(hsv_tool_mask, canny_tool_mask)
 
        # ── Cleanup ───────────────────────────────────────────────────────
        cleaned = self._clean_mask(fused)
 
        # Place ROI result back on a full-size canvas
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = cleaned
 
        self._last_crop = crop
        return full_mask
 
    def extract_largest_contour(self, mask):
        """
        Return the largest contour in the mask.
        Raises ValueError with a diagnostic message if nothing useful found.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError(
                "No contours found in tool mask.\n"
                "Check *_02_mask_raw.png — if the mat area is mostly white "
                "the HSV range needs tuning in segmentor.py."
            )
 
        largest  = max(contours, key=cv2.contourArea)
        min_area = mask.shape[0] * mask.shape[1] * 0.001
        if cv2.contourArea(largest) < min_area:
            raise ValueError(
                f"Largest contour ({cv2.contourArea(largest):.0f} px²) is "
                f"below minimum ({min_area:.0f} px²) — likely noise."
            )
 
        return largest
 
    def create_final_mask(self, image_shape, contour):
        """Filled binary mask from a single contour."""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        return mask
 
    # ── Private: Canny stream ─────────────────────────────────────────────────
 
    def _canny_foreground(self, roi):
        """
        Extract foreground from a ROI using Canny edges + flood-fill.
 
        Steps:
          1. Convert to greyscale and apply bilateral filter
             (smooths flat regions but preserves edges — better than
             Gaussian for this use case).
          2. Run Canny to get an edge map.
          3. Dilate edges to close small gaps in the tool outline.
          4. Flood-fill from all 4 corners (known background) on the
             INVERTED edge image → background region.
          5. Invert flood-fill result → foreground (tool) mask.
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
 
        # Bilateral filter: preserves tool edges, smooths uniform mat surface
        filtered = cv2.bilateralFilter(gray, d=9,
                                        sigmaColor=75, sigmaSpace=75)
 
        edges = cv2.Canny(filtered, self._canny_low, self._canny_high)
 
        # Dilate to close gaps in the outline
        k     = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (self._edge_close_px * 2 + 1,) * 2)
        edges = cv2.dilate(edges, k, iterations=1)
 
        # Flood-fill background from all 4 corners on inverted edge map
        h, w     = edges.shape
        inv_edges = cv2.bitwise_not(edges)
 
        # We need a mask 2px larger for floodFill
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        bg      = inv_edges.copy()
 
        for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            cv2.floodFill(bg, ff_mask, seed, 255,
                          loDiff=10, upDiff=10,
                          flags=cv2.FLOODFILL_FIXED_RANGE)
 
        # Background is now 255; foreground (tool) is darker
        background_mask = bg
        foreground_mask = cv2.bitwise_not(background_mask)
 
        return foreground_mask
 
    # ── Private: helpers ──────────────────────────────────────────────────────
 
    def _find_blue_crop(self, hsv):
        """
        Find the bounding box of the blue mat region and return a padded
        inner crop (x1, y1, x2, y2).  Falls back to the full image if no
        blue is detected (e.g. mat is flipped — caught upstream).
        """
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        rows, cols = np.where(blue_mask > 0)
 
        if len(rows) == 0:
            h, w = hsv.shape[:2]
            return (0, 0, w, h)
 
        pad = self._crop_pad
        h, w = hsv.shape[:2]
        x1 = max(0,     int(cols.min()) + pad)
        y1 = max(0,     int(rows.min()) + pad)
        x2 = min(w - 1, int(cols.max()) - pad)
        y2 = min(h - 1, int(rows.max()) - pad)
 
        return (x1, y1, x2, y2)
 
    def _clean_mask(self, mask):
        """
        Morphological cleanup:
          Open  → remove small speckles outside the tool
          Close → fill remaining holes inside the tool body
                  (larger kernel than before to handle bigger chrome patches)
          Open  → final denoising pass
        """
        k_small  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
        k_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
 
        opened = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  k_small)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_medium)
        final  = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  k_small)
 
        return final
