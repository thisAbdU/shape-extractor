import cv2
import numpy as np
 
 
class ToolSegmentor:
    def __init__(self):
        # Mat blue: printed #00ADEF ink photographed under real lighting
        # HSV Hue ≈ 106 (cyan-blue). Allow ±16 for lighting/print variation.
        # S > 100, V > 60 to exclude dark shadows and near-neutral greys.
        self.lower_blue = np.array([90, 100,  60])
        self.upper_blue = np.array([122, 255, 255])
 
        # Padding (px) to step inside the blue boundary before cropping.
        # Avoids grabbing the edge-blur pixels right at the mat border.
        self._crop_pad = 15
 
    # ── Public API ────────────────────────────────────────────────────────────
 
    def segment_tool(self, image):
        """
        Isolate the tool from the cyan-blue mat surface.
 
        Key insight: the checkerboard border is NOT blue, so a naive
        bitwise_not of the blue mask makes the border appear as "tool" and
        connects to the actual tool forming one giant contour.
 
        Fix: detect the bounding box of the blue mat region first, crop to
        it, THEN segment.  Returns a full-size mask (same shape as `image`)
        with the tool white and everything else black.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
        # 1. Find inner blue boundary
        crop = self._find_blue_crop(hsv)          # (x1, y1, x2, y2)
        x1, y1, x2, y2 = crop
 
        # 2. Work only inside the blue area
        roi_img = image[y1:y2, x1:x2]
        roi_hsv = hsv[y1:y2, x1:x2]
 
        blue_mask_roi = cv2.inRange(roi_hsv, self.lower_blue, self.upper_blue)
        tool_mask_roi = cv2.bitwise_not(blue_mask_roi)
        tool_mask_roi = self._clean_mask(tool_mask_roi)
 
        # 3. Place the ROI mask back into a full-size canvas
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = tool_mask_roi
 
        # Store crop coords so extract_largest_contour can use them
        self._last_crop = crop
 
        return full_mask
 
    def extract_largest_contour(self, mask):
        """
        Extract the outermost contour of the tool from the cleaned mask.
        Raises a descriptive error if nothing meaningful is found.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError(
                "No contours found in tool mask.\n"
                "Check data/output/*_02_mask_raw.png — if the mat area is "
                "mostly white the HSV range needs tuning in segmentor.py."
            )
 
        largest = max(contours, key=cv2.contourArea)
 
        # Must be at least 0.1 % of the full mask area
        min_area = mask.shape[0] * mask.shape[1] * 0.001
        if cv2.contourArea(largest) < min_area:
            raise ValueError(
                f"Largest contour ({cv2.contourArea(largest):.0f} px²) is "
                f"smaller than the minimum ({min_area:.0f} px²) — "
                f"likely noise rather than a tool."
            )
 
        return largest
 
    def create_final_mask(self, image_shape, contour):
        """Clean binary mask containing only the filled tool contour."""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        return mask
 
    # ── Private helpers ───────────────────────────────────────────────────────
 
    def _find_blue_crop(self, hsv):
        """
        Find the bounding box of the blue mat region and return a padded
        inner crop (x1, y1, x2, y2).
 
        If no blue is found (e.g. REF mode image passed accidentally), falls
        back to the full image so the caller doesn't crash.
        """
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
 
        rows, cols = np.where(blue_mask > 0)
        if len(rows) == 0:
            h, w = hsv.shape[:2]
            return (0, 0, w, h)   # no blue found — use full image
 
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
          Open  → remove small speckles (scratches, grease marks)
          Close → fill small holes inside the tool body
          Open  → final pass to remove remaining thin noise
        """
        k_small  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
        k_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
 
        opened = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  k_small)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_medium)
        final  = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  k_small)
 
        return final
