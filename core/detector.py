import cv2
import numpy as np
import os
 
 
def get_top_down_view(image, corners, target_width_mm=600, target_height_mm=200):
    """
    Perspective transform (homography) to flatten the mat to a canonical
    top-down view.  Output is target_width_mm*5 × target_height_mm*5 pixels
    (scale = 5 px/mm, exact by construction).
    """
    scale = 5
    dst = np.array([
        [0,                         0],
        [target_width_mm  * scale,  0],
        [target_width_mm  * scale,  target_height_mm * scale],
        [0,                         target_height_mm * scale],
    ], dtype="float32")
 
    M      = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(
        image, M,
        (int(target_width_mm * scale), int(target_height_mm * scale))
    )
    return warped
 
 
def detect_mat_corners(image):
    """
    Robust 4-corner detection pipeline:
 
    Stage 1 — ArUco (DICT_4X4_50, IDs 0-3):
        Tries raw image → gamma-corrected → adaptive-threshold.
        If only 3 markers are found, the 4th corner is inferred geometrically
        (the mat is a known rectangle, so the missing corner = parallelogram rule).
 
    Stage 2 — Fallback contour detection:
        Finds the largest rectangular contour (the white mat border).
 
    Returns corners ordered [TL, TR, BR, BL] as float32 (4,2) array.
    """
    os.makedirs("data/output", exist_ok=True)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
 
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin  = 3
    params.adaptiveThreshWinSizeMax  = 80
    params.adaptiveThreshWinSizeStep = 5
    params.cornerRefinementMethod    = cv2.aruco.CORNER_REFINE_SUBPIX
 
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
 
    def try_detect(src):
        return detector.detectMarkers(src)
 
    # --- Stage 1a: raw image ---
    corners_raw, ids, _ = try_detect(image)
 
    # --- Stage 1b: gamma correction ---
    if ids is None or len(ids) < 3:
        gamma    = 1.5
        gamma_lut = np.array([((i / 255.0) ** gamma) * 255
                               for i in range(256)], dtype="uint8")
        gamma_img = cv2.LUT(gray, gamma_lut)
        corners_raw, ids, _ = try_detect(gamma_img)
 
    # --- Stage 1c: adaptive threshold ---
    if ids is None or len(ids) < 3:
        adaptive    = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        corners_raw, ids, _ = try_detect(adaptive)
 
    # ── ArUco path: 3 or 4 markers found ──────────────────────────────────
    if ids is not None and len(ids) >= 3:
        # Build a dict: marker_id → center point.
        # Filter to ONLY the 4 expected mat IDs — anything else is a false
        # positive (tool texture or background decoded as a random marker).
        VALID_IDS = {0, 1, 2, 3}
        id_to_center = {}
        for c, id_ in zip(corners_raw, ids):
            mid = int(id_[0])
            if mid in VALID_IDS:
                id_to_center[mid] = np.mean(c[0], axis=0)
 
        spurious = [int(i[0]) for i in ids if int(i[0]) not in VALID_IDS]
        if spurious:
            print(f"  [detector] Ignoring spurious marker IDs: {spurious}")
 
        print(f"  [detector] ArUco found IDs: {sorted(id_to_center.keys())}")
 
        # If spurious IDs reduced valid count below 3, fall back to contour
        if len(id_to_center) < 3:
            print("  [detector] Fewer than 3 valid IDs after filtering — "
                  "falling back to contour detection")
            return _detect_mat_by_contour(gray)
 
        # Mat corner assignment by ID (matches the PDF design):
        #   ID 3 = Top-Left
        #   ID 0 = Top-Right
        #   ID 1 = Bottom-Right
        #   ID 2 = Bottom-Left  ← often fails detection
        id_map = {3: "TL", 0: "TR", 1: "BR", 2: "BL"}
 
        if len(id_to_center) == 4:
            # All 4 found — straightforward
            tl = id_to_center[3]
            tr = id_to_center[0]
            br = id_to_center[1]
            bl = id_to_center[2]
        else:
            # 3 found — infer the missing one geometrically.
            # For a parallelogram: missing = opposite_a + opposite_b - opposite_c
            # (i.e. the 4th vertex of a parallelogram defined by the other 3)
            present = set(id_to_center.keys())
            missing = ({0, 1, 2, 3} - present).pop()
 
            pts = id_to_center.copy()
 
            if missing == 2:   # BL missing → BL = TL + BR - TR
                pts[2] = pts[3] + pts[1] - pts[0]
            elif missing == 3: # TL missing → TL = TR + BL - BR
                pts[3] = pts[0] + pts[2] - pts[1]
            elif missing == 0: # TR missing → TR = TL + BR - BL
                pts[0] = pts[3] + pts[1] - pts[2]
            elif missing == 1: # BR missing → BR = TR + BL - TL
                pts[1] = pts[0] + pts[2] - pts[3]
 
            print(f"  [detector] Inferred missing marker ID {missing} "
                  f"at ({pts[missing][0]:.0f}, {pts[missing][1]:.0f})")
 
            tl, tr, br, bl = pts[3], pts[0], pts[1], pts[2]
 
        ordered = np.array([tl, tr, br, bl], dtype="float32")
        return ordered
 
    # ── Stage 2: contour fallback ──────────────────────────────────────────
    print("  [detector] ArUco failed — falling back to contour detection")
    return _detect_mat_by_contour(gray)
 
 
def _detect_mat_by_contour(gray):
    """
    Find the white mat border by contour against a darker surface.
    Returns [TL, TR, BR, BL] float32 array.
    """
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mat detection fallback")
 
    largest = max(contours, key=cv2.contourArea)
 
    # Try to approximate to exactly 4 corners
    for eps_factor in [0.02, 0.01, 0.03, 0.04, 0.05]:
        epsilon = eps_factor * cv2.arcLength(largest, True)
        approx  = cv2.approxPolyDP(largest, epsilon, True)
        if len(approx) == 4:
            break
 
    if len(approx) != 4:
        # Fall back to convex hull extreme points
        hull   = cv2.convexHull(largest).reshape(-1, 2)
        approx = _four_extremes(hull)
    else:
        approx = approx.reshape(4, 2).astype("float32")
 
    return _sort_corners(approx)
 
 
def _four_extremes(pts):
    """Return the 4 extreme points of a point cloud as [TL,TR,BR,BL]."""
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    out  = np.array([
        pts[np.argmin(s)],    # TL: smallest x+y
        pts[np.argmin(diff)], # TR: smallest y-x
        pts[np.argmax(s)],    # BR: largest x+y
        pts[np.argmax(diff)], # BL: largest y-x
    ], dtype="float32")
    return out
 
 
def _sort_corners(pts):
    """Sort any 4-point array into [TL, TR, BR, BL] order."""
    pts  = pts.astype("float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype="float32")
