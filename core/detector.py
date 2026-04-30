import cv2
import numpy as np
import os

def get_top_down_view(image, corners, target_width_mm=600, target_height_mm=200):
    """Performs perspective transform (homography) to flatten the mat."""
    scale = 5 
    dst = np.array([
        [0, 0],
        [target_width_mm * scale, 0],
        [target_width_mm * scale, target_height_mm * scale],
        [0, target_height_mm * scale]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (int(target_width_mm * scale), int(target_height_mm * scale)))
    return warped

def detect_mat_corners(image):
    """
    Robust Detection Pipeline:
    1. ArUco with gamma correction and aggressive adaptive thresholding
    2. Fallback to mat contour detection against dark background
    """
    os.makedirs("data/output", exist_ok=True)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try ArUco with multiple preprocessing approaches
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 80
    parameters.adaptiveThreshWinSizeStep = 5
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(image)
    
    # Try gamma correction if needed
    if ids is None or len(ids) < 4:
        gamma = 1.5
        gamma_corrected = np.array(255 * (gray / 255) ** gamma, dtype='uint8')
        corners, ids, _ = detector.detectMarkers(gamma_corrected)
    
    # Try adaptive thresholding if still needed
    if ids is None or len(ids) < 4:
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        corners, ids, _ = detector.detectMarkers(adaptive)

    if ids is not None and len(ids) >= 4:
        centers = np.array([np.mean(c[0], axis=0) for c in corners])
    else:
        centers = detect_mat_by_contour(gray)
    
    # 3. Geometric Sorting (TL, TR, BR, BL)
    s = centers.sum(axis=1)
    diff = np.diff(centers, axis=1)
    
    ordered_centers = np.zeros((4, 2), dtype="float32")
    ordered_centers[0] = centers[np.argmin(s)]       # Top-Left
    ordered_centers[2] = centers[np.argmax(s)]       # Bottom-Right
    ordered_centers[1] = centers[np.argmin(diff)]    # Top-Right
    ordered_centers[3] = centers[np.argmax(diff)]    # Bottom-Left
    
    return ordered_centers

def detect_mat_by_contour(gray):
    """
    Fallback method: Find the white mat by detecting its outer contour
    against the darker table surface.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Use Otsu's thresholding to separate white mat from dark background
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in mat detection fallback")
    
    # Find the largest rectangular contour (should be the mat)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate contour to a quadrilateral
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we don't get exactly 4 corners, try different epsilon values
    if len(approx) != 4:
        for epsilon_factor in [0.01, 0.03, 0.04, 0.05]:
            epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            if len(approx) == 4:
                break
        
        # If still not 4 corners, use convex hull to get 4 extreme points
        if len(approx) != 4:
            hull = cv2.convexHull(largest_contour)
            # Find the 4 extreme points of the convex hull
            hull_points = hull.reshape(-1, 2)
            
            # Find top-left, top-right, bottom-right, bottom-left
            s = hull_points.sum(axis=1)
            diff = np.diff(hull_points, axis=1)
            
            ordered_corners = np.zeros((4, 2), dtype="float32")
            ordered_corners[0] = hull_points[np.argmin(s)]       # Top-Left
            ordered_corners[2] = hull_points[np.argmax(s)]       # Bottom-Right
            ordered_corners[1] = hull_points[np.argmin(diff)]    # Top-Right
            ordered_corners[3] = hull_points[np.argmax(diff)]    # Bottom-Left
            
            return ordered_corners
    
    if len(approx) != 4:
        raise ValueError(f"Expected 4 corners, got {len(approx)}")
    
        
    # Return the 4 corner points
    return approx.reshape(4, 2).astype("float32")