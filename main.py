"""
main.py — Shape Extractor Pipeline
=====================================
 
Two modes, auto-detected or forced via CLI:
 
  MAT MODE  (--mode mat):
    - Image contains the 600x200mm calibrated mat
    - Perspective-warp the mat to a flat top-down view
    - pixels_per_mm is KNOWN (scale=5) — no circle detection needed
    - Segment tool from the blue mat surface via HSV masking
 
  REF MODE  (--mode ref):
    - Image has NO mat, just a tool + 32mm reference circle on any surface
    - Detect the 32mm circle to establish pixels_per_mm
    - Segment tool by excluding the reference circle + background
 
Usage:
    python3 main.py                              # batch data/input/*.HEIC
    python3 main.py data/input/IMG_01.HEIC
    python3 main.py data/input/IMG_01.HEIC --mode ref
    python3 main.py data/input/IMG_01.HEIC --mode mat
"""
 
import argparse
import glob
import os
 
import cv2
import numpy as np
 
from core.processor import load_image
from core.detector  import detect_mat_corners, get_top_down_view
from core.measurer  import MetrologyEngine
from core.segmentor import ToolSegmentor
from core.exporter  import SVGExporter, DXFExporter
 
# ── Constants ──────────────────────────────────────────────────────────────────
MAT_WIDTH_MM    = 600
MAT_HEIGHT_MM   = 200
# get_top_down_view uses scale=5 internally, so the warped image is
# 3000 × 1000 px → exactly 5 px/mm.  This is NOT an estimate.
MAT_SCALE_PX_MM = 5
REF_DIAM_MM     = 32.0
OUT_DIR         = "data/output"
 
 
# ── Diagnostic saves ───────────────────────────────────────────────────────────
def save_debug(name: str, img: np.ndarray, label: str = ""):
    path = os.path.join(OUT_DIR, name)
    out  = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if label:
        cv2.putText(out, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite(path, out)
    print(f"    [dbg] {path}")
 
 
# ── Mode auto-detection ────────────────────────────────────────────────────────
def auto_detect_mode(image: np.ndarray) -> str:
    # Quick ArUco probe
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector   = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    _, ids, _  = detector.detectMarkers(image)
    if ids is not None and len(ids) >= 3:
        # The mat's BL marker (ID 2) reliably fails detection due to its
        # print-resolution pattern — 3 markers is sufficient to confirm mat mode.
        # detector.py will infer the 4th corner geometrically.
        print(f"  [auto] {len(ids)} ArUco markers found → MAT mode")
        return "mat"
 
    # Large-rectangle probe
    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _   = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest  = max(cnts, key=cv2.contourArea)
        img_area = image.shape[0] * image.shape[1]
        epsilon  = 0.02 * cv2.arcLength(largest, True)
        approx   = cv2.approxPolyDP(largest, epsilon, True)
        fill_pct = cv2.contourArea(largest) / img_area * 100
        if len(approx) == 4 and fill_pct > 30:
            print(f"  [auto] Large quad ({fill_pct:.0f}% of frame) → MAT mode")
            return "mat"
 
    print("  [auto] No mat detected → REF mode")
    return "ref"
 
 
# ── Mat orientation check ─────────────────────────────────────────────────────
def _check_mat_orientation(warped: np.ndarray, base_name: str):
    """
    After warping, verify the blue mat surface is actually visible.
    If less than 5% of the warped image is the expected blue colour the mat
    is almost certainly flipped face-down (showing the brown cardboard back).
    Raises a clear error so the user knows exactly what went wrong instead of
    silently producing nonsense measurements.
    """
    hsv        = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 100,  60])
    upper_blue = np.array([122, 255, 255])
    blue_mask  = cv2.inRange(hsv, lower_blue, upper_blue)
 
    total_px   = warped.shape[0] * warped.shape[1]
    blue_px    = int(blue_mask.sum() // 255)
    blue_pct   = blue_px / total_px * 100
 
    print(f"  [mat] Blue surface coverage: {blue_pct:.1f}% of warped image")
 
    if blue_pct < 5.0:
        # Save an annotated copy so the user can see what the pipeline saw
        annotated = warped.copy()
        cv2.putText(annotated, "MAT FLIPPED? No blue detected",
                    (30, warped.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5, cv2.LINE_AA)
        save_debug(f"{base_name}_01_warped_FLIPPED.png", annotated)
        raise ValueError(
            f"Only {blue_pct:.1f}% of the warped image is blue "
            f"(expected >5%%).\n"
            f"Possible causes:\n"
            f"  1. Mat is FACE-DOWN — flip it so the blue side faces up.\n"
            f"  2. Mat is not in frame — make sure all 4 corners are visible.\n"
            f"  3. Extreme lighting — check {base_name}_01_warped.png."
        )
 
 
# ── MAT pipeline ───────────────────────────────────────────────────────────────
def run_mat_pipeline(img: np.ndarray, base_name: str):
    """
    Pipeline for photos taken on the 600×200 mm calibration mat.
 
    The warp target is defined by us (3000 × 1000 px for a 600×200 mm mat),
    so pixels_per_mm = 5 exactly — no reference circle detection needed here.
    """
    # 1. Perspective rectification
    print("  [mat] Detecting mat corners …")
    corners = detect_mat_corners(img)
    warped  = get_top_down_view(img, corners,
                                target_width_mm=MAT_WIDTH_MM,
                                target_height_mm=MAT_HEIGHT_MM)
    save_debug(f"{base_name}_01_warped.png", warped, "warped mat")
 
    # 1b. Sanity-check: is the blue mat surface actually visible?
    _check_mat_orientation(warped, base_name)
 
    pixels_per_mm = MAT_SCALE_PX_MM          # exact — no estimation needed
    print(f"  [mat] Scale: {pixels_per_mm} px/mm (exact, from warp definition)")
 
    # 2. Segment tool from blue surface
    print("  [mat] Segmenting tool from mat surface …")
    segmentor    = ToolSegmentor()
    tool_mask    = segmentor.segment_tool(warped)
    save_debug(f"{base_name}_02_mask_raw.png", tool_mask, "raw mask")
 
    # Warn if mask suspiciously large (likely blue range mismatch)
    cnts, _ = cv2.findContours(tool_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        mat_area = warped.shape[0] * warped.shape[1]
        lg_area  = cv2.contourArea(max(cnts, key=cv2.contourArea))
        pct      = lg_area / mat_area * 100
        print(f"  [mat] Largest mask region: {pct:.1f}% of mat")
        if pct > 80:
            print("  [WARN] Mask covers >80% — the blue HSV range in ToolSegmentor")
            print("         probably doesn't match your mat colour.")
            print("         Open _02_mask_raw.png to inspect, then tune")
            print("         lower_blue / upper_blue in core/segmentor.py.")
 
    tool_contour = segmentor.extract_largest_contour(tool_mask)
    final_mask   = segmentor.create_final_mask(warped.shape, tool_contour)
    save_debug(f"{base_name}_03_mask_final.png", final_mask, "final mask")
 
    overlay = warped.copy()
    cv2.drawContours(overlay, [tool_contour], -1, (0, 255, 0), 6)
    save_debug(f"{base_name}_04_contour.png", overlay, "contour overlay")
 
    _measure_and_export(tool_contour, final_mask,
                        pixels_per_mm, base_name,
                        canvas_w_mm=MAT_WIDTH_MM, canvas_h_mm=MAT_HEIGHT_MM)
 
 
# ── REF pipeline ───────────────────────────────────────────────────────────────
def run_ref_pipeline(img: np.ndarray, base_name: str):
    """
    Pipeline for photos WITHOUT a mat — a 32 mm circle provides the scale.
    """
    h, w = img.shape[:2]
 
    # 1. Detect reference circle (very permissive — no prior scale knowledge)
    print("  [ref] Detecting 32 mm reference circle …")
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
 
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=60,
        param2=30,
        minRadius=10,
        maxRadius=0,       # no upper limit — we don't know scale yet
    )
 
    if circles is None:
        raise ValueError(
            "No circles detected.  Ensure the 32 mm reference object is "
            "clearly visible and contrasts with the background, or switch "
            "to --mode mat if you used the calibration mat."
        )
 
    circles = np.round(circles[0]).astype(int)
 
    # Show all candidates for debugging
    dbg = img.copy()
    for (cx_, cy_, cr_) in circles:
        cv2.circle(dbg, (cx_, cy_), cr_, (0, 200, 255), 3)
    save_debug(f"{base_name}_01_circle_candidates.png", dbg,
               f"{len(circles)} candidates")
 
    # Pick the best candidate: prefer circles not touching the image border
    margin = 20
 
    def score(xyr):
        x, y, r = xyr
        near_border = (x - r < margin or y - r < margin or
                       x + r > w - margin or y + r > h - margin)
        # Also mildly prefer medium-sized circles (not tiny, not huge)
        size_penalty = abs(r - 80)          # 80 px is a rough prior for 32 mm
        return -size_penalty if not near_border else -1e9
 
    best_circle   = max(circles, key=score)
    cx, cy, cr    = best_circle
    pixels_per_mm = MetrologyEngine.calibrate(cr * 2, REF_DIAM_MM)
    print(f"  [ref] Chosen circle: centre=({cx},{cy})  radius={cr}px")
    print(f"  [ref] Scale: {pixels_per_mm:.3f} px/mm")
 
    dbg2 = img.copy()
    cv2.circle(dbg2, (cx, cy), cr, (0, 255, 0), 5)
    cv2.circle(dbg2, (cx, cy),  5, (0, 255, 0), -1)
    cv2.putText(dbg2, f"{REF_DIAM_MM:.0f}mm ref", (cx + cr + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    save_debug(f"{base_name}_02_chosen_circle.png", dbg2,
               f"{pixels_per_mm:.2f} px/mm")
 
    # 2. Erase the reference circle from the image before segmentation
    img_no_ref = img.copy()
    cv2.circle(img_no_ref, (cx, cy), cr + 15, (128, 128, 128), -1)
    save_debug(f"{base_name}_03_ref_erased.png", img_no_ref, "ref erased")
 
    # 3. Segment tool
    print("  [ref] Segmenting tool …")
    segmentor    = ToolSegmentor()
    tool_mask    = segmentor.segment_tool(img_no_ref)
    save_debug(f"{base_name}_04_mask_raw.png", tool_mask, "raw mask")
 
    tool_contour = segmentor.extract_largest_contour(tool_mask)
    final_mask   = segmentor.create_final_mask(img.shape, tool_contour)
    save_debug(f"{base_name}_05_mask_final.png", final_mask, "final mask")
 
    overlay = img.copy()
    cv2.drawContours(overlay, [tool_contour], -1, (0, 255, 0), 5)
    cv2.circle(overlay, (cx, cy), cr, (0, 200, 255), 4)   # keep ref visible
    save_debug(f"{base_name}_06_contour.png", overlay, "contour overlay")
 
    canvas_w_mm = w / pixels_per_mm
    canvas_h_mm = h / pixels_per_mm
    _measure_and_export(tool_contour, final_mask,
                        pixels_per_mm, base_name,
                        canvas_w_mm=canvas_w_mm, canvas_h_mm=canvas_h_mm)
 
 
# ── Shared measurement + export ────────────────────────────────────────────────
def _measure_and_export(contour, mask, pixels_per_mm,
                        base_name, canvas_w_mm, canvas_h_mm):
    metrology       = MetrologyEngine(pixels_per_mm)
    real_w, real_h  = metrology.measure_contour(contour)
 
    # minAreaRect can orient either way — report longer dim as "length"
    length_mm = max(real_w, real_h)
    width_mm  = min(real_w, real_h)
    area_mm2  = cv2.contourArea(contour) / (pixels_per_mm ** 2)
 
    print(f"\n  ┌─ Tool measurements ───────────────────")
    print(f"  │  Length : {length_mm:7.1f} mm")
    print(f"  │  Width  : {width_mm:7.1f} mm")
    print(f"  │  Area   : {area_mm2:7.0f} mm²")
    print(f"  └───────────────────────────────────────\n")
 
    svg_exp = SVGExporter(pixels_per_mm)
    svg_exp.save_high_res_mask(
        mask, contour,
        os.path.join(OUT_DIR, f"{base_name}_tool_hires.png")
    )
    svg_exp.contour_to_svg(
        contour, canvas_w_mm, canvas_h_mm,
        os.path.join(OUT_DIR, f"{base_name}_tool.svg")
    )
 
    dxf_exp = DXFExporter(pixels_per_mm)
    dxf_exp.contour_to_dxf(
        contour,
        os.path.join(OUT_DIR, f"{base_name}_tool.dxf")
    )
 
    print(f"  Outputs written to  {OUT_DIR}/")
 
 
# ── Top-level runner ───────────────────────────────────────────────────────────
def run_pipeline(image_path: str, mode: str = "auto"):
    if not os.path.exists(image_path):
        print(f"[ FATAL ] File not found: {image_path}")
        return
 
    os.makedirs(OUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
 
    print(f"\n{'='*60}")
    print(f"  File : {os.path.basename(image_path)}")
    print(f"  Mode : {mode}")
    print(f"{'='*60}")
 
    try:
        img = load_image(image_path)
        if img is None:
            print("[ FATAL ] Could not decode image.")
            return
        print(f"  Size : {img.shape[1]} × {img.shape[0]} px")
 
        if mode == "auto":
            mode = auto_detect_mode(img)
 
        if mode == "mat":
            run_mat_pipeline(img, base_name)
        elif mode == "ref":
            run_ref_pipeline(img, base_name)
        else:
            print(f"[ FATAL ] Unknown mode '{mode}'.")
            return
 
        print("[ SUCCESS ] Pipeline completed!")
 
    except Exception as exc:
        print(f"[ FAIL ]  {exc}")
        raise
 
 
def run_batch(input_dir: str = "data/input", mode: str = "auto"):
    patterns = ["*.HEIC", "*.heic", "*.JPG", "*.jpg", "*.PNG", "*.png"]
    files    = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_dir, p)))
    files = sorted(set(files))
 
    if not files:
        print(f"[ INFO ] No images found in {input_dir}")
        return
 
    for f in files:
        run_pipeline(f, mode=mode)
 
 
# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shape Extractor")
    parser.add_argument(
        "image", nargs="?",
        help="Single image path. Omit to batch-process --input-dir."
    )
    parser.add_argument(
        "--mode", choices=["auto", "mat", "ref"], default="auto",
        help="Calibration mode (default: auto)"
    )
    parser.add_argument(
        "--input-dir", default="data/input",
        help="Batch input directory (default: data/input)"
    )
    args = parser.parse_args()
 
    if args.image:
        run_pipeline(args.image, mode=args.mode)
    else:
        run_batch(input_dir=args.input_dir, mode=args.mode)
