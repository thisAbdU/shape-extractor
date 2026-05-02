"""
core/exporter.py
================
SVG and DXF export with CNC-ready contour smoothing.
"""
 
import cv2
import numpy as np
 
 
# ── Smoothing defaults ────────────────────────────────────────────────────────
DEFAULT_SMOOTH_WINDOW  = 11   # Gaussian kernel width (points)
DEFAULT_SMOOTH_ITERS   = 3    # smoothing passes
DEFAULT_SUBSAMPLE_STEP = 3    # keep every Nth point
 
 
def smooth_contour(contour,
                   window: int = DEFAULT_SMOOTH_WINDOW,
                   iterations: int = DEFAULT_SMOOTH_ITERS,
                   subsample_step: int = DEFAULT_SUBSAMPLE_STEP):
    """
    Smooth a contour's coordinates and subsample for CNC output.
 
    Args:
        contour        : OpenCV contour array, shape (N, 1, 2)
        window         : Gaussian kernel width in points
        iterations     : number of smoothing passes
        subsample_step : keep every Nth point after smoothing
 
    Returns:
        Smoothed, subsampled contour as float32 array, shape (M, 1, 2)
    """
    pts = contour.reshape(-1, 2).astype(float)
    n   = len(pts)
 
    if n < window * 2:
        # Contour too short to smooth meaningfully — return as-is
        return contour.astype(np.float32)
 
    # Build Gaussian kernel
    x      = np.linspace(-2, 2, window)
    kernel = np.exp(-x ** 2)
    kernel /= kernel.sum()
 
    half = window // 2
 
    for _ in range(iterations):
        # Wrap-around padding so the closed contour is smooth at the seam
        px = np.concatenate([pts[-half:, 0], pts[:, 0], pts[:half, 0]])
        py = np.concatenate([pts[-half:, 1], pts[:, 1], pts[:half, 1]])
 
        smoothed_x = np.convolve(px, kernel, mode='valid')
        smoothed_y = np.convolve(py, kernel, mode='valid')
 
        # convolve output length = len(input) - len(kernel) + 1
        # with our padding: len(input) = n + window - 1, output = n
        pts[:, 0] = smoothed_x[:n]
        pts[:, 1] = smoothed_y[:n]
 
    # Subsample
    pts = pts[::subsample_step]
 
    return pts.reshape(-1, 1, 2).astype(np.float32)
 
 
# ── SVGExporter ───────────────────────────────────────────────────────────────
 
class SVGExporter:
    def __init__(self, pixels_per_mm: float):
        self.pixels_per_mm = pixels_per_mm
 
    def contour_to_svg(self, contour, width_mm, height_mm, output_path,
                        smooth: bool = True):
        """
        Convert a contour to SVG with 1 unit = 1mm.
 
        Args:
            contour     : OpenCV contour (raw pixel coordinates)
            width_mm    : SVG canvas width in mm
            height_mm   : SVG canvas height in mm
            output_path : file path for the output .svg
            smooth      : apply CNC smoothing before export (default True)
        """
        if smooth:
            contour = smooth_contour(contour)
 
        # Convert pixels → mm
        points_mm = [
            (pt[0][0] / self.pixels_per_mm, pt[0][1] / self.pixels_per_mm)
            for pt in contour
        ]
 
        svg  = self._svg_header(width_mm, height_mm)
        svg += f'  <path d="{self._to_path(points_mm)}" '
        svg += f'fill="black" stroke="none" />\n'
        svg += '</svg>'
 
        with open(output_path, 'w') as f:
            f.write(svg)
 
    def save_high_res_mask(self, mask, contour, output_path,
                            scale_factor: int = 10):
        """Save a high-resolution PNG of the tool mask."""
        h, w = mask.shape
        scaled_contour = (contour * scale_factor).astype(np.int32)
        hires = np.zeros((h * scale_factor, w * scale_factor), dtype=np.uint8)
        cv2.drawContours(hires, [scaled_contour], -1, 255, -1)
        cv2.imwrite(output_path, hires)
 
    # ── Private ──────────────────────────────────────────────────────────────
 
    def _svg_header(self, w_mm, h_mm):
        return (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg width="{w_mm}mm" height="{h_mm}mm" '
            f'viewBox="0 0 {w_mm} {h_mm}" '
            f'xmlns="http://www.w3.org/2000/svg">\n'
        )
 
    def _to_path(self, points):
        if not points:
            return ""
        path = f"M {points[0][0]:.3f} {points[0][1]:.3f}"
        for x, y in points[1:]:
            path += f" L {x:.3f} {y:.3f}"
        path += " Z"
        return path
 
 
# ── DXFExporter ───────────────────────────────────────────────────────────────
 
class DXFExporter:
    def __init__(self, pixels_per_mm: float):
        self.pixels_per_mm = pixels_per_mm
 
    def contour_to_dxf(self, contour, output_path, smooth: bool = True):
        """
        Convert a contour to DXF R14 LWPOLYLINE for CNC/laser cutting.
 
        Args:
            contour     : OpenCV contour (raw pixel coordinates)
            output_path : file path for the output .dxf
            smooth      : apply CNC smoothing before export (default True)
        """
        if smooth:
            contour = smooth_contour(contour)
 
        # Convert pixels → mm
        points_mm = [
            (pt[0][0] / self.pixels_per_mm, pt[0][1] / self.pixels_per_mm)
            for pt in contour
        ]
 
        dxf  = self._dxf_header()
        dxf += self._polyline(points_mm)
        dxf += '  0\nEOF\n'
 
        with open(output_path, 'w') as f:
            f.write(dxf)
 
    # ── Private ──────────────────────────────────────────────────────────────
 
    def _dxf_header(self):
        return '''  0
SECTION
  2
HEADER
  0
ENDSEC
  0
SECTION
  2
TABLES
  0
TABLE
  2
LAYER
  0
LAYER
  2
0
 70
0
  62
7
  6
Continuous
  0
ENDTAB
  0
ENDSEC
  0
SECTION
  2
ENTITIES
'''
 
    def _polyline(self, points):
        if not points:
            return ""
        dxf = '  0\nLWPOLYLINE\n  8\n0\n 90\n{}\n'.format(len(points))
        for x, y in points:
            dxf += f' 10\n{x:.3f}\n 20\n{y:.3f}\n'
        dxf += ' 70\n1\n'
        return dxf
 
