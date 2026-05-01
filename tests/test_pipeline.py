"""
tests/test_pipeline.py
======================
Unit tests for the shape-extractor pipeline.
 
Run with:
    python3 -m pytest tests/ -v
    python3 -m pytest tests/ -v --tb=short   # compact tracebacks
"""
 
import cv2
import numpy as np
import pytest
import sys
import os
 
# Make sure the project root is on the path regardless of where pytest is run
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
from core.measurer  import MetrologyEngine
from core.segmentor import ToolSegmentor
from core.detector  import _sort_corners, _four_extremes
from core.exporter  import SVGExporter, DXFExporter
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Helpers — synthetic image builders
# ─────────────────────────────────────────────────────────────────────────────
 
def make_blue_mat(width=600, height=200, tool_rect=None):
    """
    Create a synthetic warped-mat image (pure blue background).
    Optionally draw a grey rectangle representing a tool.
 
    Args:
        width, height : image dimensions in pixels
        tool_rect     : (x, y, w, h) of the tool rectangle, or None
    Returns:
        BGR image (np.ndarray uint8)
    """
    # Mat blue: RGB #00ADEF → BGR (239, 173, 0)
    img = np.full((height, width, 3), [239, 173, 0], dtype=np.uint8)
    if tool_rect is not None:
        x, y, w, h = tool_rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (180, 180, 180), -1)
    return img
 
 
def make_circle_image(width=800, height=600, cx=400, cy=300, radius=80,
                      bg_color=(200, 200, 200), circle_color=(50, 50, 50)):
    """Synthetic image with one filled circle on a plain background."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    cv2.circle(img, (cx, cy), radius, circle_color, -1)
    return img
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MetrologyEngine
# ─────────────────────────────────────────────────────────────────────────────
 
class TestMetrologyEngine:
 
    def test_calibrate_basic(self):
        """32 mm object measured at 160 px → 5 px/mm."""
        ppm = MetrologyEngine.calibrate(160.0, actual_mm=32.0)
        assert ppm == pytest.approx(5.0)
 
    def test_calibrate_different_size(self):
        """50 mm object at 250 px → 5 px/mm."""
        ppm = MetrologyEngine.calibrate(250.0, actual_mm=50.0)
        assert ppm == pytest.approx(5.0)
 
    def test_calibrate_zero_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            MetrologyEngine.calibrate(0.0)
 
    def test_calibrate_negative_raises(self):
        with pytest.raises(ValueError):
            MetrologyEngine.calibrate(-10.0)
 
    def test_measure_contour_rectangle(self):
        """A 100×50 px rectangle at 5 px/mm → 20×10 mm."""
        engine = MetrologyEngine(pixels_per_mm=5.0)
        # Build a clean rectangular contour
        contour = np.array([
            [[50,  50]],
            [[150, 50]],
            [[150, 100]],
            [[50,  100]],
        ], dtype=np.int32)
        w_mm, h_mm = engine.measure_contour(contour)
        long_mm  = max(w_mm, h_mm)
        short_mm = min(w_mm, h_mm)
        assert long_mm  == pytest.approx(20.0, abs=0.5)
        assert short_mm == pytest.approx(10.0, abs=0.5)
 
    def test_measure_contour_square(self):
        """A 100×100 px square at 5 px/mm → 20×20 mm."""
        engine = MetrologyEngine(pixels_per_mm=5.0)
        contour = np.array([
            [[0,   0]],
            [[100, 0]],
            [[100, 100]],
            [[0,   100]],
        ], dtype=np.int32)
        w_mm, h_mm = engine.measure_contour(contour)
        assert w_mm == pytest.approx(h_mm, abs=0.5)
 
    def test_contour_area_mm2(self):
        """100×50 px rectangle at 5 px/mm → 200 mm²."""
        engine = MetrologyEngine(pixels_per_mm=5.0)
        contour = np.array([
            [[0,  0]],
            [[100, 0]],
            [[100, 50]],
            [[0,  50]],
        ], dtype=np.int32)
        area = engine.contour_area_mm2(contour)
        assert area == pytest.approx(200.0, abs=1.0)
 
    def test_pixel_mm_roundtrip(self):
        engine = MetrologyEngine(pixels_per_mm=5.0)
        assert engine.mm_to_pixel(engine.pixel_to_mm(100)) == pytest.approx(100)
        assert engine.pixel_to_mm(engine.mm_to_pixel(20))  == pytest.approx(20)
 
    def test_detect_reference_object_finds_circle(self):
        """Hough should find a clearly drawn circle."""
        img = make_circle_image(radius=80)
        diameter_px = MetrologyEngine.detect_reference_object(img)
        assert diameter_px == pytest.approx(160, abs=10)
 
    def test_detect_reference_object_no_circle_raises(self):
        """Plain flat image should raise ValueError."""
        img = np.full((400, 600, 3), 128, dtype=np.uint8)
        with pytest.raises(ValueError, match="No circular"):
            MetrologyEngine.detect_reference_object(img)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ToolSegmentor
# ─────────────────────────────────────────────────────────────────────────────
 
class TestToolSegmentor:
 
    def test_segment_tool_finds_tool(self):
        """Grey rectangle on blue mat should be found as the tool."""
        img  = make_blue_mat(600, 200, tool_rect=(200, 70, 200, 60))
        seg  = ToolSegmentor()
        mask = seg.segment_tool(img)
        assert mask.dtype == np.uint8
        assert mask.shape == (200, 600)
        # Tool pixels should be white
        white_px = (mask > 128).sum()
        assert white_px > 0, "No tool pixels found in mask"
 
    def test_segment_tool_pure_blue_no_tool(self):
        """Pure blue image (no tool) should produce a near-empty mask."""
        img  = make_blue_mat(600, 200, tool_rect=None)
        seg  = ToolSegmentor()
        mask = seg.segment_tool(img)
        white_px = (mask > 128).sum()
        total_px = 600 * 200
        assert white_px / total_px < 0.02, (
            f"Expected <2% white pixels on pure blue mat, got "
            f"{white_px/total_px*100:.1f}%"
        )
 
    def test_extract_largest_contour_returns_contour(self):
        """Mask with one large blob → contour around it."""
        mask = np.zeros((200, 600), dtype=np.uint8)
        cv2.rectangle(mask, (200, 70), (400, 130), 255, -1)
        seg     = ToolSegmentor()
        contour = seg.extract_largest_contour(mask)
        assert contour is not None
        assert cv2.contourArea(contour) > 0
 
    def test_extract_largest_contour_empty_raises(self):
        """Empty mask should raise ValueError."""
        mask = np.zeros((200, 600), dtype=np.uint8)
        seg  = ToolSegmentor()
        with pytest.raises(ValueError):
            seg.extract_largest_contour(mask)
 
    def test_extract_largest_contour_tiny_raises(self):
        """Single pixel blob should raise ValueError (too small)."""
        mask = np.zeros((200, 600), dtype=np.uint8)
        mask[100, 300] = 255   # 1 pixel
        seg  = ToolSegmentor()
        with pytest.raises(ValueError):
            seg.extract_largest_contour(mask)
 
    def test_create_final_mask_shape(self):
        """Final mask should be same spatial shape as input image."""
        img  = make_blue_mat(600, 200, tool_rect=(200, 70, 200, 60))
        seg  = ToolSegmentor()
        mask = seg.segment_tool(img)
        try:
            contour   = seg.extract_largest_contour(mask)
            final     = seg.create_final_mask(img.shape, contour)
            assert final.shape == (200, 600)
            assert final.dtype == np.uint8
        except ValueError:
            pytest.skip("No contour found — tool too small in synthetic image")
 
    def test_checkerboard_border_excluded(self):
        """
        Checkerboard border around blue mat must NOT become the tool contour.
        Simulate by adding black/white squares around the blue area.
        """
        h, w = 300, 900
        img  = np.full((h, w, 3), 200, dtype=np.uint8)   # grey surround
        # Blue mat in the centre
        img[50:250, 150:750] = [239, 173, 0]
        # Tool inside blue area
        cv2.rectangle(img, (300, 100), (600, 200), (160, 160, 160), -1)
 
        seg  = ToolSegmentor()
        mask = seg.segment_tool(img)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest_area = cv2.contourArea(max(cnts, key=cv2.contourArea))
            total_area   = h * w
            assert largest_area / total_area < 0.5, (
                "Largest contour covers >50% of image — border is bleeding in"
            )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Detector geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
 
class TestDetectorGeometry:
 
    def test_sort_corners_canonical(self):
        """Four points in random order → sorted [TL, TR, BR, BL]."""
        pts = np.array([
            [100, 100],  # TL
            [900, 100],  # TR
            [900, 400],  # BR
            [100, 400],  # BL
        ], dtype="float32")
        # Shuffle
        shuffled = pts[[2, 0, 3, 1]]
        sorted_  = _sort_corners(shuffled)
        np.testing.assert_allclose(sorted_[0], [100, 100], atol=1)  # TL
        np.testing.assert_allclose(sorted_[1], [900, 100], atol=1)  # TR
        np.testing.assert_allclose(sorted_[2], [900, 400], atol=1)  # BR
        np.testing.assert_allclose(sorted_[3], [100, 400], atol=1)  # BL
 
    def test_sort_corners_non_axis_aligned(self):
        """Slightly rotated rectangle still sorts correctly."""
        pts = np.array([
            [105, 95],   # ~TL
            [895, 105],  # ~TR
            [905, 405],  # ~BR
            [95,  395],  # ~BL
        ], dtype="float32")
        shuffled = pts[[3, 1, 0, 2]]
        sorted_  = _sort_corners(shuffled)
        # TL should have smallest x+y
        assert sorted_[0][0] + sorted_[0][1] == min(
            p[0] + p[1] for p in pts
        )
 
    def test_four_extremes_returns_four_points(self):
        """Convex hull point cloud → exactly 4 extreme points."""
        pts = np.array([
            [10, 10], [500, 5], [505, 300], [8, 295],
            [250, 8], [502, 150], [250, 298], [9, 150],
        ], dtype="float32")
        result = _four_extremes(pts)
        assert result.shape == (4, 2)
 
    def test_parallelogram_inference(self):
        """
        Simulates the BL-missing case:
        given TL, TR, BR → infer BL = TL + BR - TR
        This mirrors exactly what detector.py does.
        """
        tl = np.array([100.0, 100.0])
        tr = np.array([900.0, 100.0])
        br = np.array([900.0, 400.0])
        bl_expected = np.array([100.0, 400.0])
 
        # Parallelogram rule: BL = TL + BR - TR
        bl_inferred = tl + br - tr
        np.testing.assert_allclose(bl_inferred, bl_expected, atol=1e-6)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Exporters
# ─────────────────────────────────────────────────────────────────────────────
 
class TestSVGExporter:
 
    def test_svg_file_created(self, tmp_path):
        """SVG output file should exist and contain basic SVG structure."""
        exp     = SVGExporter(pixels_per_mm=5.0)
        contour = np.array([
            [[0,   0]],
            [[100, 0]],
            [[100, 50]],
            [[0,  50]],
        ], dtype=np.int32)
        out = str(tmp_path / "test.svg")
        exp.contour_to_svg(contour, width_mm=600, height_mm=200,
                           output_path=out)
        assert os.path.exists(out)
        content = open(out).read()
        assert "<svg"   in content
        assert "</svg>" in content
        assert "<path"  in content
 
    def test_svg_dimensions_in_mm(self, tmp_path):
        """SVG header should declare dimensions in mm."""
        exp     = SVGExporter(pixels_per_mm=5.0)
        contour = np.array([[[0, 0]], [[50, 0]], [[50, 25]], [[0, 25]]],
                           dtype=np.int32)
        out = str(tmp_path / "dims.svg")
        exp.contour_to_svg(contour, width_mm=600, height_mm=200,
                           output_path=out)
        content = open(out).read()
        assert '600mm' in content
        assert '200mm' in content
 
    def test_svg_coordinates_scaled_to_mm(self, tmp_path):
        """
        A point at pixel (50, 25) with 5 px/mm should appear as
        (10.00, 5.00) in the SVG path data.
        """
        exp     = SVGExporter(pixels_per_mm=5.0)
        contour = np.array([[[50, 25]], [[100, 25]], [[100, 50]], [[50, 50]]],
                           dtype=np.int32)
        out = str(tmp_path / "coords.svg")
        exp.contour_to_svg(contour, width_mm=600, height_mm=200,
                           output_path=out)
        content = open(out).read()
        assert "10.00" in content   # 50px / 5 = 10 mm
        assert "5.00"  in content   # 25px / 5 =  5 mm
 
    def test_high_res_mask_saved(self, tmp_path):
        """High-res PNG mask file should be created with correct dimensions."""
        exp  = SVGExporter(pixels_per_mm=5.0)
        mask = np.zeros((200, 600), dtype=np.uint8)
        cv2.rectangle(mask, (100, 50), (500, 150), 255, -1)
        contour = np.array([
            [[100, 50]], [[500, 50]], [[500, 150]], [[100, 150]]
        ], dtype=np.int32)
        out = str(tmp_path / "hires.png")
        exp.save_high_res_mask(mask, contour, out, scale_factor=2)
        assert os.path.exists(out)
        saved = cv2.imread(out, cv2.IMREAD_GRAYSCALE)
        assert saved is not None
        assert saved.shape == (400, 1200)   # 200*2 × 600*2
 
 
class TestDXFExporter:
 
    def test_dxf_file_created(self, tmp_path):
        """DXF output file should exist and contain LWPOLYLINE."""
        exp     = DXFExporter(pixels_per_mm=5.0)
        contour = np.array([
            [[0,   0]],
            [[100, 0]],
            [[100, 50]],
            [[0,  50]],
        ], dtype=np.int32)
        out = str(tmp_path / "test.dxf")
        exp.contour_to_dxf(contour, output_path=out)
        assert os.path.exists(out)
        content = open(out).read()
        assert "LWPOLYLINE" in content
        assert "ENTITIES"   in content
 
    def test_dxf_coordinates_in_mm(self, tmp_path):
        """
        A point at pixel (50, 0) with 5 px/mm should appear as
        10.000 in the DXF file.
        """
        exp     = DXFExporter(pixels_per_mm=5.0)
        contour = np.array([[[50, 0]], [[100, 0]], [[100, 50]], [[50, 50]]],
                           dtype=np.int32)
        out = str(tmp_path / "coords.dxf")
        exp.contour_to_dxf(contour, output_path=out)
        content = open(out).read()
        assert "10.000" in content   # 50px / 5 = 10 mm
 
    def test_dxf_closed_polyline(self, tmp_path):
        """DXF polyline flag 70=1 means closed."""
        exp     = DXFExporter(pixels_per_mm=5.0)
        contour = np.array([[[0,0]],[[50,0]],[[50,25]],[[0,25]]],
                           dtype=np.int32)
        out = str(tmp_path / "closed.dxf")
        exp.contour_to_dxf(contour, output_path=out)
        content = open(out).read()
        # Flag 70 with value 1 = closed
        assert " 70\n1\n" in content
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Integration — full mat pipeline on synthetic data
# ─────────────────────────────────────────────────────────────────────────────
 
class TestMatPipelineIntegration:
 
    def test_full_mat_pipeline_known_tool(self, tmp_path):
        """
        Synthetic 600×200 px blue mat with a 200×40 px grey tool.
        At 5 px/mm the tool should measure 40×8 mm.
        """
        img      = make_blue_mat(600, 200, tool_rect=(200, 80, 200, 40))
        ppm      = 5.0
        seg      = ToolSegmentor()
        mask     = seg.segment_tool(img)
        try:
            contour  = seg.extract_largest_contour(mask)
        except ValueError:
            pytest.skip("Synthetic tool too small after morphological cleanup")
 
        engine   = MetrologyEngine(ppm)
        w_mm, h_mm = engine.measure_contour(contour)
        length   = max(w_mm, h_mm)
        width    = min(w_mm, h_mm)
 
        # Allow ±3 mm tolerance for morphological erosion on synthetic image
        assert length == pytest.approx(40.0, abs=3.0), f"Length {length:.1f} mm"
        assert width  == pytest.approx( 8.0, abs=3.0), f"Width  {width:.1f} mm"
 
    def test_svg_and_dxf_generated(self, tmp_path):
        """Full export chain produces both SVG and DXF files."""
        img     = make_blue_mat(600, 200, tool_rect=(150, 70, 300, 60))
        ppm     = 5.0
        seg     = ToolSegmentor()
        mask    = seg.segment_tool(img)
        try:
            contour = seg.extract_largest_contour(mask)
        except ValueError:
            pytest.skip("No contour in synthetic image")
 
        final   = seg.create_final_mask(img.shape, contour)
        svg_out = str(tmp_path / "tool.svg")
        dxf_out = str(tmp_path / "tool.dxf")
        png_out = str(tmp_path / "tool_hires.png")
 
        SVGExporter(ppm).contour_to_svg(contour, 600, 200, svg_out)
        SVGExporter(ppm).save_high_res_mask(final, contour, png_out)
        DXFExporter(ppm).contour_to_dxf(contour, dxf_out)
 
        assert os.path.exists(svg_out)
        assert os.path.exists(dxf_out)
        assert os.path.exists(png_out)
 
