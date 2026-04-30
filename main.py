import os
import cv2
import numpy as np
from core.processor import load_image
from core.detector import detect_mat_corners, get_top_down_view
from core.measurer import MetrologyEngine
from core.segmentor import ToolSegmentor
from core.exporter import SVGExporter, DXFExporter

def run_pipeline(image_path):
    if not os.path.exists(image_path):
        print(f"[ FATAL ] File not found: {image_path}")
        return

    try:
        # Load image
        img = load_image(image_path)
        if img is None:
            print("[ FATAL ] OpenCV failed to decode the image.")
            return
        
        # Optical rectification
        corners = detect_mat_corners(img)
        warped = get_top_down_view(img, corners, target_width_mm=600, target_height_mm=200)
        cv2.imwrite("data/output/08_rectified_mat.png", warped)
        
        # Scale calibration
        reference_diameter_px = MetrologyEngine.detect_reference_object(warped)
        pixels_per_mm = MetrologyEngine.calibrate(reference_diameter_px, actual_mm=32.0)
        metrology = MetrologyEngine(pixels_per_mm)
        
        # Tool segmentation
        segmentor = ToolSegmentor()
        tool_mask = segmentor.segment_tool(warped)
        tool_contour = segmentor.extract_largest_contour(tool_mask)
        final_mask = segmentor.create_final_mask(warped.shape, tool_contour)
        cv2.imwrite("data/output/15_final_tool_mask.png", final_mask)
        
        # Measure dimensions
        real_w, real_h = metrology.measure_contour(tool_contour)
        print(f"Tool dimensions: {real_w:.1f}mm x {real_h:.1f}mm")
        
        # Generate outputs
        svg_exporter = SVGExporter(pixels_per_mm)
        svg_exporter.save_high_res_mask(final_mask, tool_contour, "data/output/16_tool_mask_highres.png")
        svg_exporter.contour_to_svg(tool_contour, 600, 200, "data/output/17_tool_digital_twin.svg")
        
        dxf_exporter = DXFExporter(pixels_per_mm)
        dxf_exporter.contour_to_dxf(tool_contour, "data/output/17_tool_digital_twin.dxf")
        
        print("[ SUCCESS ] Pipeline completed!")
        
    except Exception as e:
        print(f"[ FAIL ] Pipeline error: {e}")
        raise

def run_batch_processing(input_dir="data/input"):
    """Process all HEIC files in the input directory"""
    import glob
    
    heic_files = glob.glob(os.path.join(input_dir, "*.HEIC"))
    if not heic_files:
        print(f"[ INFO ] No HEIC files found in {input_dir}")
        return
    
    for heic_file in heic_files:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(heic_file)}")
        print(f"{'='*60}")
        run_pipeline(heic_file)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process specific file provided as argument
        image_path = sys.argv[1]
        run_pipeline(image_path)
    else:
        # Process all files in data/input directory
        run_batch_processing()