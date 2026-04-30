# Shape Extractor

Automated pipeline for converting photos of tools on a calibration mat into millimeter-accurate 1:1 scale digital twins (DXF/SVG) for CNC/Laser cutting.

## Overview

The Shape Extractor processes raw HEIC images through a 4-stage pipeline:
1. **Optical Rectification** - Detects ArUco markers and flattens perspective
2. **Scaling Calibration** - Uses 32mm reference object for pixel-to-mm conversion
3. **Tool Segmentation** - Isolates tools from background using HSV masking
4. **Output Generation** - Creates high-res masks, SVG, and DXF files

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline on your image
python main.py
```

## Project Structure

```
shape-extractor/
├── core/                   # Core processing modules
│   ├── detector.py         # ArUco detection and perspective transform
│   ├── measurer.py         # Scale calibration and measurements
│   ├── segmentor.py       # Tool segmentation and cleaning
│   ├── exporter.py        # SVG/DXF output generation
│   └── processor.py       # Image loading utilities
├── data/
│   ├── input/             # Active processing queue (images to be processed)
│   └── output/            # Results and debug images
├── resources/             # Reference materials & test image library
│   ├── photo-mat-print test.pdf  # Printable ArUco markers & 32mm circle
│   └── IMG_*.HEIC         # Test image collection
├── utils/                 # Utility scripts
│   └── file_manager.py   # Manage test images between directories
├── main.py                # Pipeline orchestrator
└── requirements.txt       # Python dependencies
```

## Setup

### 1. Prepare Physical Reference
1. Print `resources/photo-mat-print test.pdf` at 1:1 scale
2. Cut ArUco markers and 32mm circle
3. Place markers on mat corners
4. Position 32mm circle on the mat

### 2. Capture Image
- Take photo of tools on the calibrated mat
- Ensure good lighting (avoid heavy glare)
- Save as HEIC/JPG/PNG in `data/input/`

### 3. Manage Test Images
```bash
# List available test images in resources/
python utils/file_manager.py list

# Move specific image to processing queue
python utils/file_manager.py move --file IMG_5214.HEIC

# Move all test images to processing queue
python utils/file_manager.py move

# Clear processing queue
python utils/file_manager.py clear
```

### 4. Run Pipeline
```bash
# Process all images in data/input/
python main.py

# Process specific image
python main.py data/input/IMG_5214.HEIC
```

## Output Files

### Final Results
- `17_tool_digital_twin.svg` - Vector file (1 unit = 1mm)
- `17_tool_digital_twin.dxf` - CNC/Laser cutting format
- `16_tool_mask_highres.png` - High-resolution binary mask

### Debug Images
- `01_original_gray.png` - Original grayscale image
- `04_aruco_detected.png` - Detected ArUco markers
- `08_rectified_mat.png` - Perspective-corrected view
- `08_reference_circle.png` - Detected 32mm circle
- `15_final_tool_mask.png` - Clean tool silhouette
- Plus 10+ intermediate processing steps

## Technical Details

### ArUco Detection
- Dictionary: `DICT_4X4_50`
- Robust detection with gamma correction and adaptive thresholding
- Fallback to mat contour detection for glare-heavy images

### Scale Calibration
- 32mm circular reference object
- Automatic pixel-to-millimeter ratio calculation
- Typical result: ~5 pixels/mm at current resolution

### Tool Segmentation
- HSV color space for blue background separation
- Morphological operations (opening/closing) for noise removal
- Largest contour extraction for tool identification

### Output Formats
- **SVG**: Scalable vector with 1:1 mm scaling
- **DXF**: Basic R14 format for CNC compatibility
- **PNG**: High-resolution binary mask (10x scale)

## Dependencies

- `opencv-contrib-python` - Computer vision and ArUco detection
- `numpy` - Numerical operations
- `pillow` + `pillow-heif` - HEIC image support
- `scipy` - Spatial calculations
- `imutils` - Image utilities

## Troubleshooting

### ArUco Detection Fails
- Check lighting conditions
- Ensure markers are flat and visible
- Pipeline automatically falls back to mat contour detection

### Scale Calibration Issues
- Verify 32mm circle is printed at correct size
- Ensure circle is clearly visible in the image
- Check debug image `08_reference_circle.png`

### Tool Segmentation Problems
- Adjust HSV color ranges in `segmentor.py`
- Modify morphological kernel sizes for different tools
- Check intermediate masks in `data/output/`

## Logging

Each run generates a timestamped log file:
```
data/output/shape_extractor_YYYYMMDD_HHMMSS.log
```

Contains detailed pipeline steps, measurements, and any errors encountered.