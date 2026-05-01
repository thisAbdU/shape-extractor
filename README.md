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
# Shape Extractor

A small, self-contained pipeline that converts photos of tools (taken on a
calibration mat or with a 32 mm reference object) into millimetre-accurate
vector outputs (SVG and DXF) plus high-resolution masks. It's intended for
quickly producing 1:1 digital twins for CNC/laser cutting or documentation.

Key features
- Two calibration modes: `mat` (600×200 mm printed mat with ArUco markers)
	and `ref` (single 32 mm circular reference object).
- Robust ArUco + contour-based mat detection with fallbacks for glare.
- HSV-based segmentation tuned for a cyan/blue mat surface.
- Outputs: SVG (1 unit = 1 mm), DXF (basic LWPOLYLINE), and high-res mask PNGs.

Quick start
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place your images in `data/input/` (HEIC, JPG, PNG supported). It's already moved by default. if you want to start from scratch clear and move images from resources to here using the commnad given above. Then run:

```bash
# process all images in data/input/
python main.py

# or process a single image
python main.py data/input/IMG_5214.HEIC
```

Modes
- auto (default): pipeline will try to detect the mat (via ArUco). If it
	finds the mat it runs the `mat` pipeline; otherwise it falls back to `ref`.
- mat: uses a fixed warp that maps the printed 600×200 mm mat to a 3000×1000
	px image (5 px/mm) before segmentation.
- ref: detects the 32 mm circle in the photo to compute pixels/mm.

Project layout

```
./
├─ main.py                # pipeline entrypoint and CLI
├─ requirements.txt       # Python dependencies
├─ core/                  # core pipeline modules (detector, segmentor, etc.)
├─ utils/                 # helpers (file_manager, visualizer)
├─ data/
│  ├─ input/              # put images to process here
│  └─ output/             # pipeline outputs and debug images
├─ resources/             # printable mat, test images
└─ tests/                 # unit tests (pytest)
```

Outputs
- {basename}_tool.svg       — vector outline scaled in millimetres
- {basename}_tool.dxf       — DXF polyline suitable for CNC/laser tools
- {basename}_tool_hires.png — high-resolution binary mask of the tool
- Numerous debug PNGs written to `data/output/` (warp, masks, overlays).

Troubleshooting & tips
- If mat mode claims the mat is upside-down, inspect the saved warped image
	(saved as `{basename}_01_warped.png`) — the code checks for blue coverage
	and will error if the mat appears to be the cardboard back.
- If no circle is detected in `ref` mode, make sure the 32 mm reference
	object is clearly visible and contrasts with the background.
- Tweak HSV ranges and morphological kernel sizes in
	`core/segmentor.py` if segmentation is noisy for your images.

Running tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

License & notes
- This repository is a small, MIT-style utility for rapid prototyping. See the
	source in `core/` for implementation details and unit tests in `tests/`.
