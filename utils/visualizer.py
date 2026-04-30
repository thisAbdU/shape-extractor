"""Drawing helpers for annotated output images.

This module intentionally avoids heavy imaging libs for tests; it provides
bounding-box rendering logic that returns a copy of the array with simple
pixel-level annotations.
"""
from typing import Tuple, List
import numpy as np


def draw_boxes(image: np.ndarray, boxes: List[Tuple[int, int, int, int]], color: Tuple[int,int,int]=(255,0,0)) -> np.ndarray:
    """Return a copy of image with boxes drawn as 1-pixel rectangles.

    Args:
        image: HxWx3 uint8
        boxes: list of (x,y,w,h)
    """
    out = image.copy()
    h, w = out.shape[:2]
    for (x, y, bw, bh) in boxes:
        # top/bottom
        for i in range(x, min(x+bw, w)):
            if 0 <= y < h:
                out[y, i, :] = color
            if 0 <= y+bh-1 < h:
                out[y+bh-1, i, :] = color
        for j in range(y, min(y+bh, h)):
            if 0 <= x < w:
                out[j, x, :] = color
            if 0 <= x+bw-1 < w:
                out[j, x+bw-1, :] = color
    return out
