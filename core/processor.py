import cv2
import numpy as np
from PIL import Image
import pillow_heif

def load_image(path):
    """Handles HEIC and standard formats, returning a BGR OpenCV image."""
    if path.lower().endswith('.heic'):
        heif_file = pillow_heif.read_heif(path)
        image = Image.frombytes(
            heif_file.mode, heif_file.size, heif_file.data, "raw"
        )
        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return cv2.imread(path)

def preprocess(image):
    """Standardizes image for contour detection: Grayscale -> Blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur removes 'noise' like grease or scratches on the mat
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    return blurred