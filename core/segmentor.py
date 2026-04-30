import cv2
import numpy as np

class ToolSegmentor:
    def __init__(self):
        pass
    
    def segment_tool(self, image):
        """
        Isolate the tool from the blue workspace background using HSV color masking.
        Returns a binary mask of the tool.
        """
        # Convert to HSV color space for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for blue background (workspace)
        # Blue typically has Hue around 100-130 in OpenCV
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask for blue background
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Invert to get the tool (non-blue areas)
        tool_mask = cv2.bitwise_not(blue_mask)
        
        # Apply morphological operations to clean the mask
        cleaned_mask = self._clean_mask(tool_mask)
        
        return cleaned_mask
    
    def _clean_mask(self, mask):
        """
        Clean the binary mask using morphological operations to remove noise
        and smooth the tool's edges.
        """
        # Define kernel sizes based on image resolution
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Opening: Remove small noise (erosion followed by dilation)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Closing: Fill small holes (dilation followed by erosion)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)
        
        # Additional opening to remove remaining small artifacts
        final_cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small)
        
        return final_cleaned
    
    def extract_largest_contour(self, mask):
        """
        Extract the outermost contour of the tool from the cleaned mask.
        Returns the largest contour (assumed to be the tool).
        """
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No contours found in tool mask")
        
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter out very small contours (noise)
        if cv2.contourArea(largest_contour) < 1000:
            raise ValueError("Largest contour too small - likely noise")
        
        return largest_contour
    
    def create_final_mask(self, image_shape, contour):
        """
        Create a clean binary mask containing only the tool contour.
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill the contour
        return mask
