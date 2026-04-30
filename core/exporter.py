import cv2
import numpy as np

class SVGExporter:
    def __init__(self, pixels_per_mm):
        self.pixels_per_mm = pixels_per_mm
    
    def contour_to_svg(self, contour, width_mm, height_mm, output_path):
        """
        Convert a contour to SVG format with 1 unit = 1mm scaling.
        """
        # Convert contour points from pixels to millimeters
        points_mm = []
        for point in contour:
            x_mm = point[0][0] / self.pixels_per_mm
            y_mm = point[0][1] / self.pixels_per_mm
            points_mm.append((x_mm, y_mm))
        
        # Create SVG content
        svg_content = self._create_svg_header(width_mm, height_mm)
        
        # Add the tool path
        path_data = self._contour_to_path(points_mm)
        svg_content += f'  <path d="{path_data}" fill="black" stroke="none" />\n'
        
        # Close SVG
        svg_content += '</svg>'
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(svg_content)
    
    def _create_svg_header(self, width_mm, height_mm):
        """Create SVG header with proper dimensions."""
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width_mm}mm" height="{height_mm}mm" viewBox="0 0 {width_mm} {height_mm}" 
     xmlns="http://www.w3.org/2000/svg">
'''
    
    def _contour_to_path(self, points):
        """Convert contour points to SVG path data."""
        if not points:
            return ""
        
        # Start with the first point
        path = f"M {points[0][0]:.2f} {points[0][1]:.2f}"
        
        # Add line segments for remaining points
        for point in points[1:]:
            path += f" L {point[0]:.2f} {point[1]:.2f}"
        
        # Close the path
        path += " Z"
        
        return path
    
    def save_high_res_mask(self, mask, contour, output_path, scale_factor=10):
        """
        Save a high-resolution PNG mask of the tool.
        """
        # Create high-resolution version
        h, w = mask.shape
        high_res_w = w * scale_factor
        high_res_h = h * scale_factor
        
        # Scale up the contour
        scaled_contour = contour * scale_factor
        
        # Create high-res mask
        high_res_mask = np.zeros((high_res_h, high_res_w), dtype=np.uint8)
        cv2.drawContours(high_res_mask, [scaled_contour], -1, 255, -1)
        
        # Save the high-resolution mask
        cv2.imwrite(output_path, high_res_mask)

class DXFExporter:
    def __init__(self, pixels_per_mm):
        self.pixels_per_mm = pixels_per_mm
    
    def contour_to_dxf(self, contour, output_path):
        """
        Convert a contour to DXF format for CNC/Laser cutting.
        Basic DXF R14 format with lines only.
        """
        # Convert contour points from pixels to millimeters
        points_mm = []
        for point in contour:
            x_mm = point[0][0] / self.pixels_per_mm
            y_mm = point[0][1] / self.pixels_per_mm
            points_mm.append((x_mm, y_mm))
        
        # Create basic DXF content
        dxf_content = self._create_dxf_header()
        
        # Add entities section with polyline
        dxf_content += self._create_polyline(points_mm)
        
        # Close DXF
        dxf_content += '  0\nEOF\n'
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(dxf_content)
    
    def _create_dxf_header(self):
        """Create basic DXF header."""
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
    
    def _create_polyline(self, points):
        """Create DXF polyline entity from points."""
        if not points:
            return ""
        
        dxf = '''  0
LWPOLYLINE
  8
0
 90
{}
'''.format(len(points))
        
        # Add vertices
        for point in points:
            dxf += f''' 10
{point[0]:.3f}
 20
{point[1]:.3f}
'''
        
        # Close the polyline
        dxf += ''' 70
1
'''
        
        return dxf
