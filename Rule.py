import cv2
from PIL import Image
import pytesseract
import numpy as np
import math

class Rule:
  def __init__(self, input_path: str = None, output_path: str = None):
    self.input_path = input_path
    self.output_path = output_path
    
  def set_input_path(self, input_path: str):
    self.input_path = input_path
    
  def set_output_path(self, output_path: str):
    self.output_path = output_path
  
  def rescale_image(self, image, target_dpi=300):
    """Modified to accept direct image input"""
    try:
        original_dpi = 96  # Default DPI
        scaling_factor = target_dpi / original_dpi
        height, width = image.shape[:2]
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"Error in rescale_image: {e}")
        return image
    
  def compute_skew(self, src_img):
    if len(src_img.shape) == 3:
      h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
      h, w = src_img.shape
    else:
      print('Unsupported image type')
      return 0.0
    
    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30, minLineLength=w / 4.0, maxLineGap=h / 4.0)
    
    if lines is None or len(lines) == 0:
      return 0.0
    
    angle = 0.0
    cnt = 0
    
    for line in lines:
      x1, y1, x2, y2 = line[0]
      ang = np.arctan2(y2 - y1, x2 - x1)
      if math.fabs(ang) <= math.radians(30):
        angle += ang
        cnt += 1
    
    if cnt == 0:
      return 0.0
    
    return (angle / cnt) * 180 / math.pi

  def rotate_image(self, image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])
    bound_w = int(image.shape[1] * abs_cos + image.shape[0] * abs_sin)
    bound_h = int(image.shape[1] * abs_sin + image.shape[0] * abs_cos)
    
    rot_mat[0, 2] += bound_w / 2 - image_center[0]
    rot_mat[1, 2] += bound_h / 2 - image_center[1]
    
    result = cv2.warpAffine(image, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return result
  
  def get_text_coordinates(self, image):
    """Modified to accept direct image input"""
    text_coords = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return text_coords

  def detect_text_regions(self, image):
    """Modified to accept direct image input"""
    # Menggunakan morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    grad = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw
  
  def run(self, rule: int, image = None) -> np.ndarray:
    """Modified to accept direct image input"""
    if image is None and self.input_path:
        image = cv2.imread(self.input_path)
    
    # Ensure image is in correct format for each rule
    if len(image.shape) == 2:  # If grayscale, convert to BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Process image based on rule number
    if rule == 1:
        processed_image = self.rescale_image(image)
    elif rule == 2:
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif rule == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
    elif rule == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, processed_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif rule == 5:
        processed_image = cv2.medianBlur(image, 3)
    elif rule == 6:
        angle = self.compute_skew(image)
        processed_image = self.rotate_image(image, angle)
    elif rule == 7:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            processed_image = image[y:y+h, x:x+w]
        else:
            processed_image = image
    elif rule == 8:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed_image = clahe.apply(gray)
    elif rule == 9:
        # Ensure image is in color format for denoising
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        processed_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        processed_image = image
    
    if self.output_path:
        cv2.imwrite(self.output_path, processed_image)
    
    return processed_image