import cv2
from PIL import Image
import pytesseract
import numpy as np
import math

class Rule:
  def __init__(self, input_path, output_path):
    self.input_path = input_path
    self.output_path = output_path
    
  def set_input_path(self, input_path: str):
    self.input_path = input_path
    
  def set_output_path(self, output_path: str):
    self.output_path = output_path
  
  def rescale_image(self, target_dpi=300):
    try:
      with Image.open(self.input_path) as img:
        original_dpi = img.info.get('dpi', (96, 96))[0]
    except Exception as e:
      original_dpi = 96
    image = cv2.imread(self.input_path)
    scaling_factor = target_dpi / original_dpi
    height, width = image.shape[:2]
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    rescaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return rescaled_image
  
  def get_gray_image(self):
    image = cv2.imread(self.input_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
  
  def adaptive_threshold(self):
    gray_image = self.get_gray_image()
    adaptive_thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
    return adaptive_thresh_image
  
  def binarize_image(self):
    gray_image = self.get_gray_image()
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image
  
  def remove_noise(self):
    image = cv2.imread(self.input_path, cv2.IMREAD_GRAYSCALE)
    noise_removed_image = cv2.medianBlur(image, 3)
    return noise_removed_image
  
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

  def deskew_image(self):
    image = cv2.imread(self.input_path)
    angle = self.compute_skew(image)
    deskewed_image = self.rotate_image(image, angle)
    return deskewed_image
  
  def remove_border(self):
    image = self.get_gray_image()
    _, thresholded_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
      max_contour = max(contours, key=cv2.contourArea)
      x, y, w, h = cv2.boundingRect(max_contour)
      cropped_image = image[y:y+h, x:x+w]
      return cropped_image
    return image
  
  def get_text_coordinates(self):
    image = self.remove_border()
    text_coords = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return text_coords
  
  def enhance_contrast(self):
    """Meningkatkan contrast teks"""
    image = self.get_gray_image()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    return enhanced

  def denoise_strong(self):
      """Denoising yang lebih kuat untuk comic-style image"""
      image = cv2.imread(self.input_path)
      denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
      return cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

  def detect_text_regions(self):
      """Deteksi dan isolasi region teks"""
      image = self.get_gray_image()
      # Menggunakan morphological operations
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
      grad = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
      _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
      return bw
  
  def run(self, method: int):
    if method == 1:
      image = self.rescale_image()
    elif method == 2:
      image = self.adaptive_threshold()
    elif method == 3:
      image = self.binarize_image()
    elif method == 4:
      image = self.remove_noise()
    elif method == 5:
      image = self.deskew_image()
    elif method == 6:
      image = self.remove_border()
    elif method == 7:
      image = self.enhance_contrast()
    elif method == 8:
      image = self.denoise_strong()
    elif method == 9:
      image = self.detect_text_regions()
    else:
      raise ValueError("Invalid method")
    # simpan imagenya ya
    cv2.imwrite(self.output_path, image)
    return image