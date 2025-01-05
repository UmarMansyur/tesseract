import pytesseract
from Levenshtein import distance
from jiwer import wer
from Rule import Rule


class Tesseract:
  def __init__(self, input_path: str, output_path: str):
    self.input_path = input_path
    self.output_path = output_path
    self.ground_truth = None
    self.ocr_result = None
    
  def set_input_path(self, input_path: str):
    self.input_path = input_path
    
  def set_ground_truth(self, ground_truth: str):
    self.ground_truth = ground_truth
    
  def set_ocr_result(self, ocr_result: str):
    self.ocr_result = ocr_result
    
  def tesseract(self, image_output):
    try:
        text = pytesseract.image_to_string(image_output, lang='vie+ind')
        self.ocr_result = text
        with open('ocr_result.txt', 'a', encoding='utf-8') as file:
            file.write(text + '\n')
        return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None

  def get_test(self):
    # Ensure both texts are not None and convert to strings
    if self.ground_truth is None or self.ocr_result is None:
        raise ValueError("Both ground truth and OCR result must be set")
    
    # Preprocess texts: convert to string, strip whitespace, and ensure single line
    gt_text = str(self.ground_truth).strip().replace('\n', ' ')
    ocr_text = str(self.ocr_result).strip().replace('\n', ' ')
    
    # Split ground truth and OCR result into words
    gt_words = gt_text.split()
    ocr_words = ocr_text.split()

    # Compute Levenshtein distance (character level)
    lev_distance = distance(gt_text, ocr_text)

    # Compute Word Error Rate (WER)
    result_wer = wer(reference=gt_text, hypothesis=ocr_text)

    # Compute Character Error Rate (CER)
    result_cer = lev_distance / len(gt_text) if gt_text else 0

    # Compute Word Accuracy Rate (WAR)
    correct_words = sum(1 for gt, ocr in zip(gt_words, ocr_words) if gt == ocr)
    result_war = correct_words / len(gt_words) if gt_words else 0

    # Compute Character Accuracy Rate (CAR)
    result_car = 1 - result_cer

    # Compute Word Recognition Rate (WRR)
    result_wrr = 1 - result_wer

    return {
        'WER': result_wer,
        'CER': result_cer,
        'WAR': result_war,
        'CAR': result_car,
        'WRR': result_wrr,
        'Levenshtein': lev_distance
    }

  def run(self, rule: int = 1, ground_truth: str = None):
    rule_processor = Rule(self.input_path, self.output_path)
    processed_image = rule_processor.run(rule)
    
    # Set ground truth jika diberikan
    if ground_truth:
        self.set_ground_truth(ground_truth)
        
    # Jalankan OCR
    ocr_result = self.tesseract(processed_image)
    
    # Jika ingin langsung mendapatkan metrics
    if self.ground_truth and ocr_result:
        return self.get_test()