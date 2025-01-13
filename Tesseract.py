import pytesseract
from typing import Tuple, Optional
import numpy as np
import jiwer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Tesseract:
    def __init__(self, input_path: str = None, output_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.ground_truth = None
        self.predicted_text = None

    def set_ground_truth(self, ground_truth: str):
        """Set ground truth text for comparison"""
        self.ground_truth = ground_truth
      
    
    def original_tesseract(self, image) -> str:
        """Perform OCR on the image and return the text"""
        text = pytesseract.image_to_string(image, lang='eng')
        return text

    def tesseract(self, image) -> str:
        """Perform OCR on the image and return the text"""
        try:
            # bahasa sundanese
            text = pytesseract.image_to_string(image, lang='eng')
            # Perform OCR
            self.predicted_text = ' '.join(text.lower().split())
            
            # Save to output file if path is provided
            if self.output_path:
                with open(self.output_path, 'w', encoding='utf-8') as f:
                    f.write(self.predicted_text)
            
            return self.predicted_text
        except Exception as e:
            print(f"Error during OCR: {e}")
            return ""

    def get_test(self) -> Tuple[float, float]:
        """Calculate WER and CER if ground truth is available"""
        if self.ground_truth is None or self.predicted_text is None:
            return 0.0, 0.0  # Return zeros if no ground truth or prediction

        try:
            transforms = jiwer.Compose([
              jiwer.ToLowerCase(),
              jiwer.RemoveWhiteSpace(replace_by_space=True),
              jiwer.RemoveMultipleSpaces(),
              jiwer.Strip()
            ])
            
            truth_processed = transforms(self.ground_truth)
            pred_processed = transforms(self.predicted_text)
            
            # Calculate WER
            wer = jiwer.wer(truth_processed, pred_processed)
            
            # Calculate CER
            cer = jiwer.cer(truth_processed, pred_processed)
            
            return wer, cer
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return 0.0, 0.0

    def calculate_word_level_metrics(self) -> Tuple[float, float, float, float]:
        """Calculate accuracy, precision, recall, and F1 score at word level"""
        if self.ground_truth is None or self.predicted_text is None:
            return 0.0, 0.0, 0.0, 0.0

        try:
            ground_truth_clean = ' '.join(self.ground_truth.lower().split())
            predicted_text_clean = ' '.join(self.predicted_text.lower().split())
            # Split texts into words
            ground_truth_words = ground_truth_clean.split()  
            predicted_words = predicted_text_clean.split()

            # Pad the shorter list with empty strings to match lengths
            max_length = max(len(ground_truth_words), len(predicted_words))
            ground_truth_words.extend([''] * (max_length - len(ground_truth_words)))
            predicted_words.extend([''] * (max_length - len(predicted_words)))

            # Create binary lists for exact word matches
            y_true = [1 if word != '' else 0 for word in ground_truth_words]
            y_pred = [1 if gt == pred and gt != '' else 0 
                     for gt, pred in zip(ground_truth_words, predicted_words)]

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            return accuracy, precision, recall, f1
        except Exception as e:
            print(f"Error calculating word-level metrics: {e}")
            return 0.0, 0.0, 0.0, 0.0

    def get_metrics(self) -> dict:
        """Get all available metrics"""
        wer, cer = self.get_test()
        accuracy, precision, recall, f1 = self.calculate_word_level_metrics()
        
        return {
            'wer': wer * 100,  # Convert to percentage
            'cer': cer * 100,  # Convert to percentage
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'predicted_text': self.predicted_text
        }
