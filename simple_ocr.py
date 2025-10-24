"""
Simple OCR Text Extraction Module
Extracts text from preprocessed menu images
"""

import pytesseract
from PIL import Image


class SimpleOCR:
    """Simple OCR text extractor for menu images"""
    
    def __init__(self):
        """
        Initialize OCR
        Sets default Tesseract path for Windows
        """

        # Set Tesseract path for Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Test if Tesseract is available
        try:
            version = pytesseract.get_tesseract_version()
            print(f"[✓] Tesseract {version} detected")
        except pytesseract.TesseractNotFoundError:
            raise Exception("[x] Tesseract not found. Please install Tesseract OCR.")
        except FileNotFoundError:
            raise Exception("[x] Tesseract executable not in PATH.")
        except Exception as e:
            print(f"[✗] Unexpected error: {e}")
    

    def extract_text(self, image_path, lang='eng'):
        """
        Extract text from image
        
        Args:
            image_path: Path to image file (or cv2 image array)
            lang: Language code ('eng' for English, 'chi_tra' for Traditional Chinese)
        
        Returns:
            Extracted text as string
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path)
            else:
                # If cv2 image array, convert to PIL
                image = Image.fromarray(image_path)
            
            # Extract text
            text = pytesseract.image_to_string(image, lang=lang)
            
            print(f"[✓] Text extraction completed ({len(text)} characters)")
            return text
        
        except FileNotFoundError:
            print(f"[✗] Error: Image file not found: {image_path}")
            return ""
        except Exception as e:
            print(f"[✗] OCR Error: {e}")
            return ""
    
    def extract_text_with_confidence(self, image_path, lang='eng'):
        """
        Extract text with confidence scores
        
        Args:
            image_path: Path to image file
            lang: Language code
        
        Returns:
            Dictionary with text and confidence data
        """
        try:
            if isinstance(image_path, str):
                image = Image.open(image_path)
            else:
                image = Image.fromarray(image_path)
            
            # Get detailed data
            data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
            
            # Filter out empty text and low confidence
            results = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if text and conf > 0:  # Only include text with confidence > 0
                    results.append({
                        'text': text,
                        'confidence': conf,
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    })
            
            print(f"[✓] Extracted {len(results)} text elements")
            return results
        
        except Exception as e:
            print(f"[✗] OCR Error: {e}")
            return []
    
    def save_text(self, text, output_path):
        """
        Save extracted text to file
        
        Args:
            text: Text to save
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"[✓] Saved text to: {output_path}")
        except Exception as e:
            print(f"[✗] Error saving file: {e}")
    
    def extract_and_clean(self, image_path, lang='eng'):
        """
        Extract text and apply basic cleaning
        
        Args:
            image_path: Path to image file
            lang: Language code
        
        Returns:
            Cleaned text
        """
        text = self.extract_text(image_path, lang)
        
        # Basic cleaning
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Remove empty lines
                cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        print(f"[✓] Cleaned text: {len(cleaned_lines)} lines")
        
        return cleaned_text
    
    def get_supported_languages(self):
        """
        Get list of installed languages
        
        Returns:
            List of language codes
        """
        try:
            langs = pytesseract.get_languages()
            print(f"Supported languages: {', '.join(langs)}")
            return langs
        except Exception as e:
            print(f"[✗] Error getting languages: {e}")
            return []


# Example usage
if __name__ == "__main__":
    ocr = SimpleOCR()
    
    # Check supported languages
    ocr.get_supported_languages()
    
    # Method 1: Simple text extraction
    print("\n--- Method 1: Simple Extraction ---")
    text = ocr.extract_text("menu_processed.jpg", lang='eng')
    print(text)
    
    # Method 2: Extract with confidence scores
    print("\n--- Method 2: With Confidence ---")
    results = ocr.extract_text_with_confidence("menu_processed.jpg", lang='eng')
    for item in results[:10]:  # Show first 10 items
        print(f"{item['text']:20s} (confidence: {item['confidence']}%)")
    
    # Method 3: Extract and clean
    print("\n--- Method 3: Cleaned Text ---")
    cleaned = ocr.extract_and_clean("menu_processed.jpg", lang='eng')
    print(cleaned)
    
    # Save to file
    ocr.save_text(cleaned, "menu_extracted.txt")