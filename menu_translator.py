"""
Image Preprocessing Module - For Menu OCR Optimization
Suitable for English menu screenshots to improve OCR recognition accuracy
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from simple_ocr import SimpleOCR
from pathlib import Path


class MenuImagePreprocessor:
    """Menu Image Preprocessor"""
    
    def __init__(self, image_path):
        """Takes the path to the image file and initialize the preprocessor"""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"[x] Unable to load image: {image_path}")
        self.processed_image = self.original_image.copy()
        self.result_file_name = ""  # for file-naming purposes
    
    def convert_to_grayscale(self):
        """Convert image to grayscale image"""
        self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        print("[✓] Converted to grayscale")
        self.result_file_name += "_grayscale"
        return self.processed_image
    
    def denoise(self, method='gaussian', kernel_size=5):
        """
        Apply noise reduction
        
        Args:
            method: Denoising method ('gaussian', 'median', 'bilateral')
            kernel_size: Kernel size (must be odd number)
        
        Returns:
            Denoised image

        Notes:
            - Gaussian mode: smooth and natural blur, fastest
            - Median mode: for salt-and-pepper noise, medium speed
            - Bilateral mode: denoise in smooth area and still keep words sharp, slowest
        """
        if len(self.processed_image.shape) == 3:
            # If color image, convert to grayscale first
            self.convert_to_grayscale()
        
        if method == 'gaussian':
            # Gaussian blur - suitable for general noise
            self.processed_image = cv2.GaussianBlur(
                self.processed_image, 
                (kernel_size, kernel_size), 
                0
            )
            self.result_file_name += "_denoise(gaussian)"
        elif method == 'median':
            # Median filter - suitable for salt-and-pepper noise
            self.processed_image = cv2.medianBlur(self.processed_image, kernel_size)
            self.result_file_name += "_denoise(median)"
        elif method == 'bilateral':
            # Bilateral filter - edge-preserving denoising
            self.processed_image = cv2.bilateralFilter(
                self.processed_image, 
                kernel_size, 
                75, 
                75
            )
            self.result_file_name += "_denoise(bilateral)"
        
        print(f"[✓] Completed {method} denoising")
        return self.processed_image
    
    def binarize(self, method='adaptive', block_size=11, c=2):
        """
        Apply binarization (thresholding)
        
        Args:
            method: Binarization method ('simple', 'adaptive', 'otsu')
            block_size: Block size for adaptive threshold (must be odd number)
            c: Constant subtracted from weighted mean for adaptive threshold
        
        Returns:
            Binarized image

        Notes:
            - Simple: one standard for the whole picture
            - Adaptive: divide pics into areas, every area has different standard
            - Otsu: find the best standard for the whole picture
        """
        if len(self.processed_image.shape) == 3:
            self.convert_to_grayscale()
        
        if method == 'simple':
            # Simple threshold
            _, self.processed_image = cv2.threshold(
                self.processed_image, 
                127, 
                255, 
                cv2.THRESH_BINARY
            )
            self.result_file_name += "_binarize(simple)"
        elif method == 'adaptive':
            # Adaptive threshold - suitable for uneven lighting
            self.processed_image = cv2.adaptiveThreshold(
                self.processed_image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c
            )
            self.result_file_name += "_binarize(adaptive)"
        elif method == 'otsu':
            # Otsu's automatic threshold
            _, self.processed_image = cv2.threshold(
                self.processed_image,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            self.result_file_name += "_binarize(otsu)"
        
        print(f"[✓] Completed {method} binarization")
        return self.processed_image
    
    def enhance_contrast(self):
        """Enhance contrast using CLAHE, returns contrast-enhanced image"""
        if len(self.processed_image.shape) == 3:
            self.convert_to_grayscale()
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.processed_image = clahe.apply(self.processed_image)
        
        print("[✓] Enhanced contrast")
        self.result_file_name += "_contrast"
        return self.processed_image
    
    def deskew(self):
        """Correct image skew/tilt, returns deskewed image"""
        if len(self.processed_image.shape) == 3:
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.processed_image.copy()
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is not None:
            # Calculate average angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                if -45 < angle < 45:  # Only consider near-horizontal lines
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                
                # Rotate image
                (h, w) = self.processed_image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                self.processed_image = cv2.warpAffine(
                    self.processed_image,
                    rotation_matrix,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                
                print(f"[✓] Corrected skew angle: {median_angle:.2f}°")
                self.result_file_name += "_deskew"
            else:
                print("⚠ No significant skew detected")
        else:
            print("⚠ No lines detected, skipping deskew")
        
        return self.processed_image
    
    def resize(self, scale_factor=None, max_width=None, max_height=None):
        """
        Resize the image
        
        Args:
            scale_factor: Scaling factor (e.g., 0.5 means half size)
            max_width: Maximum width
            max_height: Maximum height
        
        Returns:
            Resized image
        """
        h, w = self.processed_image.shape[:2]
        
        if scale_factor:
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
        elif max_width or max_height:
            # Calculate scaling to fit within max dimensions
            scale_w = max_width / w if max_width else float('inf')
            scale_h = max_height / h if max_height else float('inf')
            scale = min(scale_w, scale_h, 1.0)  # Don't upscale
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            return self.processed_image
        
        self.processed_image = cv2.resize(
            self.processed_image, 
            (new_w, new_h), 
            interpolation=cv2.INTER_AREA
        )
        
        print(f"[✓] Resized to {new_w}x{new_h}")
        self.result_file_name += "_resize"
        return self.processed_image
    
    def morphological_operations(self, operation='close', kernel_size=3):
        """
        Apply morphological operations
        
        Args:
            operation: Operation type ('dilate', 'erode', 'open', 'close')
            kernel_size: Kernel size
        
        Returns:
            Processed image
        
        Notes:
            - Dilate: for broken letters
            - Erode: for letters that are too thick
            - Open: erode -> dilate
            - Close: dilate -> erode
        """
        if len(self.processed_image.shape) == 3:
            self.convert_to_grayscale()
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'dilate':
            # Make text thicker
            self.processed_image = cv2.dilate(self.processed_image, kernel, iterations=1)
            self.result_file_name += "_morph(dilate)"
        elif operation == 'erode':
            # Make text thinner
            self.processed_image = cv2.erode(self.processed_image, kernel, iterations=1)
            self.result_file_name += "_morph(erode)"
        elif operation == 'open':
            # Remove small noise (erosion then dilation)
            self.processed_image = cv2.morphologyEx(self.processed_image, cv2.MORPH_OPEN, kernel)
            self.result_file_name += "_morph(open)"
        elif operation == 'close':
            # Fill small holes (dilation then erosion)
            self.processed_image = cv2.morphologyEx(self.processed_image, cv2.MORPH_CLOSE, kernel)
            self.result_file_name += "_morph(close)"
        
        print(f"[✓] Applied {operation} morphological operation")
        return self.processed_image
    
    def preprocess_pipeline(self, pipeline='standard', custom_options=None):
        """
        Run a complete preprocessing pipeline
        
        Args:
            pipeline: Pipeline type ('standard', 'aggressive', 'minimal')
        
        Returns:
            Fully preprocessed image

        Notes:
            - Minimal:          grayscale -> binarize(adaptive)
            - Standard:         grayscale -> denoise(gaussian) -> contrast -> binarize(adaptive)
            - Aggressive:       deskew -> grayscale -> denoise(bilateral) -> contrast -> binarize(otsu) -> morph(close)
            - Custom:           user options
        """
        print(f"\n--- Running {pipeline} preprocessing pipeline ---")
        
        if pipeline == 'minimal':
            # For high-quality screenshots
            self.convert_to_grayscale()
            self.binarize(method='adaptive')
        
        elif pipeline == 'standard':
            # For typical menu photos
            self.convert_to_grayscale()
            self.denoise(method='gaussian', kernel_size=3)
            self.enhance_contrast()
            self.binarize(method='adaptive', block_size=11, c=2)
        
        elif pipeline == 'aggressive':
            # For poor quality or skewed images
            self.deskew()
            self.convert_to_grayscale()
            self.denoise(method='bilateral', kernel_size=5)
            self.enhance_contrast()
            self.binarize(method='otsu')
            self.morphological_operations('close', kernel_size=2)

        elif pipeline == 'custom':
            # Primarily for experimentation
            # Deskew -> Grayscale -> Denoise -> Contrast -> Binarize -> Morph
            print("\n--- Custom Pipeline (Experimental Mode) ---")
    
    
            # Helper function for yes/no questions
            def ask_yes_no(prompt):
                return input(f" * {prompt} (y/n) Your Answer: ").lower() == 'y'

            # Helper function for options questions
            def ask_option(prompt, options_dict):
                option = input(f" ** {prompt} Your Answer: ").lower()
                if option in options_dict:
                    return options_dict[option]
                else:
                    print("[x] Invalid option, skipping this step")
                    return None
            
            # Deskew
            if ask_yes_no("Do you choose to deskew this image?"):
                self.deskew()
    

            # Grayscale
            self.convert_to_grayscale()
    
            # Denoise
            if ask_yes_no("Do you choose to denoise this image?"):
                method = ask_option(
                    "How do you want to denoise? (g=gaussian, m=median, b=bilateral)",
                    {'g': 'gaussian', 'm': 'median', 'b': 'bilateral'}
                )
                if method:
                    self.denoise(method=method)
    
            # Contrast
            if ask_yes_no("Do you choose to enhance the contrast?"):
                self.enhance_contrast()
    
            # Binarize
            if ask_yes_no("Do you choose to binarize this image?"):
                method = ask_option(
                    "How do you want to binarize? (s=simple, a=adaptive, o=otsu)",
                    {'s': 'simple', 'a': 'adaptive', 'o': 'otsu'}
                )

                if method:
                    self.binarize(method=method)
    
            # Morphology
            if ask_yes_no("Do you choose to apply morphological operations?"):
                operation = ask_option(
                    "Which operation? (d=dilate, e=erode, o=open, c=close)",
                    {'d': 'dilate', 'e': 'erode', 'o': 'open', 'c': 'close'}
                )
                if operation:
                    self.morphological_operations(operation=operation)
                

        print("--- Preprocessing complete ---\n")
        return self.processed_image
    
    def save_image(self, output_path):
        """
        Save the processed image
        
        Args:
            output_path: Path to save the image
        """
        cv2.imwrite(output_path, self.processed_image)
        print(f"[✓] Saved processed image to: {output_path}")
    
    def show_comparison(self):
        """
        Display original vs processed image side by side
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        if len(self.original_image.shape) == 3:
            axes[0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(self.original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Processed image
        if len(self.processed_image.shape) == 3:
            axes[1].imshow(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
        else:
            axes[1].imshow(self.processed_image, cmap='gray')
        axes[1].set_title('Processed Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Get filename from user input
    #filename = input("Please enter the filename: ")

    # for testing purpose
    filename = 'sharps1.png'

    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found in current directory")
        exit()
    
    # Check format compatibility
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    _, ext = os.path.splitext(filename.lower())
    
    if ext not in supported_formats:
        print(f"Error: Format '{ext}' is not supported")
        print(f"Supported formats: {', '.join(supported_formats)}")
        exit()
    
    # Define where OCR results will go
    result_folder = Path("test_results")
    result_folder.mkdir(exist_ok=True)
    
    # Initialize preprocessor
    print(f"\nProcessing '{filename}'...")
    
    # Let's try 320 options!!
    if input(" *** Do you wish to try every single possibility? (y/n) Your Answer: ").lower() == 'y':
        all_options = []
        
        deskew_options = [True, False]
        denoise_options = ['gaussian', 'median', 'bilateral', None]
        contrast_options = [True, False]
        binarize_options = ['simple', 'adaptive', 'otsu', None]
        morph_options = ['dilate', 'erode', 'open', 'close', None]
        
        for s in deskew_options:
            for n in denoise_options:
                for c in contrast_options:
                    for b in binarize_options:
                        for m in morph_options:
                            all_options.append([s,n,c,b,m])
        
        print(f'[✓✓✓✓✓...] number of options created: {len(all_options)} options')

        for deskew_opt, denoise_opt, contrast_opt, binarize_opt, morph_opt in all_options:
            preprocessor = MenuImagePreprocessor(filename)
            
            if deskew_opt:
                preprocessor.deskew()
            
            if denoise_opt:
                preprocessor.denoise(method=denoise_opt)

            if contrast_opt:
                preprocessor.enhance_contrast()

            if binarize_opt:
                preprocessor.binarize(method=binarize_opt)

            if morph_opt:
                preprocessor.morphological_operations(operation=morph_opt)
            
            ocr = SimpleOCR()

            cleaned_text = ocr.extract_and_clean(preprocessor.processed_image, lang='eng')

            result_file_name = f"{preprocessor.result_file_name[1:]}.txt"
            result_file_path = result_folder / result_file_name
            ocr.save_text(cleaned_text, result_file_path)
            print(f'--- {result_file_name} saved ---')


    # original code, keep intact
    else:
        preprocessor = MenuImagePreprocessor(filename)
        # Run preprocessing pipeline
        processed = preprocessor.preprocess_pipeline(pipeline='custom')
    
        # Generate output filename
        name_without_ext = os.path.splitext(filename)[0]
        output_filename = f"{name_without_ext}_processed{ext}"
    
        # Save result
        preprocessor.save_image(output_filename)
    
        # Show comparison
        ###preprocessor.show_comparison()

        ocr = SimpleOCR()
    
        # Check supported languages
        ocr.get_supported_languages()
    
        # Method 1: Simple text extraction
        print("\n--- Method 1: Simple Extraction ---")
        text = ocr.extract_text(output_filename, lang='eng')
        print(text)

        # Method 2: Extract with confidence scores
        print("\n--- Method 2: With Confidence ---")
        results = ocr.extract_text_with_confidence(output_filename, lang='eng')
        for item in results[:10]:  # Show first 10 items
            print(f"{item['text']:20s} (confidence: {item['confidence']}%)")
    
        # Method 3: Extract and clean
        print("\n--- Method 3: Cleaned Text ---")
        cleaned = ocr.extract_and_clean(output_filename, lang='eng')
        print(cleaned)
    
        # Save to file
        result_file_name = f"{preprocessor.result_file_name[1:]}.txt"
        result_file_path = result_folder / result_file_name
        ocr.save_text(cleaned, result_file_path)
    