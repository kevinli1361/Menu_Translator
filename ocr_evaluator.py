"""
OCR Result Evaluator and Optimizer
Compares OCR results against ground truth and finds the best preprocessing pipeline
"""

import difflib
import os
import glob

class OCREvaluator:
    """Evaluates OCR accuracy against ground truth"""
    
    def __init__(self, ground_truth_path):
        """Initialize evaluator with the correct answer file path"""
        self.ground_truth = self.load_text(ground_truth_path)
        print(f"[✓] Loaded ground truth: {len(self.ground_truth)} characters")
    

    def load_text(self, file_path):
        """Load text from file and return text content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except FileNotFoundError:
            print(f"[✗] Error: File not found: {file_path}")
            return ""
        except Exception as e:
            print(f"[✗] Error loading file: {e}")
            return ""
    

    def calculate_character_accuracy(self, ocr_result):
        """
        Calculate character-level accuracy using edit distance
        Args: ocr_result (OCR extracted text)
        Returns: Dictionary with accuracy metrics
        """
        # Use SequenceMatcher to calculate similarity
        matcher = difflib.SequenceMatcher(None, self.ground_truth, ocr_result)
        ratio = matcher.ratio()
        
        # Calculate edit distance (Levenshtein distance approximation)
        opcodes = matcher.get_opcodes()
        errors = sum(1 for tag, _, _, _, _ in opcodes if tag != 'equal')
        
        return {
            'similarity_ratio': ratio * 100,  # 0-100%
            'character_accuracy': ratio * 100,
            'edit_distance': errors,
            'ground_truth_length': len(self.ground_truth),
            'ocr_result_length': len(ocr_result)
        }
    
    def calculate_word_accuracy(self, ocr_result):
        """Takes OCR extracted text, calculate word-level accuracy, and returns percentage"""
        # Split into words
        ground_truth_words = set(self.ground_truth.lower().split())
        ocr_words = set(ocr_result.lower().split())
        
        # Calculate intersection
        correct_words = ground_truth_words & ocr_words
        total_words = len(ground_truth_words)
        
        if total_words == 0:
            return 0.0
        
        word_accuracy = (len(correct_words) / total_words) * 100
        
        return {
            'word_accuracy': word_accuracy,
            'correct_words': len(correct_words),
            'total_words': total_words,
            'missing_words': list(ground_truth_words - ocr_words)[:10]  # Show first 10
        }
    
    def calculate_line_accuracy(self, ocr_result):
        """Takes OCR extracted text, calculate line-by-line accuracy, and returns line accuracy metrics"""

        ground_truth_lines = [line.strip() for line in self.ground_truth.split('\n') if line.strip()]
        ocr_lines = [line.strip() for line in ocr_result.split('\n') if line.strip()]
        
        # Match lines
        correct_lines = 0
        for gt_line in ground_truth_lines:
            for ocr_line in ocr_lines:
                similarity = difflib.SequenceMatcher(None, gt_line, ocr_line).ratio()
                if similarity > 0.9:  # 90% similar
                    correct_lines += 1
                    break
        
        line_accuracy = (correct_lines / len(ground_truth_lines)) * 100 if ground_truth_lines else 0
        
        return {
            'line_accuracy': line_accuracy,
            'correct_lines': correct_lines,
            'total_lines': len(ground_truth_lines)
        }
    
    def evaluate(self, ocr_result, pipeline_name="unknown"):
        """
        Comprehensive evaluation
        
        Args:
            ocr_result: OCR extracted text
            pipeline_name: Name of the preprocessing pipeline used
        
        Returns:
            Complete evaluation metrics
        """
        char_metrics = self.calculate_character_accuracy(ocr_result)
        word_metrics = self.calculate_word_accuracy(ocr_result)
        line_metrics = self.calculate_line_accuracy(ocr_result)
        
        # Overall score (weighted average)
        overall_score = (
            char_metrics['character_accuracy'] * 0.5 +
            word_metrics['word_accuracy'] * 0.3 +
            line_metrics['line_accuracy'] * 0.2
        )
        
        results = {
            'pipeline': pipeline_name,
            'overall_score': overall_score,
            **char_metrics,
            **word_metrics,
            **line_metrics
        }
        
        return results
    
    def print_evaluation(self, results):
        """
        Print evaluation results in a readable format
        
        Args:
            results: Evaluation results dictionary
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS: {results['pipeline']}")
        print(f"{'='*60}")
        print(f"Overall Score:          {results['overall_score']:.2f}%")
        print("\nCharacter Level:")
        print(f"  Similarity Ratio:     {results['similarity_ratio']:.2f}%")
        print(f"  Edit Distance:        {results['edit_distance']}")
        print(f"  Length (GT/OCR):      {results['ground_truth_length']}/{results['ocr_result_length']}")
        print("\nWord Level:")
        print(f"  Word Accuracy:        {results['word_accuracy']:.2f}%")
        print(f"  Correct Words:        {results['correct_words']}/{results['total_words']}")
        if results['missing_words']:
            print(f"  Missing Words:        {', '.join(results['missing_words'][:5])}")
        print("\nLine Level:")
        print(f"  Line Accuracy:        {results['line_accuracy']:.2f}%")
        print(f"  Correct Lines:        {results['correct_lines']}/{results['total_lines']}")
        print(f"{'='*60}\n")
    
    def compare_results(self, results_list):
        """
        Compare multiple OCR results and rank them
        
        Args:
            results_list: List of evaluation result dictionaries
        
        Returns:
            Sorted list of results (best first)
        """
        sorted_results = sorted(results_list, key=lambda x: x['overall_score'], reverse=True)
        
        print(f"\n{'='*60}")
        print("RANKING OF PREPROCESSING PIPELINES")
        print(f"{'='*60}")
        print(f"{'Rank':<6} {'Pipeline':<20} {'Score':<10} {'Char%':<10} {'Word%':<10}")
        print(f"{'-'*60}")
        
        for i, result in enumerate(sorted_results, 1):
            print(f"{i:<6} {result['pipeline']:<20} {result['overall_score']:>6.2f}%   "
                  f"{result['character_accuracy']:>6.2f}%   {result['word_accuracy']:>6.2f}%")
        
        print(f"{'='*60}")
        print(f"\n🏆 BEST PIPELINE: {sorted_results[0]['pipeline']} "
              f"(Score: {sorted_results[0]['overall_score']:.2f}%)\n")
        
        return sorted_results
    
    def save_comparison(self, results_list, output_path="comparison_results.txt"):
        """
        Save comparison results to file
        
        Args:
            results_list: List of evaluation results
            output_path: Output file path
        """
        sorted_results = sorted(results_list, key=lambda x: x['overall_score'], reverse=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("OCR PIPELINE COMPARISON RESULTS\n")
                f.write("="*60 + "\n\n")
                
                for i, result in enumerate(sorted_results, 1):
                    f.write(f"Rank {i}: {result['pipeline']}\n")
                    f.write(f"  Overall Score: {result['overall_score']:.2f}%\n")
                    f.write(f"  Character Accuracy: {result['character_accuracy']:.2f}%\n")
                    f.write(f"  Word Accuracy: {result['word_accuracy']:.2f}%\n")
                    f.write(f"  Line Accuracy: {result['line_accuracy']:.2f}%\n")
                    f.write(f"  Edit Distance: {result['edit_distance']}\n")
                    f.write("\n")
                
                f.write(f"Best Pipeline: {sorted_results[0]['pipeline']}\n")
            
            print(f"✓ Saved comparison results to: {output_path}")
        except Exception as e:
            print(f"✗ Error saving results: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize evaluator with ground truth
    evaluator = OCREvaluator("correct_answer.txt")
    
    # Automatically read all txt files from test_results folder
    test_results_folder = "test_results"
    
    # Check if folder exists
    if not os.path.exists(test_results_folder):
        print(f"[✗] Error: Folder '{test_results_folder}' not found")
        print("Please create the folder and add your result files")
        exit()
    
    # Get all .txt files
    txt_files = glob.glob(os.path.join(test_results_folder, "*.txt"))
    
    if not txt_files:
        print(f"[✗] No .txt files found in '{test_results_folder}' folder")
        exit()
    
    print(f"[✓] Found {len(txt_files)} result files:")
    for file_path in txt_files:
        print(f"  - {os.path.basename(file_path)}")
    print()
    
    # Evaluate all files
    results = []
    for file_path in txt_files:
        # Extract pipeline name from filename
        pipeline_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load and evaluate
        ocr_result = evaluator.load_text(file_path)
        if ocr_result:
            result = evaluator.evaluate(ocr_result, pipeline_name)
            evaluator.print_evaluation(result)
            results.append(result)
    
    # Compare all results
    if results:
        best_results = evaluator.compare_results(results)
        evaluator.save_comparison(results)
    
    
    # Or evaluate a single result directly
    # ocr_text = evaluator.load_text("my_ocr_result.txt")
    # result = evaluator.evaluate(ocr_text, "my_pipeline")
    # evaluator.print_evaluation(result)