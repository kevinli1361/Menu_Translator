# 🧾 Menu Translator  

## 🧩 Problem  
In my daily work, I often need to translate restaurant menus for my supervisor.  
Manually doing this is time-consuming and repetitive, so I developed this application to improve translation efficiency and streamline the workflow.  

## 🎯 Ultimate Goal  
Enable users to instantly convert menus into Chinese while preserving the original layout and design.

The menu can come from:  
- 📷 a phone camera photo  
- 💻 a desktop or mobile screenshot  
- 🖨️ a scanned image  

The goal is to make the translated version look **natural, accurate, and visually identical** to the original menu.  

## 🪜 Development Progress  

### ✅ First Step (completed)  
Implemented the `MenuImagePreprocessor` class in `menu_translator.py`, which handles menu image preprocessing.  

This module can perform: (but does not have to every single time)  
- Grayscale conversion  
- Binarization  
- Deskewing  
- Resizing  
- Denoising  
- Contrast enhancement  
- Morphological operations  

I use **pipelines** for actually processing those images:
- Minimal:      grayscale -> binarize(adaptive)
- Standard:     grayscale -> denoise(gaussian) -> contrast -> binarize(adaptive)
- Aggressive:   deskew -> grayscale -> denoise(bilateral) -> contrast -> binarize(otsu) -> morph(close)
- Custom:       for experimentation and future machine learning purposes

Note(10/28/2025): I add a "Try Everything" mode where you can create 320 test results from different combinations of processing methods.

### ✅ Second Step (completed)  
Developed the `SimpleOCR` class in `simple_ocr.py`, which extracts Chinese text from menu images and saves the output to **`menu_extracted.txt`**.  

### ⏳ Third Step (pending)  
Next, I plan to parse the extracted text into three categories:
**dish name**
**description**
**price**
These will be then translated and exported to **`menu_translated.txt`**.  

## 🚀 Ideas for Future Development  

- 🔁 **Pipeline Customization & Evaluation**  
  I'll experiment with different combinations of preprocessing methods.
  Then, compare the OCR output against a manually verified “correct answer” stored in **`correct_answer.txt`**, and calculate accuracy scores.  
  Ultimately, the system could **learn automatically** which preprocessing combinations yield the best recognition results (hopefully a self-optimizing, ML-driven pipeline).  

- 🖋️ **Visual Integration Enhancement**  
  Develop an image rendering module that overlays translated text **seamlessly** on top of the original menu image.
  Avoid visible artifacts (like the faint overlay line used in Google Translate).  
  The goal is to achieve **perfect visual blending** while keeping both the content and aesthetics intact.  

## 📂 Project Structure  

menu-translator
├── menu_translator.py      Image preprocessing (grayscale, binarization, etc.)
├── simple_ocr.py           OCR text extraction logic
├── menu_extracted.txt      Extracted raw text from menu images
├── menu_translated.txt     (Pending) Translated output
├── correct_answer.txt      (Optional) Ground truth for evaluation
├── README.md               Project documentation
└── sharps1.png             test menu image from Sharps Roasthouse

## 🧠 Tech Stack  
- Python (cannot be the latest 3.14. I spent too much time figuring out it doesn't work well with numpy)
- OpenCV (dependent on numpy)
- Tesseract OCR
- Pillow (PIL)
- NumPy
- uv (virtual environment management tool)

## 📫 Contact  
👤 **Kevin Li**  
If you have suggestions or want to contribute, feel free to open an issue or pull request.  


