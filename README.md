# Intelligent Endoscopic Assistance System (IEAS) 🩺🔬

## 📌 Project Overview
[cite_start]This project is a functional simulation of a modern **Medical Endoscope System**, developed as part of the **Medical Equipment (II) [SBE3220]** course[cite: 1, 3, 4]. [cite_start]The system integrates real-time video stream manipulation, illumination control, and advanced digital image processing to assist clinicians in visualizing and analyzing internal body cavities[cite: 7, 19, 31].

[cite_start]Built using **Python**, **OpenCV**, and **PyQt5**, the application provides a professional GUI that simulates the physical controls of an endoscopic insertion tube and light source, alongside automated feature extraction for diagnostic support[cite: 27, 34].

---

## ✨ Key Features

### 1. Imaging & Navigation Subsystem 🕹️
* [cite_start]**Multi-Source Input:** Supports live camera feeds, video file playback, and static image analysis[cite: 9, 10, 19].
* [cite_start]**Virtual Insertion Tube:** Simulates 4-way movement (Pan/Tilt) and dynamic Zoom (1.0x to 3.0x) using keyboard (WASD/Arrows) or GUI controls[cite: 11, 14].
* [cite_start]**Real-time Display:** Dual-view interface showing the raw camera feed alongside the processed medical output[cite: 18, 19].

### 2. Illumination Control 💡
* [cite_start]**Intensity Modulation:** Simulates LED light source control by adjusting brightness and contrast levels to optimize visualization in different anatomical environments[cite: 6, 7, 16].

### 3. Advanced Image Processing (Bonus) 🛠️
[cite_start]To enhance diagnostic clarity, the system includes a modular processing pipeline[cite: 31]:
* [cite_start]**Denoising:** Gaussian filtering to reduce sensor noise[cite: 32].
* [cite_start]**Contrast Boost:** Implementation of **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for better visibility of tissue textures[cite: 33].
* **Edge Detection:** Canny edge detection to highlight vessel boundaries or lesions.
* **Pseudo-Coloring:** Multiple colormaps (Jet, Hot, Cool, Bone, HSV) to aid in thermal/medical heatmap visualization.

### 4. Intelligent Feature Extraction 🧠
[cite_start]The system automatically extracts and displays quantitative data from the feed[cite: 34]:
* [cite_start]**Shape Features:** Contour counting, largest area detection, and circularity metrics[cite: 35].
* [cite_start]**Color Features:** Mean BGR values, dominant hue detection, and saturation analysis[cite: 36].
* [cite_start]**Texture Analysis:** Local Binary Patterns (LBP) calculation, providing Energy, Entropy, and Contrast data[cite: 37].
* **Heatmap Generation:** A specialized texture-based heatmap highlighting areas of high visual interest (e.g., polyps or vessels).

---

## 🛠️ System Architecture
The project is structured into modular components for scalability:
* `main_window.py`: The central GUI and logic controller.
* `image_processor.py`: Contains enhancement algorithms (CLAHE, Denoising).
* `feature_extractor.py`: Handles mathematical analysis of shape, color, and texture.
* `navigation.py`: Manages the mathematical transformations for pan, tilt, and zoom.
* `illumination.py`: Simulates the light source intensity logic.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.x
* OpenCV (`opencv-python`)
* PyQt5
* NumPy

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Endoscopic-Assistance-System.git](https://github.com/YourUsername/Endoscopic-Assistance-System.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install opencv-python PyQt5 numpy
    ```
3.  **Run the application:**
    ```bash
    python main.py
    ```

---

## ⌨️ Controls Reference
| Action | Key / Control |
| :--- | :--- |
| **Move Up/Down/Left/Right** | `W`, `S`, `A`, `D` or Arrow Keys |
| **Zoom In / Out** | `+` / `-` |
| **Reset View** | `R` key |
| **Capture Frame** | `Space` bar |

---

## 📊 Project Results
The system provides critical diagnostic outputs:
* **Enhanced Visualization:** Clearer imaging of internal tissues through real-time contrast adjustment and noise reduction.
* **Diagnostic Mapping:** Automated heatmap generation identifying potential areas of interest like polyps or lesions based on texture density.
* **Quantitative Metrics:** Real-time feedback on tissue characteristics (Color, Shape, Texture) to support clinical decision-making.

---
