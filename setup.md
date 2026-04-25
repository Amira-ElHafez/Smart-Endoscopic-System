# Intelligent Endoscopic Assistance System
### SBE3220 — Medical Equipment II

---

## Project Structure

```
endoscope/
├── main.py                  ← Entry point
├── output/                  ← Captured frames saved here
├── main_window.py       ← PyQt5 GUI (all subsystems wired together)
├── illumination.py      ← LED brightness/contrast control
├── image_processor.py   ← Noise reduction, CLAHE, edge detection
├── navigation.py        ← Pan/tilt simulation via keyboard
└── feature_extractor.py ← Shape, color, texture feature extraction
```

---

## Setup

### 1. Install dependencies
```bash
pip install PyQt5 opencv-python numpy
```

### 2. Run the application
```bash
python main.py
```


---

## How to Use

### Source Selection
| Button | Action |
|--------|--------|
| Connect Camera | Opens webcam (index 0). Falls back to test pattern if unavailable. |
| Load Video | Opens any .mp4 / .avi / .mov file. Loops automatically. |
| Load Image | Opens a still image (.png / .jpg / .bmp). |
| Stop | Stops the current source. |

### Navigation (Insertion Tube Simulation)
| Key | Action |
|-----|--------|
| W / ↑ | Pan up |
| S / ↓ | Pan down |
| A / ← | Pan left |
| D / → | Pan right |
| R | Reset to center |
| Space | Capture current frame |

You can also use the on-screen arrow buttons in the Navigation panel.

### Illumination
- **Brightness slider** → simulates LED intensity (0–200%)
- **Contrast slider** → adjusts image contrast (0–200%)

### Image Processing (Bonus)
| Option | Method |
|--------|--------|
| Noise Reduction | Gaussian blur (cv2.GaussianBlur) |
| CLAHE Enhancement | Adaptive histogram eq. in LAB space |
| Edge Detection | Canny edges overlaid in cyan |
| Pseudo-color map | Jet / Hot / Cool / Bone / HSV colormaps |

### Feature Extraction
Click **Extract Features** to compute:

**Shape:** contour count, largest contour area, circularity (0=line, 1=perfect circle)

**Color:** mean BGR values, dominant hue (degrees), mean saturation

**Texture (LBP):** LBP energy, LBP entropy, texture contrast (gradient magnitude)

Captured frames are saved to the `output/` folder with a timestamp filename.

---

## Subsystem Map

```
[Camera / Video / Image / Test Pattern]
        ↓
[Illumination Controller]  ← Brightness & Contrast sliders
        ↓
[Navigation Controller]    ← Keyboard / buttons → pan crop
        ↓
[Image Processor]          ← Denoise → CLAHE → Edges → Colormap
        ↓
[PyQt5 Display]            ← Original (left) | Processed (right)
        ↓
[Feature Extractor]        ← Shape + Color + Texture on demand
```