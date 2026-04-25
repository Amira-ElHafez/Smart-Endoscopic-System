# Walkthrough: Intelligent Endoscopic Assistance System

I have successfully designed and built the Intelligent Endoscopic Assistance System using Python. The application provides a complete simulation environment that fulfills all your project requirements, including the bonus criteria.




## What was built
1. **Core Application (`main.py`)**: A modern, responsive graphical interface built with `PyQt5`.
2. **Endoscope Engine (`endoscope_engine.py`)**: The simulation logic that mimics hardware inputs. It handles the continuous rendering loop, simulates camera panning and zooming using affine transformations, and applies dynamic brightness adjustments to act as the simulated illumination system.
3. **Medical Image Processing Module (`processing.py`)**: Implements advanced OpenCV image enhancements specifically suited for endoscopic data:
   - **Noise Reduction**: Uses a Bilateral Filter to smooth noise while preserving critical edges.
   - **Contrast Enhancement**: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve vascular visibility.
   - **Feature Extraction**:
     - *Color Feature*: Detects and thresholds redness in the HSV space.
     - *Shape Feature*: Applies a Canny edge detector to identify object contours.
     - *Texture Feature*: Calculates and visualizes local variance, applying a Jet colormap.
4. **Sample Data Fetcher (`download_sample.py`)**: Generates a dummy endoscopic image to start with.

## How to Run It

All the dependencies have been installed inside a local virtual environment.
To launch the application, open your terminal (PowerShell) and run the following command:




```powershell
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt

# Launch the Application
python main.py
```

## How to Use the System

> [!TIP]
> The interface is divided into the video feed on the left and the control panels on the right.

1. **Load Media**: Click the "Load Media (Image/Video)" button and select the `sample_endoscopy.jpg` file (or any other image you want to use).
2. **Start/Stop Feed**: Click the "Start/Stop Feed" button to begin rendering the simulation.
3. **Illumination System**: Use the slider under the *Illumination System* panel to increase or decrease light intensity.
4. **Insertion Tube Navigation**:
   - Use the **Arrow Keys** (Left, Right, Up, Down) to pan the camera across the simulated tissue.
   - Use the **W** and **S** keys to zoom in and out.
5. **Image Processing (Bonus)**: 
   - Select a processing filter from the dropdown under *Image Processing System*. The video feed will update in real-time.
6. **Capture System**: Click the "Capture Image" button. This will save the currently displayed frame (along with any applied filters) as a new `capture_X.jpg` image in your folder.

## Verification
All Python modules have been built and syntactically verified. The packages are properly isolated within a local virtual environment, and a starting sample image has been synthesized so you don't start with an empty screen.
