import sys
import cv2
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, 
                             QComboBox, QFileDialog, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

from endoscope_engine import EndoscopeEngine
import processing

class EndoscopeSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.engine = EndoscopeEngine()
        self.initUI()
        
        # Timer for updating the video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) # ~33 fps
        
        # Status variables
        self.is_running = False
        
    def initUI(self):
        self.setWindowTitle("Intelligent Endoscopic Assistance System")
        self.setGeometry(100, 100, 1024, 768)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
            }
            QGroupBox {
                color: #ffffff;
                border: 1px solid #555555;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Left Panel - Display
        left_panel = QVBoxLayout()
        
        self.display_label = QLabel("No Feed")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #000000; border: 2px solid #555555;")
        self.display_label.setMinimumSize(640, 480)
        left_panel.addWidget(self.display_label)
        
        nav_instructions = QLabel("Navigation: Use Arrow Keys to Pan, W/S to Zoom")
        nav_instructions.setAlignment(Qt.AlignCenter)
        nav_instructions.setFont(QFont("Arial", 10, QFont.Bold))
        nav_instructions.setStyleSheet("color: #00ff00;")
        left_panel.addWidget(nav_instructions)
        
        main_layout.addLayout(left_panel, 3)
        
        # Right Panel - Controls
        right_panel = QVBoxLayout()
        
        # Setup Section
        setup_group = QGroupBox("System Setup")
        setup_layout = QVBoxLayout()
        
        self.btn_load = QPushButton("Load Media (Image/Video)")
        self.btn_load.clicked.connect(self.load_media)
        setup_layout.addWidget(self.btn_load)
        
        self.btn_start = QPushButton("Start/Stop Feed")
        self.btn_start.clicked.connect(self.toggle_feed)
        setup_layout.addWidget(self.btn_start)
        setup_group.setLayout(setup_layout)
        right_panel.addWidget(setup_group)
        
        # Illumination System
        illumination_group = QGroupBox("Illumination System")
        illum_layout = QVBoxLayout()
        
        self.light_label = QLabel("Light Intensity: 100%")
        illum_layout.addWidget(self.light_label)
        
        self.light_slider = QSlider(Qt.Horizontal)
        self.light_slider.setMinimum(10)
        self.light_slider.setMaximum(300)
        self.light_slider.setValue(100)
        self.light_slider.valueChanged.connect(self.change_light)
        illum_layout.addWidget(self.light_slider)
        illumination_group.setLayout(illum_layout)
        right_panel.addWidget(illumination_group)
        
        # Image Processing System
        processing_group = QGroupBox("Image Processing System")
        proc_layout = QVBoxLayout()
        
        self.combo_processing = QComboBox()
        self.combo_processing.addItems([
            "None",
            "Noise Reduction (Bilateral)",
            "Contrast Enhancement (CLAHE)",
            "Extract Color Feature (Redness)",
            "Extract Shape Feature (Edges)",
            "Extract Texture Feature (Variance)"
        ])
        self.combo_processing.currentTextChanged.connect(self.change_processing)
        proc_layout.addWidget(self.combo_processing)
        processing_group.setLayout(proc_layout)
        right_panel.addWidget(processing_group)
        
        # Capture System
        capture_group = QGroupBox("Control System")
        cap_layout = QVBoxLayout()
        self.btn_capture = QPushButton("Capture Image")
        self.btn_capture.clicked.connect(self.capture_image)
        cap_layout.addWidget(self.btn_capture)
        capture_group.setLayout(cap_layout)
        right_panel.addWidget(capture_group)
        
        right_panel.addStretch(1)
        main_layout.addLayout(right_panel, 1)
        
        # Set focus to capture key events
        self.setFocusPolicy(Qt.StrongFocus)

    def load_media(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Media File", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if fileName:
            if self.engine.load_media(fileName):
                self.is_running = True
                self.display_label.setText("Media Loaded")
                
    def toggle_feed(self):
        self.is_running = not self.is_running
        
    def change_light(self):
        val = self.light_slider.value()
        self.light_label.setText(f"Light Intensity: {val}%")
        self.engine.set_brightness(val / 100.0)
        
    def change_processing(self, text):
        self.engine.processing_mode = text
        
    def capture_image(self):
        if not self.is_running or self.engine.display_image is None:
            return
        
        # Processed image is currently displayed
        frame = self.process_frame(self.engine.display_image)
        
        filename = "capture.jpg"
        i = 1
        while os.path.exists(filename):
            filename = f"capture_{i}.jpg"
            i += 1
            
        cv2.imwrite(filename, frame)
        print(f"Captured saved as {filename}")
        
    def keyPressEvent(self, event):
        """ Handle Navigation Controls """
        pan_step = 20
        zoom_step = 0.1
        
        if event.key() == Qt.Key_Left:
            self.engine.navigate(pan_step, 0, 0)
        elif event.key() == Qt.Key_Right:
            self.engine.navigate(-pan_step, 0, 0)
        elif event.key() == Qt.Key_Up:
            self.engine.navigate(0, pan_step, 0)
        elif event.key() == Qt.Key_Down:
            self.engine.navigate(0, -pan_step, 0)
        elif event.key() == Qt.Key_W:
            self.engine.navigate(0, 0, zoom_step)
        elif event.key() == Qt.Key_S:
            self.engine.navigate(0, 0, -zoom_step)
            
    def process_frame(self, frame):
        mode = self.engine.processing_mode
        if mode == "Noise Reduction (Bilateral)":
            return processing.apply_noise_reduction(frame)
        elif mode == "Contrast Enhancement (CLAHE)":
            return processing.apply_contrast_enhancement(frame)
        elif mode == "Extract Color Feature (Redness)":
            return processing.extract_color_feature(frame)
        elif mode == "Extract Shape Feature (Edges)":
            return processing.extract_shape_feature(frame)
        elif mode == "Extract Texture Feature (Variance)":
            return processing.extract_texture_feature(frame)
        return frame

    def update_frame(self):
        if not self.is_running:
            return
            
        frame = self.engine.get_current_frame()
        if frame is None:
            return
            
        # Apply selected processing
        processed = self.process_frame(frame)
        
        # Convert to Qt Image for display
        rgb_image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qt_img)
        self.display_label.setPixmap(pixmap.scaled(
            self.display_label.width(), 
            self.display_label.height(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EndoscopeSimulator()
    window.show()
    sys.exit(app.exec_())
