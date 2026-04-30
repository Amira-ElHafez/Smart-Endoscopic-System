"""
Main Window — Intelligent Endoscopic Assistance System
All controls update the display INSTANTLY on change (real-time).
"""

import cv2
import numpy as np
import datetime
import os

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QComboBox,
    QStatusBar, QGridLayout, QCheckBox, QFileDialog, QMessageBox,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeyEvent

from illumination      import IlluminationController
from image_processor   import ImageProcessor
from navigation        import NavigationController
from feature_extractor import FeatureExtractor
from gi_classifier     import GIClassifier, CLASS_SEVERITY, SEVERITY_COLORS, CLASS_DISPLAY


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Endoscopic Assistance System — SBE3220")
        self.setMinimumSize(1300, 800)

        # -- Core modules ----------------------------------------------------
        self.illumination = IlluminationController()
        self.processor    = ImageProcessor()
        self.navigation   = NavigationController()
        self.extractor    = FeatureExtractor()

        # -- State ------------------------------------------------------------
        self.cap            = None      # cv2.VideoCapture
        self.using_camera   = False
        self.is_running     = False
        self.frame_count    = 0

        # raw_frame  = frame straight from source (before any processing)
        # base_frame = raw_frame after illumination + navigation crop
        self.raw_frame  = None
        self.base_frame = None          # reprocessed when illum/nav changes
        self.proc_frame = None          # reprocessed when proc options change

        # -- AI classifier state -----------------------------------------------
        self.classifier     = GIClassifier()
        self._clf_loaded    = False
        self._auto_classify_counter = 0   # throttle: run every N ticks

        self._build_ui()
        self._apply_theme()
        self._connect_signals()
        self.setFocus()

        # Timer drives the source; processing is event-driven
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

        self.status("Ready — load a video/image or connect camera.")

    # ========================================================================
    # UI Construction
    # ========================================================================

    def _build_ui(self):
        root_w  = QWidget();  self.setCentralWidget(root_w)
        root_lyt = QHBoxLayout(root_w)
        root_lyt.setContentsMargins(16, 16, 16, 16)
        root_lyt.setSpacing(16)

        root_lyt.addWidget(self._panel_controls(),  stretch=0)
        root_lyt.addWidget(self._panel_display(),   stretch=1)
        root_lyt.addWidget(self._panel_features(),  stretch=0)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    # -- Left panel ---------------------------------------------------------
    def _panel_controls(self):
        w = QWidget(); w.setFixedWidth(270)
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(12)

        # Source Controls
        g = QGroupBox("Source Controls"); gl = QVBoxLayout(g)
        self.btn_camera     = QPushButton("Connect Camera"); self.btn_camera.setCheckable(True)
        self.btn_load_video = QPushButton("Load Video")
        self.btn_load_image = QPushButton("Load Image")
        self.btn_play_pause = QPushButton("⏸ Pause"); self.btn_play_pause.setEnabled(False)

        for b in [self.btn_camera, self.btn_load_video, self.btn_load_image, self.btn_play_pause]:
            gl.addWidget(b)
        lay.addWidget(g)

        # Illumination Settings
        g = QGroupBox("Illumination Settings"); gl = QVBoxLayout(g)

        row_b = QHBoxLayout()
        row_b.addWidget(QLabel("Brightness"));
        self.lbl_bright = QLabel("100 %"); self.lbl_bright.setAlignment(Qt.AlignRight)
        row_b.addWidget(self.lbl_bright)
        gl.addLayout(row_b)
        self.sl_bright = QSlider(Qt.Horizontal); self.sl_bright.setRange(0, 200); self.sl_bright.setValue(100)
        gl.addWidget(self.sl_bright)

        row_c = QHBoxLayout()
        row_c.addWidget(QLabel("Contrast"));
        self.lbl_contrast = QLabel("100 %"); self.lbl_contrast.setAlignment(Qt.AlignRight)
        row_c.addWidget(self.lbl_contrast)
        gl.addLayout(row_c)
        self.sl_contrast = QSlider(Qt.Horizontal); self.sl_contrast.setRange(0, 200); self.sl_contrast.setValue(100)
        gl.addWidget(self.sl_contrast)
        lay.addWidget(g)

        # Navigation (W A S D / Arrows / + -)
        g = QGroupBox("Navigation (W A S D / + -)"); gl = QGridLayout(g)
        self.btn_up    = self._nav_btn("▲"); self.btn_down  = self._nav_btn("▼")
        self.btn_left  = self._nav_btn("◄"); self.btn_right = self._nav_btn("►")
        self.btn_reset = self._nav_btn("●"); self.btn_reset.setToolTip("Reset to center")

        # Zoom Buttons
        self.btn_zoom_in  = self._nav_btn("➕"); self.btn_zoom_in.setToolTip("Zoom In")
        self.btn_zoom_out = self._nav_btn("➖"); self.btn_zoom_out.setToolTip("Zoom Out")

        gl.addWidget(self.btn_zoom_out, 0, 0, Qt.AlignCenter)
        gl.addWidget(self.btn_up,       0, 1, Qt.AlignCenter)
        gl.addWidget(self.btn_zoom_in,  0, 2, Qt.AlignCenter)
        gl.addWidget(self.btn_left,     1, 0, Qt.AlignCenter)
        gl.addWidget(self.btn_reset,    1, 1, Qt.AlignCenter)
        gl.addWidget(self.btn_right,    1, 2, Qt.AlignCenter)
        gl.addWidget(self.btn_down,     2, 1, Qt.AlignCenter)

        self.lbl_nav = QLabel("Pos: (0, 0)  |  Zoom: 1.0x")
        self.lbl_nav.setAlignment(Qt.AlignCenter)
        self.lbl_nav.setStyleSheet("color: #9ca3af; margin-top: 8px;")
        gl.addWidget(self.lbl_nav, 3, 0, 1, 3)
        lay.addWidget(g)

        # Image Enhancement Processing
        g = QGroupBox("Image Enhancement"); gl = QVBoxLayout(g)
        self.chk_denoise = QCheckBox("Noise Reduction (Gaussian)"); self.chk_denoise.setChecked(True)
        self.chk_clahe   = QCheckBox("CLAHE Contrast Boost"); self.chk_clahe.setChecked(True)
        self.chk_edges   = QCheckBox("Edge Detection (Canny)")
        gl.addWidget(self.chk_denoise); gl.addWidget(self.chk_clahe); gl.addWidget(self.chk_edges)

        gl.addWidget(QLabel("Pseudo-color Map:"))
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["None", "Jet", "Hot", "Cool", "Bone", "HSV"])
        gl.addWidget(self.cmb_cmap)
        lay.addWidget(g)

        lay.addStretch()
        return w

    def _nav_btn(self, text):
        b = QPushButton(text)
        b.setFixedSize(45, 45)
        b.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #1f2937;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #3b82f6;
            }
            QPushButton:pressed {
                background-color: #2563eb;
            }
        """)
        return b

    # -- Centre display -----------------------------------------------------
    def _panel_display(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("LIVE ENDOSCOPIC FEED")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #f3f4f6; letter-spacing: 2px;")

        self.btn_capture = QPushButton("📷 Capture Frame")
        self.btn_capture.setEnabled(False)
        self.btn_capture.setFixedSize(140, 36)
        self.btn_capture.setStyleSheet("background-color: #059669; border:none; border-radius: 6px;")

        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.btn_capture)
        lay.addLayout(header)

        row = QHBoxLayout()
        def video_col(label_text, attr_name):
            col = QVBoxLayout()
            lbl_hdr = QLabel(label_text.upper()); lbl_hdr.setAlignment(Qt.AlignCenter)
            lbl_hdr.setStyleSheet("color: #9ca3af; font-size: 12px; font-weight: bold; letter-spacing: 1px;")

            lbl_vid = QLabel(); lbl_vid.setAlignment(Qt.AlignCenter)
            lbl_vid.setFixedSize(640, 480)

            lbl_vid.setStyleSheet("""
                background-color: #000000;
                border: 2px solid #1f2937;
                border-radius: 12px;
                color: #4b5563;
                font-size: 14px;
            """)
            lbl_vid.setText("NO SIGNAL")

            col.addWidget(lbl_hdr); col.addWidget(lbl_vid)
            setattr(self, attr_name, lbl_vid)
            return col

        row.addLayout(video_col("Original Camera Feed", "lbl_orig"))
        row.addLayout(video_col("Processed Feed", "lbl_proc"))
        lay.addLayout(row)

        info_lyt = QHBoxLayout()
        self.lbl_overlay = QLabel("◉ X: +0  Y: +0  Tilt: 0°  Zoom: 1.0x")
        self.lbl_overlay.setStyleSheet("color: #3b82f6; font-size: 14px; font-weight: bold;")

        self.lbl_info = QLabel("FPS: --  |  Frame: 0  |  Size: --")
        self.lbl_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_info.setStyleSheet("color: #6b7280; font-size: 13px; font-family: monospace;")

        info_lyt.addWidget(self.lbl_overlay)
        info_lyt.addWidget(self.lbl_info)
        lay.addLayout(info_lyt)

        lay.addStretch()
        return w

    # -- Right features panel -----------------------------------------------
    def _panel_features(self):
        w = QWidget(); w.setFixedWidth(250)
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(12)

        self.chk_live_features = QCheckBox("📡 Live Feature Analysis")
        self.chk_live_features.setChecked(True)
        self.chk_live_features.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                font-weight: bold;
                color: #a855f7;
                padding: 8px 0px;
            }
            QCheckBox::indicator {
                width: 18px; height: 18px; border-radius: 4px;
            }
        """)
        lay.addWidget(self.chk_live_features)

        def feat_group(title, labels):
            g = QGroupBox(title); gl = QVBoxLayout(g); gl.setSpacing(6)
            lbls = {}
            for key, text in labels.items():
                lbl = QLabel(text)
                lbl.setStyleSheet("color: #d1d5db; font-family: monospace; font-size: 12px;")
                gl.addWidget(lbl)
                lbls[key] = lbl
            lay.addWidget(g); return lbls

        self.sf = feat_group("Shape Features", {
            "count": "Contours:   --", "area": "Max Area:   --", "circ": "Circular:   --"})
        self.cf = feat_group("Color Features", {
            "mean": "Mean BGR:   --", "hue": "Dom Hue:    --", "sat": "Saturation: --"})
        self.tf = feat_group("Texture (LBP)", {
            "energy": "LBP Energy: --", "entropy": "LBP Entropy:--", "contrast": "Contrast:   --"})

        fg = QGroupBox("Feature Heatmap"); fgl = QVBoxLayout(fg)
        self.lbl_feat_img = QLabel(); self.lbl_feat_img.setFixedSize(220, 165)
        self.lbl_feat_img.setAlignment(Qt.AlignCenter)
        self.lbl_feat_img.setStyleSheet("background-color: #000000; border: 1px solid #374151; border-radius: 6px;")
        fgl.addWidget(self.lbl_feat_img); lay.addWidget(fg)

        # ── AI Classification Panel ──────────────────────────────────────────
        ag = QGroupBox("🤖 AI Classification"); agl = QVBoxLayout(ag)

        self.btn_load_model = QPushButton("Load AI Model")
        self.btn_load_model.setStyleSheet(
            "background-color:#6366f1; border:none; border-radius:6px; padding:8px;")
        agl.addWidget(self.btn_load_model)

        self.btn_classify = QPushButton("🔬 Classify Frame")
        self.btn_classify.setEnabled(False)
        self.btn_classify.setStyleSheet(
            "background-color:#8b5cf6; border:none; border-radius:6px; padding:8px;")
        agl.addWidget(self.btn_classify)

        self.chk_auto_classify = QCheckBox("⚡ Auto-Classify (live)")
        self.chk_auto_classify.setEnabled(False)
        self.chk_auto_classify.setStyleSheet("""
            QCheckBox { font-size:13px; font-weight:bold; color:#a78bfa; padding:4px 0; }
            QCheckBox::indicator { width:16px; height:16px; border-radius:4px; }
        """)
        agl.addWidget(self.chk_auto_classify)

        # Result: class name
        self.lbl_clf_class = QLabel("Class: —")
        self.lbl_clf_class.setWordWrap(True)
        self.lbl_clf_class.setStyleSheet(
            "color:#f3f4f6; font-size:14px; font-weight:bold; padding:6px 0 2px 0;")
        agl.addWidget(self.lbl_clf_class)

        # Result: severity tag
        self.lbl_clf_severity = QLabel("")
        self.lbl_clf_severity.setAlignment(Qt.AlignCenter)
        self.lbl_clf_severity.setFixedHeight(22)
        self.lbl_clf_severity.setStyleSheet(
            "border-radius:4px; font-size:11px; font-weight:bold; color:#fff;")
        agl.addWidget(self.lbl_clf_severity)

        # Confidence bar (using a styled QLabel as simple bar)
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence:"))
        self.lbl_clf_conf = QLabel("— %")
        self.lbl_clf_conf.setAlignment(Qt.AlignRight)
        self.lbl_clf_conf.setStyleSheet("color:#60a5fa; font-weight:bold;")
        conf_row.addWidget(self.lbl_clf_conf)
        agl.addLayout(conf_row)

        # Visual confidence bar
        self.lbl_conf_bar = QLabel()
        self.lbl_conf_bar.setFixedHeight(8)
        self.lbl_conf_bar.setStyleSheet(
            "background-color:#1f2937; border-radius:4px;")
        agl.addWidget(self.lbl_conf_bar)

        # Status label
        self.lbl_clf_status = QLabel("Model not loaded")
        self.lbl_clf_status.setStyleSheet("color:#6b7280; font-size:11px;")
        agl.addWidget(self.lbl_clf_status)

        lay.addWidget(ag)

        lay.addStretch()
        return w

    # ========================================================================
    # Signal connections
    # ========================================================================

    def _connect_signals(self):
        self.btn_camera.clicked.connect(self._toggle_camera)
        self.btn_load_video.clicked.connect(self._load_video)
        self.btn_load_image.clicked.connect(self._load_image)

        # Play/Pause control
        self.btn_play_pause.clicked.connect(self._toggle_pause)

        self.btn_capture.clicked.connect(self._capture)
        self.sl_bright.valueChanged.connect(self._on_brightness)
        self.sl_contrast.valueChanged.connect(self._on_contrast)

        self.btn_up.clicked.connect(lambda: self._navigate("up"))
        self.btn_down.clicked.connect(lambda: self._navigate("down"))
        self.btn_left.clicked.connect(lambda: self._navigate("left"))
        self.btn_right.clicked.connect(lambda: self._navigate("right"))
        self.btn_reset.clicked.connect(lambda: self._navigate("reset"))
        self.btn_zoom_in.clicked.connect(lambda: self._navigate("zoom_in"))
        self.btn_zoom_out.clicked.connect(lambda: self._navigate("zoom_out"))

        self.chk_denoise.stateChanged.connect(self._reprocess)
        self.chk_clahe.stateChanged.connect(self._reprocess)
        self.chk_edges.stateChanged.connect(self._reprocess)
        self.cmb_cmap.currentIndexChanged.connect(self._reprocess)

        self.chk_live_features.stateChanged.connect(self._reprocess)

        # AI classification
        self.btn_load_model.clicked.connect(self._load_ai_model)
        self.btn_classify.clicked.connect(self._classify_frame)

    # ========================================================================
    # Keyboard
    # ========================================================================

    def keyPressEvent(self, e: QKeyEvent):
        k = e.key()
        if   k in (Qt.Key_W, Qt.Key_Up):    self._navigate("up")
        elif k in (Qt.Key_S, Qt.Key_Down):  self._navigate("down")
        elif k in (Qt.Key_A, Qt.Key_Left):  self._navigate("left")
        elif k in (Qt.Key_D, Qt.Key_Right): self._navigate("right")
        elif k in (Qt.Key_Plus, Qt.Key_Equal): self._navigate("zoom_in")
        elif k == Qt.Key_Minus:             self._navigate("zoom_out")
        elif k == Qt.Key_R:                 self._navigate("reset")
        elif k == Qt.Key_Space:             self._capture()
        else: super().keyPressEvent(e)

    # ========================================================================
    # Handlers
    # ========================================================================

    def _on_brightness(self, v):
        self.lbl_bright.setText(f"{v} %")
        self.illumination.set_brightness(v / 100.0)
        self._rebuild_base()

    def _on_contrast(self, v):
        self.lbl_contrast.setText(f"{v} %")
        self.illumination.set_contrast(v / 100.0)
        self._rebuild_base()

    def _navigate(self, direction: str):
        self.navigation.move(direction)
        x, y, tilt, zoom = self.navigation.get_state()
        self.lbl_nav.setText(f"Pos: ({x:+d}, {y:+d})  |  Zoom: {zoom:.1f}x")
        self.lbl_overlay.setText(f"◉ X: {x:+d}   Y: {y:+d}   Tilt: {tilt}°   Zoom: {zoom:.1f}x")
        self.status(f"Navigation → {direction}  |  ({x:+d}, {y:+d}) | {zoom:.1f}x")
        self._rebuild_base()

    def _rebuild_base(self):
        if self.raw_frame is None: return
        f = self.illumination.apply(self.raw_frame)
        f = self.navigation.apply_to_frame(f)
        self.base_frame = f
        self._show(self.lbl_orig, f)
        self._reprocess()

    def _reprocess(self, *_):
        if self.base_frame is None: return
        f = self.base_frame.copy()
        if self.chk_denoise.isChecked(): f = self.processor.denoise(f)
        if self.chk_clahe.isChecked():   f = self.processor.enhance_clahe(f)
        if self.chk_edges.isChecked():   f = self.processor.detect_edges(f)
        cmap = self.cmb_cmap.currentText()
        if cmap != "None":               f = self.processor.apply_colormap(f, cmap)

        self.proc_frame = f
        self._show(self.lbl_proc, f)

        if hasattr(self, 'chk_live_features') and self.chk_live_features.isChecked():
            self._extract_features()

    def _tick(self):
        frame = None
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if not self.using_camera:
                    # Looping video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                if not ret:
                    self._stop(); return
        else:
            frame = self._test_pattern()

        self.frame_count += 1
        self.raw_frame = frame
        self._rebuild_base()

        h, w = frame.shape[:2]
        fps  = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30
        self.lbl_info.setText(f"FPS: {fps:.0f}  |  Frame: {self.frame_count}  |  {w}×{h}")

        # Auto-classify throttle (~1 per second at 30 fps)
        if (self._clf_loaded
                and hasattr(self, 'chk_auto_classify')
                and self.chk_auto_classify.isChecked()):
            self._auto_classify_counter += 1
            if self._auto_classify_counter >= 30:
                self._auto_classify_counter = 0
                self._classify_frame()

    # ========================================================================
    # Source management
    # ========================================================================

    def _toggle_camera(self, checked):
        if checked:
            # Stop any existing source before opening camera
            self._stop()
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Camera Error", "No camera found — using test pattern.")
                self.cap = None; self.btn_camera.setChecked(False)
            else:
                self.using_camera = True
            self._start()
        else:
            self._stop()

    def _load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video (*.mp4 *.avi *.mov *.mkv *.wmv)")
        if path:
            self._stop()
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened(): QMessageBox.warning(self, "Error", "Cannot open video."); return
            self.using_camera = False
            self._start()

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image (*.png *.jpg *.jpeg *.bmp *.tif)")
        if path:
            frame = cv2.imread(path)
            if frame is None: QMessageBox.warning(self, "Error", "Cannot open image."); return

            # Stop camera or video if running
            self._stop()
            self.raw_frame = frame
            self._rebuild_base()

            self.btn_capture.setEnabled(True)

            h, w = frame.shape[:2]
            self.lbl_info.setText(f"FPS: 0 (Image)  |  Frame: 1  |  {w}×{h}")
            self.status(f"Image loaded: {os.path.basename(path)}")

    def _start(self):
        self.is_running = True
        self.timer.start(33)
        self.btn_play_pause.setEnabled(True)
        self.btn_play_pause.setText("⏸ Pause")
        self.btn_capture.setEnabled(True)
        self.btn_camera.setText("Disconnect Camera" if self.using_camera else "Connect Camera")

    def _stop(self):
        """Internal cleanup when switching sources or stopping playback."""
        self.is_running = False
        self.timer.stop()
        if self.cap: self.cap.release(); self.cap = None
        self.using_camera = False
        self.btn_camera.setChecked(False); self.btn_camera.setText("Connect Camera")
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.setText("⏸ Pause")

    def _toggle_pause(self):
        """Toggle Play/Pause state"""
        if self.is_running:
            # Pause playback
            self.is_running = False
            self.timer.stop()
            self.btn_play_pause.setText("▶ Resume")
            self.status("Source Paused.")
        else:
            # Resume playback
            self.is_running = True
            self.timer.start(33)
            self.btn_play_pause.setText("⏸ Pause")
            self.status("Source Resumed.")

    # ========================================================================
    # Capture & Extraction
    # ========================================================================

    def _capture(self):
        if self.raw_frame is None: return
        os.makedirs("output", exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"output/capture_{ts}.png"
        cv2.imwrite(path, self.raw_frame)
        self.status(f"Screenshot saved → {path}")
        self.btn_capture.setText("✅ Saved!")
        QTimer.singleShot(1500, lambda: self.btn_capture.setText("📷 Capture Frame"))

    def _extract_features(self):
        src = self.proc_frame if self.proc_frame is not None else self.base_frame
        if src is None: return

        try:
            f = self.extractor.extract_all(src)
            self.sf["count"].setText(f"Contours:   {f['contour_count']}")
            self.sf["area"].setText(f"Max Area:   {f['largest_area']:.0f} px²")
            self.sf["circ"].setText(f"Circular:   {f['circularity']:.3f}")

            b, g, r = f["mean_color"]
            self.cf["mean"].setText(f"Mean BGR:   ({b:.0f},{g:.0f},{r:.0f})")
            self.cf["hue"].setText(f"Dom Hue:    {f['dominant_hue']:.1f}°")
            self.cf["sat"].setText(f"Saturation: {f['saturation']:.1f}")

            self.tf["energy"].setText(f"LBP Energy: {f['lbp_energy']:.4f}")
            self.tf["entropy"].setText(f"LBP Entropy:{f['lbp_entropy']:.4f}")
            self.tf["contrast"].setText(f"Contrast:   {f['texture_contrast']:.2f}")

            # Update Heatmap
            fm = self.extractor.get_feature_map(src)
            self.lbl_feat_img.setPixmap(
                self._to_pixmap(fm).scaled(
                    self.lbl_feat_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        except Exception as e:
            print(f"Feature Extraction Error: {e}")

    # ========================================================================
    # AI Classification
    # ========================================================================

    def _load_ai_model(self):
        """Load the Keras model (may take a few seconds on first run)."""
        self.lbl_clf_status.setText("Loading model…")
        self.lbl_clf_status.setStyleSheet("color:#f59e0b; font-size:11px;")
        QApplication.processEvents()          # repaint before blocking load

        ok = self.classifier.load()
        if ok:
            self._clf_loaded = True
            self.btn_classify.setEnabled(True)
            self.chk_auto_classify.setEnabled(True)
            self.btn_load_model.setText("✅ Model Loaded")
            self.btn_load_model.setEnabled(False)
            self.lbl_clf_status.setText(
                f"Ready  •  input {self.classifier.input_size[0]}×{self.classifier.input_size[1]}")
            self.lbl_clf_status.setStyleSheet("color:#10b981; font-size:11px;")
            self.status("AI model loaded successfully.")
        else:
            QMessageBox.warning(self, "Model Error", self.classifier.error or "Unknown error")
            self.lbl_clf_status.setText("Load failed")
            self.lbl_clf_status.setStyleSheet("color:#ef4444; font-size:11px;")

    def _classify_frame(self):
        """Run inference on the current processed frame."""
        src = self.proc_frame if self.proc_frame is not None else self.base_frame
        if src is None or not self._clf_loaded:
            return
        try:
            cls_name, conf, _ = self.classifier.classify(src)
            self._update_classification_ui(cls_name, conf)
        except Exception as e:
            self.lbl_clf_class.setText(f"Error: {e}")
            print(f"Classification error: {e}")

    def _update_classification_ui(self, cls_name: str, confidence: float):
        """Refresh the classification result widgets."""
        display = CLASS_DISPLAY.get(cls_name, cls_name)
        severity = CLASS_SEVERITY.get(cls_name, "normal")
        color = SEVERITY_COLORS.get(severity, "#6b7280")
        pct = confidence * 100

        self.lbl_clf_class.setText(display)
        self.lbl_clf_class.setStyleSheet(
            f"color:{color}; font-size:14px; font-weight:bold; padding:6px 0 2px 0;")

        sev_label = severity.upper()
        self.lbl_clf_severity.setText(f"  {sev_label}  ")
        self.lbl_clf_severity.setStyleSheet(
            f"background-color:{color}; border-radius:4px; "
            f"font-size:11px; font-weight:bold; color:#fff; padding:2px 8px;")

        self.lbl_clf_conf.setText(f"{pct:.1f} %")

        bar_width = max(1, int(self.lbl_conf_bar.parent().width() * confidence * 0.9))
        self.lbl_conf_bar.setFixedWidth(bar_width)
        self.lbl_conf_bar.setStyleSheet(
            f"background-color:{color}; border-radius:4px;")

    def _show(self, label: QLabel, frame: np.ndarray):
        label.setPixmap(
            self._to_pixmap(frame).scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @staticmethod
    def _to_pixmap(frame: np.ndarray) -> QPixmap:
        if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))

    def _test_pattern(self) -> np.ndarray:
        t = self.frame_count * 0.05
        h, w = 480, 640
        img  = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        mask = (np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) < 220).astype(np.float32)
        for i in range(5):
            px = int(cx + 80 * np.cos(t + i * 1.26))
            py = int(cy + 60 * np.sin(t * 0.7 + i * 1.26))
            cv2.circle(img, (px, py), 35 + i * 8,
                       (int(60 + 40 * np.sin(t + i)),
                        int(30 + 20 * np.cos(t * 0.5)),
                        int(80 + 30 * np.sin(t * 0.3 + i))), -1)
        noise = np.random.randint(0, 10, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        return (img * mask[:, :, np.newaxis]).astype(np.uint8)

    def status(self, msg: str):
        self.status_bar.showMessage(msg)

    # ========================================================================
    # Professional Dark Theme (QSS)
    # ========================================================================

    def _apply_theme(self):
        style = """
        QMainWindow, QWidget {
            background-color: #0b0f19;
            color: #e5e7eb;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            font-size: 13px;
        }
        
        QGroupBox {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 8px;
            margin-top: 16px;
            padding: 16px 12px 10px 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 2px 8px;
            color: #60a5fa;
            font-weight: bold;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            background-color: #1f2937;
            border-radius: 4px;
        }

        QPushButton {
            background-color: #1f2937;
            border: 1px solid #374151;
            border-radius: 6px;
            padding: 8px 14px;
            color: #f3f4f6;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #3b82f6;
            border-color: #3b82f6;
            color: #ffffff;
        }
        QPushButton:pressed {
            background-color: #2563eb;
            border-color: #2563eb;
        }
        QPushButton:checked {
            background-color: #10b981;
            border-color: #10b981;
        }
        QPushButton:disabled {
            background-color: #111827;
            color: #4b5563;
            border-color: #1f2937;
        }

        QSlider::groove:horizontal {
            height: 6px;
            background: #1f2937;
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: #3b82f6;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #ffffff;
            border: 2px solid #3b82f6;
            width: 16px;
            height: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background: #60a5fa;
            border-color: #ffffff;
            transform: scale(1.1);
        }

        QComboBox {
            background-color: #1f2937;
            border: 1px solid #374151;
            border-radius: 6px;
            padding: 6px 10px;
            color: #e5e7eb;
        }
        QComboBox:hover {
            border-color: #3b82f6;
        }
        QComboBox::drop-down {
            border: none;
            padding-right: 10px;
        }
        QComboBox QAbstractItemView {
            background-color: #1f2937;
            selection-background-color: #3b82f6;
            color: #ffffff;
            border: 1px solid #374151;
            border-radius: 4px;
        }

        QCheckBox {
            spacing: 10px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            background-color: #1f2937;
            border: 2px solid #374151;
            border-radius: 4px;
        }
        QCheckBox::indicator:hover {
            border-color: #3b82f6;
        }
        QCheckBox::indicator:checked {
            background-color: #3b82f6;
            border-color: #3b82f6;
        }

        QLabel {
            color: #d1d5db;
        }
        
        QStatusBar {
            background-color: #111827;
            color: #9ca3af;
            border-top: 1px solid #1f2937;
            padding-left: 8px;
        }
        """
        self.setStyleSheet(style)