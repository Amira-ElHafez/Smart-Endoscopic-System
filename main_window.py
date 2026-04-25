"""
Main Window — Intelligent Endoscopic Assistance System
All controls update the display INSTANTLY on change (real-time).
"""

import cv2
import numpy as np
import datetime
import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QComboBox,
    QStatusBar, QGridLayout, QCheckBox, QFileDialog, QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeyEvent

from illumination      import IlluminationController
from image_processor   import ImageProcessor
from navigation        import NavigationController
from feature_extractor import FeatureExtractor


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Endoscopic Assistance System — SBE3220")
        self.setMinimumSize(1300, 800)

        # ── Core modules ────────────────────────────────────────────────────
        self.illumination = IlluminationController()
        self.processor    = ImageProcessor()
        self.navigation   = NavigationController()
        self.extractor    = FeatureExtractor()

        # ── State ────────────────────────────────────────────────────────────
        self.cap            = None      # cv2.VideoCapture
        self.using_camera   = False
        self.is_running     = False
        self.frame_count    = 0

        # raw_frame  = frame straight from source (before any processing)
        # base_frame = raw_frame after illumination + navigation crop
        self.raw_frame  = None
        self.base_frame = None          # reprocessed when illum/nav changes
        self.proc_frame = None          # reprocessed when proc options change

        self._build_ui()
        self._apply_theme()
        self._connect_signals()
        self.setFocus()

        # Timer drives the source; processing is event-driven
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

        self.status("Ready — load a video/image or connect camera.")

    # ════════════════════════════════════════════════════════════════════════
    # UI Construction
    # ════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        root_w  = QWidget();  self.setCentralWidget(root_w)
        root_lyt = QHBoxLayout(root_w)
        root_lyt.setContentsMargins(8, 8, 8, 8)
        root_lyt.setSpacing(8)

        root_lyt.addWidget(self._panel_controls(),  stretch=0)
        root_lyt.addWidget(self._panel_display(),   stretch=1)
        root_lyt.addWidget(self._panel_features(),  stretch=0)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    # ── Left panel ─────────────────────────────────────────────────────────
    def _panel_controls(self):
        w = QWidget(); w.setFixedWidth(245)
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(10)

        # Source
        g = QGroupBox("Source"); gl = QVBoxLayout(g)
        self.btn_camera     = QPushButton("Connect Camera"); self.btn_camera.setCheckable(True)
        self.btn_load_video = QPushButton("Load Video")
        self.btn_load_image = QPushButton("Load Image")
        self.btn_stop       = QPushButton("Stop"); self.btn_stop.setEnabled(False)
        for b in [self.btn_camera, self.btn_load_video, self.btn_load_image, self.btn_stop]:
            gl.addWidget(b)
        lay.addWidget(g)

        # Illumination
        g = QGroupBox("Illumination"); gl = QVBoxLayout(g)
        gl.addWidget(QLabel("Brightness (LED intensity)"))
        self.sl_bright = QSlider(Qt.Horizontal); self.sl_bright.setRange(0, 200); self.sl_bright.setValue(100)
        self.lbl_bright = QLabel("100 %")
        gl.addWidget(self.sl_bright); gl.addWidget(self.lbl_bright)
        gl.addWidget(QLabel("Contrast"))
        self.sl_contrast = QSlider(Qt.Horizontal); self.sl_contrast.setRange(0, 200); self.sl_contrast.setValue(100)
        self.lbl_contrast = QLabel("100 %")
        gl.addWidget(self.sl_contrast); gl.addWidget(self.lbl_contrast)
        lay.addWidget(g)

        # Navigation
        g = QGroupBox("Navigation  (W A S D / Arrows)"); gl = QGridLayout(g)
        self.btn_up    = self._nav_btn("▲"); self.btn_down  = self._nav_btn("▼")
        self.btn_left  = self._nav_btn("◄"); self.btn_right = self._nav_btn("►")
        self.btn_reset = self._nav_btn("●"); self.btn_reset.setToolTip("Reset to center")
        gl.addWidget(self.btn_up,    0, 1)
        gl.addWidget(self.btn_left,  1, 0)
        gl.addWidget(self.btn_reset, 1, 1)
        gl.addWidget(self.btn_right, 1, 2)
        gl.addWidget(self.btn_down,  2, 1)
        self.lbl_nav = QLabel("Position: (0, 0)"); self.lbl_nav.setAlignment(Qt.AlignCenter)
        gl.addWidget(self.lbl_nav, 3, 0, 1, 3)
        lay.addWidget(g)

        # Processing
        g = QGroupBox("Processing"); gl = QVBoxLayout(g)
        self.chk_denoise = QCheckBox("Noise Reduction  (Gaussian)"); self.chk_denoise.setChecked(True)
        self.chk_clahe   = QCheckBox("CLAHE  Contrast Enhancement"); self.chk_clahe.setChecked(True)
        self.chk_edges   = QCheckBox("Edge Detection  (Canny)")
        gl.addWidget(self.chk_denoise); gl.addWidget(self.chk_clahe); gl.addWidget(self.chk_edges)
        gl.addWidget(QLabel("Pseudo-color map:"))
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["None", "Jet", "Hot", "Cool", "Bone", "HSV"])
        gl.addWidget(self.cmb_cmap)
        lay.addWidget(g)

        # Capture
        g = QGroupBox("Capture"); gl = QVBoxLayout(g)
        self.btn_capture = QPushButton("📷  Capture Frame  [Space]"); self.btn_capture.setEnabled(False)
        gl.addWidget(self.btn_capture)
        lay.addWidget(g)

        lay.addStretch()
        return w

    def _nav_btn(self, text):
        b = QPushButton(text); b.setFixedSize(44, 44); return b

    # ── Centre display ─────────────────────────────────────────────────────
    def _panel_display(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)

        title = QLabel("Real-time Endoscopic Feed")
        title.setFont(QFont("Arial", 11, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        row = QHBoxLayout()
        def video_col(label_text, attr_name):
            col = QVBoxLayout()
            lbl_hdr = QLabel(label_text); lbl_hdr.setAlignment(Qt.AlignCenter)
            lbl_hdr.setStyleSheet("color:#aaa;font-size:11px;")
            lbl_vid = QLabel(); lbl_vid.setAlignment(Qt.AlignCenter)
            lbl_vid.setMinimumSize(480, 360)
            lbl_vid.setStyleSheet("background:#111;border-radius:6px;")
            lbl_vid.setText("No source")
            col.addWidget(lbl_hdr); col.addWidget(lbl_vid)
            setattr(self, attr_name, lbl_vid)
            return col

        row.addLayout(video_col("Original  (illumination + navigation)", "lbl_orig"))
        row.addLayout(video_col("Processed", "lbl_proc"))
        lay.addLayout(row)

        self.lbl_overlay = QLabel("◉  Center"); self.lbl_overlay.setAlignment(Qt.AlignCenter)
        self.lbl_overlay.setStyleSheet("color:#0af;font-size:12px;")
        lay.addWidget(self.lbl_overlay)

        self.lbl_info = QLabel("FPS: — | Frame: 0 | Size: —")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("color:#666;font-size:11px;")
        lay.addWidget(self.lbl_info)
        return w

    # ── Right features panel ───────────────────────────────────────────────
    def _panel_features(self):
        w = QWidget(); w.setFixedWidth(220)
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(10)

        self.btn_extract = QPushButton("Extract Features"); self.btn_extract.setEnabled(False)
        lay.addWidget(self.btn_extract)

        def feat_group(title, labels):
            g = QGroupBox(title); gl = QVBoxLayout(g)
            lbls = {}
            for key, text in labels.items():
                lbl = QLabel(text); gl.addWidget(lbl); lbls[key] = lbl
            lay.addWidget(g); return lbls

        self.sf = feat_group("Shape Features", {
            "count": "Contours: —", "area": "Largest area: —", "circ": "Circularity: —"})
        self.cf = feat_group("Color Features", {
            "mean": "Mean BGR: —", "hue": "Dominant hue: —", "sat": "Saturation: —"})
        self.tf = feat_group("Texture (LBP)", {
            "energy": "LBP energy: —", "entropy": "LBP entropy: —", "contrast": "Contrast: —"})

        fg = QGroupBox("Feature Map"); fgl = QVBoxLayout(fg)
        self.lbl_feat_img = QLabel(); self.lbl_feat_img.setFixedSize(200, 150)
        self.lbl_feat_img.setAlignment(Qt.AlignCenter)
        self.lbl_feat_img.setStyleSheet("background:#111;border-radius:4px;")
        fgl.addWidget(self.lbl_feat_img); lay.addWidget(fg)

        lay.addStretch()
        return w

    # ════════════════════════════════════════════════════════════════════════
    # Signal connections  — every control calls _reprocess immediately
    # ════════════════════════════════════════════════════════════════════════

    def _connect_signals(self):
        # Source
        self.btn_camera.clicked.connect(self._toggle_camera)
        self.btn_load_video.clicked.connect(self._load_video)
        self.btn_load_image.clicked.connect(self._load_image)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_capture.clicked.connect(self._capture)

        # Illumination — update illumination controller then reprocess base
        self.sl_bright.valueChanged.connect(self._on_brightness)
        self.sl_contrast.valueChanged.connect(self._on_contrast)

        # Navigation buttons
        self.btn_up.clicked.connect(lambda: self._navigate("up"))
        self.btn_down.clicked.connect(lambda: self._navigate("down"))
        self.btn_left.clicked.connect(lambda: self._navigate("left"))
        self.btn_right.clicked.connect(lambda: self._navigate("right"))
        self.btn_reset.clicked.connect(lambda: self._navigate("reset"))

        # Processing checkboxes and colormap — reprocess immediately
        self.chk_denoise.stateChanged.connect(self._reprocess)
        self.chk_clahe.stateChanged.connect(self._reprocess)
        self.chk_edges.stateChanged.connect(self._reprocess)
        self.cmb_cmap.currentIndexChanged.connect(self._reprocess)

        # Feature extraction
        self.btn_extract.clicked.connect(self._extract_features)

    # ════════════════════════════════════════════════════════════════════════
    # Keyboard
    # ════════════════════════════════════════════════════════════════════════

    def keyPressEvent(self, e: QKeyEvent):
        k = e.key()
        if   k in (Qt.Key_W, Qt.Key_Up):    self._navigate("up")
        elif k in (Qt.Key_S, Qt.Key_Down):  self._navigate("down")
        elif k in (Qt.Key_A, Qt.Key_Left):  self._navigate("left")
        elif k in (Qt.Key_D, Qt.Key_Right): self._navigate("right")
        elif k == Qt.Key_R:                 self._navigate("reset")
        elif k == Qt.Key_Space:             self._capture()
        else: super().keyPressEvent(e)

    # ════════════════════════════════════════════════════════════════════════
    # Illumination handlers  →  rebuild base_frame then reprocess
    # ════════════════════════════════════════════════════════════════════════

    def _on_brightness(self, v):
        self.lbl_bright.setText(f"{v} %")
        self.illumination.set_brightness(v / 100.0)
        self._rebuild_base()      # re-apply illumination + nav to raw frame

    def _on_contrast(self, v):
        self.lbl_contrast.setText(f"{v} %")
        self.illumination.set_contrast(v / 100.0)
        self._rebuild_base()

    # ════════════════════════════════════════════════════════════════════════
    # Navigation handler  →  rebuild base_frame then reprocess
    # ════════════════════════════════════════════════════════════════════════

    def _navigate(self, direction: str):
        self.navigation.move(direction)
        x, y, tilt = self.navigation.get_state()
        self.lbl_nav.setText(f"Position: ({x:+d}, {y:+d})")
        self.lbl_overlay.setText(f"◉  X:{x:+d}  Y:{y:+d}  Tilt:{tilt}°")
        self.status(f"Navigation → {direction}  |  ({x:+d}, {y:+d})")
        self._rebuild_base()

    # ════════════════════════════════════════════════════════════════════════
    # Frame pipeline
    # ════════════════════════════════════════════════════════════════════════

    def _rebuild_base(self):
        """Apply illumination + navigation crop to the latest raw frame → base_frame.
        Then immediately reprocess and update both displays."""
        if self.raw_frame is None:
            return
        f = self.illumination.apply(self.raw_frame)
        f = self.navigation.apply_to_frame(f)
        self.base_frame = f
        self._show(self.lbl_orig, f)
        self._reprocess()

    def _reprocess(self, *_):
        """Apply processing pipeline to base_frame → proc_frame and update right display."""
        if self.base_frame is None:
            return
        f = self.base_frame.copy()
        if self.chk_denoise.isChecked():
            f = self.processor.denoise(f)
        if self.chk_clahe.isChecked():
            f = self.processor.enhance_clahe(f)
        if self.chk_edges.isChecked():
            f = self.processor.detect_edges(f)
        cmap = self.cmb_cmap.currentText()
        if cmap != "None":
            f = self.processor.apply_colormap(f, cmap)
        self.proc_frame = f
        self._show(self.lbl_proc, f)

    # ════════════════════════════════════════════════════════════════════════
    # Timer tick — only reads new frames from source
    # ════════════════════════════════════════════════════════════════════════

    def _tick(self):
        frame = None
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if not self.using_camera:
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
        self.lbl_info.setText(f"FPS: {fps:.0f} | Frame: {self.frame_count} | {w}×{h}")

    # ════════════════════════════════════════════════════════════════════════
    # Source management
    # ════════════════════════════════════════════════════════════════════════

    def _toggle_camera(self, checked):
        if checked:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Camera", "No camera found — using test pattern.")
                self.cap = None; self.btn_camera.setChecked(False)
            else:
                self.using_camera = True
            self._start()
        else:
            self._stop()

    def _load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video (*.mp4 *.avi *.mov *.mkv *.wmv)")
        if path:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Error", "Cannot open video."); return
            self.using_camera = False
            self._start()

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image (*.png *.jpg *.jpeg *.bmp *.tif)")
        if path:
            frame = cv2.imread(path)
            if frame is None:
                QMessageBox.warning(self, "Error", "Cannot open image."); return
            self._stop()
            self.raw_frame = frame
            self._rebuild_base()
            self.btn_capture.setEnabled(True)
            self.btn_extract.setEnabled(True)
            self.status(f"Image: {path}")

    def _start(self):
        self.is_running = True
        self.timer.start(33)
        self.btn_stop.setEnabled(True)
        self.btn_capture.setEnabled(True)
        self.btn_extract.setEnabled(True)

    def _stop(self):
        self.is_running = False; self.timer.stop()
        if self.cap: self.cap.release(); self.cap = None
        self.using_camera = False
        self.btn_camera.setChecked(False); self.btn_camera.setText("Connect Camera")
        self.btn_stop.setEnabled(False)
        self.status("Stopped.")

    # ════════════════════════════════════════════════════════════════════════
    # Capture
    # ════════════════════════════════════════════════════════════════════════

    def _capture(self):
        if self.raw_frame is None: return
        os.makedirs("output", exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"output/capture_{ts}.png"
        cv2.imwrite(path, self.raw_frame)
        self.status(f"Saved → {path}")

    # ════════════════════════════════════════════════════════════════════════
    # Feature extraction
    # ════════════════════════════════════════════════════════════════════════

    def _extract_features(self):
        src = self.proc_frame if self.proc_frame is not None else self.base_frame
        if src is None: return
        f = self.extractor.extract_all(src)
        self.sf["count"].setText(f"Contours: {f['contour_count']}")
        self.sf["area"].setText(f"Largest area: {f['largest_area']:.0f} px²")
        self.sf["circ"].setText(f"Circularity: {f['circularity']:.3f}")
        b, g, r = f["mean_color"]
        self.cf["mean"].setText(f"Mean BGR: ({b:.0f},{g:.0f},{r:.0f})")
        self.cf["hue"].setText(f"Dominant hue: {f['dominant_hue']:.1f}°")
        self.cf["sat"].setText(f"Saturation: {f['saturation']:.1f}")
        self.tf["energy"].setText(f"LBP energy: {f['lbp_energy']:.4f}")
        self.tf["entropy"].setText(f"LBP entropy: {f['lbp_entropy']:.4f}")
        self.tf["contrast"].setText(f"Contrast: {f['texture_contrast']:.2f}")
        fm = self.extractor.get_feature_map(src)
        self.lbl_feat_img.setPixmap(
            self._to_pixmap(fm).scaled(
                self.lbl_feat_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.status("Features extracted.")

    # ════════════════════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════════════════════

    def _show(self, label: QLabel, frame: np.ndarray):
        label.setPixmap(
            self._to_pixmap(frame).scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @staticmethod
    def _to_pixmap(frame: np.ndarray) -> QPixmap:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
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

    # ════════════════════════════════════════════════════════════════════════
    # Dark theme
    # ════════════════════════════════════════════════════════════════════════

    def _apply_theme(self):
        self.setStyleSheet("""
        QMainWindow,QWidget{background:#1a1a1a;color:#e0e0e0;
            font-family:'Segoe UI',Arial,sans-serif;font-size:12px;}
        QGroupBox{border:1px solid #3a3a3a;border-radius:6px;margin-top:8px;
            padding:6px;font-weight:bold;color:#aaa;}
        QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;}
        QPushButton{background:#2d2d2d;border:1px solid #444;border-radius:5px;
            padding:6px 10px;color:#e0e0e0;}
        QPushButton:hover{background:#3a3a3a;}
        QPushButton:pressed{background:#1e6fa0;}
        QPushButton:checked{background:#1e6fa0;border-color:#2a90d0;}
        QPushButton:disabled{color:#555;}
        QSlider::groove:horizontal{height:4px;background:#333;border-radius:2px;}
        QSlider::handle:horizontal{background:#2a90d0;width:14px;height:14px;
            margin:-5px 0;border-radius:7px;}
        QSlider::sub-page:horizontal{background:#2a90d0;border-radius:2px;}
        QComboBox{background:#2d2d2d;border:1px solid #444;border-radius:4px;padding:4px;}
        QComboBox::drop-down{border:none;}
        QCheckBox::indicator{width:14px;height:14px;border:1px solid #555;
            border-radius:3px;background:#2d2d2d;}
        QCheckBox::indicator:checked{background:#2a90d0;border-color:#2a90d0;}
        QLabel{color:#ccc;}
        QStatusBar{background:#111;color:#888;border-top:1px solid #333;}
        """)