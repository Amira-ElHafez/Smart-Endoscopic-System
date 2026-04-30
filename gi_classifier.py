"""
GI Classifier — Kvasir GI-tract disease classification using a pre-trained Keras model.
Wraps the .keras model for single-frame inference in the endoscopic GUI.
"""

import os
import cv2
import numpy as np

# ---------- lazy TensorFlow import -------------------------------------------
_tf = None
_load_model = None


def _ensure_tf():
    """Import TensorFlow on first use so the app still starts without it."""
    global _tf, _load_model
    if _tf is None:
        import tensorflow as tf
        _tf = tf
        _load_model = tf.keras.models.load_model


# ---------- Kvasir v2 class labels (alphabetical folder order) ---------------
KVASIR_CLASSES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis",
]

# Severity colouring used in the GUI
#   green  = anatomical landmark (normal finding)
#   yellow = endoscopic procedure artifact
#   red    = pathological finding
CLASS_SEVERITY = {
    "normal-cecum":            "normal",
    "normal-pylorus":          "normal",
    "normal-z-line":           "normal",
    "dyed-lifted-polyps":      "procedure",
    "dyed-resection-margins":  "procedure",
    "esophagitis":             "pathological",
    "polyps":                  "pathological",
    "ulcerative-colitis":      "pathological",
}

SEVERITY_COLORS = {
    "normal":       "#10b981",   # green
    "procedure":    "#f59e0b",   # amber
    "pathological": "#ef4444",   # red
}

# Pretty display names
CLASS_DISPLAY = {
    "dyed-lifted-polyps":      "Dyed & Lifted Polyps",
    "dyed-resection-margins":  "Dyed Resection Margins",
    "esophagitis":             "Esophagitis",
    "normal-cecum":            "Normal Cecum",
    "normal-pylorus":          "Normal Pylorus",
    "normal-z-line":           "Normal Z-Line",
    "polyps":                  "Polyps",
    "ulcerative-colitis":      "Ulcerative Colitis",
}


class GIClassifier:
    """Loads a .keras model and classifies endoscopic frames."""

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.input_size = (224, 224)          # fallback; overwritten on load
        self._model_path = model_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "kvasir_gi_model.keras"
        )
        self._loaded = False
        self._load_error: str | None = None

    # ----- public API --------------------------------------------------------

    def load(self) -> bool:
        """Load the Keras model.  Returns True on success."""
        try:
            _ensure_tf()
            if not os.path.isfile(self._model_path):
                self._load_error = f"Model file not found:\n{self._model_path}"
                return False

            self.model = _load_model(self._model_path)

            # Discover expected input shape from the model itself
            inp_shape = self.model.input_shape          # e.g. (None, 224, 224, 3)
            if inp_shape and len(inp_shape) == 4:
                self.input_size = (inp_shape[1], inp_shape[2])

            self._loaded = True
            self._load_error = None
            return True

        except Exception as e:
            self._load_error = str(e)
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def error(self) -> str | None:
        return self._load_error

    def classify(self, frame: np.ndarray) -> tuple[str, float, dict]:
        """
        Classify a BGR frame.

        Returns
        -------
        class_name : str   – predicted Kvasir class key
        confidence : float – probability of the top class (0‒1)
        all_probs  : dict  – {class_name: probability} for every class
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded — call load() first.")

        img = self._preprocess(frame)
        preds = self.model.predict(img, verbose=0)[0]       # shape (8,)
        idx = int(np.argmax(preds))

        class_name = KVASIR_CLASSES[idx] if idx < len(KVASIR_CLASSES) else f"class_{idx}"
        confidence = float(preds[idx])
        all_probs  = {KVASIR_CLASSES[i]: float(preds[i]) for i in range(len(preds))}

        return class_name, confidence, all_probs

    # ----- internal ----------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize + normalise a BGR frame for the model."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
        return np.expand_dims(arr, axis=0)                  # (1, H, W, 3)
