

import sys
import os

# IMPORTANT: Import tensorflow BEFORE PyQt5 on Windows.
# PyQt5 and TensorFlow conflict over certain DLLs (like OpenMP) if PyQt5 loads first.
# Importing tensorflow first ensures its DLLs are initialized correctly.
try:
    import tensorflow
except ImportError:
    pass  # Handle gracefully if tensorflow is not installed

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Endoscopic Assistance System")
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()