# napari_ui.py
"""
Piccola interfaccia Qt per caricare rapidamente uno stack di immagini RGB
in **napari** e annotarlo.

⚙️  **Dipendenze** (consigliato un venv):
    pip install "napari[all]" PyQt5 scikit-image numpy

▶️  **Esecuzione** (da terminale):
    python napari_ui.py

📦  **Packaging macOS**
    
    Con PyInstaller ≥ 6.x una riga basta (modalità *onedir*; niente
    `--onefile`).  Le due opzioni *collect* assicurano che **tutti** i moduli
    pigri (lazy‑loader) di napari/vispy finiscano dentro lʼapp, evitando gli
    errori «No module named …».  Lʼ`--hidden-import` è un extra di sicurezza
    per versioni vecchie dei relativi hook.

    pyinstaller --noconfirm --windowed --name NapariRGBLoader \
                --collect-all napari \
                --collect-submodules napari \
                --hidden-import napari.viewer \
                --collect-all vispy \
                --collect-submodules vispy \
                napari_ui.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
import re

# -----------------------------------------------------------------------------
# Fix per applicazione "frozen" (PyInstaller): indica a Qt dove trovare i plugin
# -----------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    qt_plugins = Path(sys._MEIPASS) / "PyQt5" / "Qt" / "plugins"  # type: ignore[attr-defined]
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_plugins))

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import QSize
import numpy as np
from skimage.io import imread
import napari

# -----------------------------------------------------------------------------
# "napari.viewer" è un sub‑module importato *dinamicamente*; lo importiamo qui
# esplicitamente così PyInstaller lo vede e lo include nel bundle.
# -----------------------------------------------------------------------------
import napari.viewer  # noqa: F401 – import for side‑effects only


class NapariLoader(QWidget):
    """Finestra principale con barra percorso + pulsanti colorati."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Napari RGB Loader")
        self.setFixedSize(420, 160)

        main_layout = QVBoxLayout(self)

        # ------------------------------------------------------------------
        # riga percorso + pulsante sfoglia
        # ------------------------------------------------------------------
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText("Cartella RGBintegrals (…/RGBintegrals)")
        path_layout.addWidget(self.path_edit)

        browse_btn = self._colored_button("📂", "#4dabf7")
        browse_btn.clicked.connect(self._browse)
        path_layout.addWidget(browse_btn)
        main_layout.addLayout(path_layout)

        # ------------------------------------------------------------------
        # pulsanti Avvia ed Esci
        # ------------------------------------------------------------------
        btn_layout = QHBoxLayout()
        run_btn = self._colored_button("▶", "#51cf66")
        run_btn.clicked.connect(self._run_napari)
        btn_layout.addWidget(run_btn)

        quit_btn = self._colored_button("✕", "#ff6b6b")
        quit_btn.clicked.connect(self.close)
        btn_layout.addWidget(quit_btn)
        main_layout.addLayout(btn_layout)

    # ----------------------------------------------------------------------
    # helper
    # ----------------------------------------------------------------------
    @staticmethod
    def _colored_button(text: str, color: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedSize(QSize(60, 60))
        btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {color};
                border: none;
                color: white;
                font-size: 24px;
                border-radius: 6px;
            }}
            QPushButton:hover {{ opacity: 0.85; }}
            """
        )
        return btn

    # ------------------------------------------------------------------
    # slot
    # ------------------------------------------------------------------
    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Seleziona la cartella RGBintegrals"
        )
        if folder:
            self.path_edit.setText(folder)

    def _run_napari(self) -> None:
        """Carica le immagini e avvia il viewer napari in modo sicuro."""
        try:
            rgb_folder = Path(self.path_edit.text().strip())
            if not rgb_folder.is_dir():
                raise ValueError("Cartella inesistente o non selezionata")

            image_paths = sorted(
                rgb_folder.glob("layer_*.png"),
                key=lambda x: int(re.findall(r'\d+', x.stem)[0])
            )
            if not image_paths:
                raise ValueError("Nessun file layer_*.png trovato nella cartella")

            # Caricamento immagini (tutte devono avere la stessa shape)
            imgs = [imread(str(p)) for p in image_paths]
            shapes = {im.shape for im in imgs}
            if len(shapes) != 1:
                raise ValueError(f"Le immagini hanno dimensioni diverse: {shapes}")

            rgb_stack = np.stack(imgs, axis=0)  # (N, H, W, 3)

            # Avvia napari con visualizzazione RGB corretta
            viewer = napari.Viewer()
            viewer.add_image(rgb_stack, rgb=True, name="RGB Integrals")
            viewer.add_labels(
                np.zeros(rgb_stack.shape[:-1], dtype=int), name="Manual Labels"
            )
            viewer.window._qt_window.move(self.x() + 100, self.y() + 100)
        except Exception as exc:
            QMessageBox.critical(self, "Errore", str(exc))
            traceback.print_exc()



# -----------------------------------------------------------------------------
def main() -> None:
    app = QApplication(sys.argv)
    window = NapariLoader()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
