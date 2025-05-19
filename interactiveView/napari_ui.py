# napari_ui_expanded.py

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
import re
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QProgressBar, QLabel, QScrollArea
)
from PyQt5.QtCore import QSize
from skimage.io import imread, imsave
import napari

class NapariTrainer(QWidget):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Napari RGB & Training UI")
        self.setFixedSize(600, 500)

        self.rgb_path_edit = QLineEdit(self)
        self.rgb_path_edit.setPlaceholderText("RGBintegrals Path")

        self.training_paths_layout = QVBoxLayout()
        self.add_training_path_input()

        self.prediction_name_edit = QLineEdit(self)
        self.prediction_name_edit.setPlaceholderText("Prediction Layer Name")

        self.model_save_path_edit = QLineEdit(self)
        self.model_save_path_edit.setPlaceholderText("Model Save Path")

        self.export_path_edit = QLineEdit(self)
        self.export_path_edit.setPlaceholderText("Export Predictions Path")

        self.progress_bar = QProgressBar(self)

        main_layout = QVBoxLayout(self)

        # RGB Loader
        rgb_layout = QHBoxLayout()
        rgb_layout.addWidget(QLabel("RGB Path:"))
        rgb_layout.addWidget(self.rgb_path_edit)
        browse_rgb_btn = self._button("📂")
        browse_rgb_btn.clicked.connect(self.browse_rgb)
        rgb_layout.addWidget(browse_rgb_btn)
        load_btn = self._button("Load RGB")
        load_btn.clicked.connect(self.load_rgb_napari)
        rgb_layout.addWidget(load_btn)

        main_layout.addLayout(rgb_layout)

        # Training Data Inputs
        train_label = QLabel("Training Data Paths:")
        main_layout.addWidget(train_label)

        train_scroll = QScrollArea()
        train_widget = QWidget()
        train_widget.setLayout(self.training_paths_layout)
        train_scroll.setWidget(train_widget)
        train_scroll.setWidgetResizable(True)
        train_scroll.setFixedHeight(150)

        add_training_btn = self._button("➕ Add Training Path")
        add_training_btn.clicked.connect(self.add_training_path_input)

        main_layout.addWidget(train_scroll)
        main_layout.addWidget(add_training_btn)

        # Prediction Name
        pred_layout = QHBoxLayout()
        pred_layout.addWidget(QLabel("Prediction Layer Name:"))
        pred_layout.addWidget(self.prediction_name_edit)
        main_layout.addLayout(pred_layout)

        # Export Path
        export_layout = QHBoxLayout()
        export_layout.addWidget(QLabel("Export Path:"))
        export_layout.addWidget(self.export_path_edit)
        browse_export_btn = self._button("📂")
        browse_export_btn.clicked.connect(self.browse_export)
        export_layout.addWidget(browse_export_btn)
        main_layout.addLayout(export_layout)

        # Buttons
        buttons_layout = QHBoxLayout()

        train_btn = self._button("Train")
        train_btn.clicked.connect(self.train_model)

        inference_btn = self._button("Inference")
        inference_btn.clicked.connect(self.run_inference)

        export_btn = self._button("Export")
        export_btn.clicked.connect(self.export_predictions)

        buttons_layout.addWidget(train_btn)
        buttons_layout.addWidget(inference_btn)
        buttons_layout.addWidget(export_btn)

        main_layout.addLayout(buttons_layout)
        main_layout.addWidget(self.progress_bar)

    def _button(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedSize(QSize(120, 40))
        return btn

    def browse_rgb(self):
        path = QFileDialog.getExistingDirectory(self, "Select RGBintegrals Folder")
        if path:
            self.rgb_path_edit.setText(path)

    def browse_export(self):
        path = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if path:
            self.export_path_edit.setText(path)

    def add_training_path_input(self):
        path_edit = QLineEdit(self)
        path_edit.setPlaceholderText("Training Data Path")
        browse_btn = self._button("📂")
        browse_btn.clicked.connect(lambda: self.browse_training_path(path_edit))

        layout = QHBoxLayout()
        layout.addWidget(path_edit)
        layout.addWidget(browse_btn)

        self.training_paths_layout.addLayout(layout)

    def browse_training_path(self, path_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select Training Folder")
        if path:
            path_edit.setText(path)

    def check_inputs(self):
        if not self.rgb_path_edit.text().strip():
            QMessageBox.warning(self, "Errore", "Percorso RGB non specificato.")
            return False
        if not self.model_save_path_edit.text().strip():
            QMessageBox.warning(self, "Errore", "Percorso di salvataggio modello non specificato.")
            return False
        if not self.prediction_name_edit.text().strip():
            QMessageBox.warning(self, "Errore", "Nome del layer di predizione non specificato.")
            return False
        if not self.export_path_edit.text().strip():
            QMessageBox.warning(self, "Errore", "Percorso export non specificato.")
            return False
        return True


    def load_rgb_napari(self):
        path = Path(self.rgb_path_edit.text().strip())
        image_paths = sorted(path.glob("layer_*.png"), key=lambda x: int(re.findall(r'\d+', x.stem)[0]))
        imgs = [imread(str(p)) for p in image_paths]
        rgb_stack = np.stack(imgs, axis=0)
        viewer = napari.Viewer()
        viewer.add_image(rgb_stack, rgb=True, name="RGB Integrals")
        viewer.add_labels(np.zeros(rgb_stack.shape[:-1], dtype=int), name="Manual Labels")

    def train_model(self):
        import torch
        from torch.utils.data import Dataset, DataLoader
        import torch.nn as nn
        import torch.nn.functional as F
        import numpy as np
        from skimage.io import imread
        import os
        from glob import glob
        from PyQt5.QtWidgets import QMessageBox, QApplication

        if not self.check_inputs():
            return

        class Simple2ClassNet(nn.Module):
            def __init__(self):
                super(Simple2ClassNet, self).__init__()
                self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.conv3 = nn.Conv2d(32, 2, 1)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = self.conv3(x)
                return x

        def partial_cross_entropy(logits, labels, min_w=1.0, max_w=50.0):
            criterion = nn.CrossEntropyLoss()
            return criterion(logits, labels)

        class NapariMultiLayerDataset(Dataset):
            def __init__(self, images_list, masks_list):
                self.images = [img.astype(np.float32) for img in images_list]
                self.masks = [mask.astype(np.int64) for mask in masks_list]

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img = self.images[idx]
                mask = self.masks[idx]
                img_tensor = torch.from_numpy(np.expand_dims(img, axis=0))
                mask_tensor = torch.from_numpy(mask)
                return img_tensor, mask_tensor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL_DIR = self.model_save_path_edit.text().strip()
        if not MODEL_DIR:
            QMessageBox.warning(self, "Errore", "Specifica un percorso valido per salvare il modello.")
            return
        os.makedirs(MODEL_DIR, exist_ok=True)

        try:
            training_paths = []
            for i in range(self.training_paths_layout.count()):
                layout = self.training_paths_layout.itemAt(i)
                if layout is not None:
                    path_edit = layout.itemAt(0).widget()
                    path = path_edit.text().strip()
                    if path:
                        training_paths.append(path)

            if not training_paths:
                QMessageBox.warning(self, "Errore", "Nessun percorso di training valido.")
                return

            all_images_to_train = []
            all_masks_to_train = []

            for path in training_paths:
                image_files = sorted(glob(os.path.join(path, 'layer_*.png')))
                mask_files = sorted(glob(os.path.join(path, 'mask_*.png')))

                for img_file, mask_file in zip(image_files, mask_files):
                    img = imread(img_file).astype(np.float32) / 255.0
                    mask = imread(mask_file).astype(np.int64)
                    all_images_to_train.append(img)
                    all_masks_to_train.append(mask)

            if not all_images_to_train:
                QMessageBox.warning(self, "Errore", "Nessuna immagine trovata per il training.")
                return

            dataset = NapariMultiLayerDataset(all_images_to_train, all_masks_to_train)
            loader = DataLoader(dataset, batch_size=4, shuffle=True)

            model = Simple2ClassNet().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            epochs = 60
            self.progress_bar.setMaximum(epochs)

            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for imgs, masks in loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    optimizer.zero_grad()
                    logits = model(imgs)
                    loss = partial_cross_entropy(logits, masks)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(loader)
                self.progress_bar.setValue(epoch + 1)
                QApplication.processEvents()

            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "trained_model.pt"))
            QMessageBox.information(self, "Training", f"Training completato con successo. Avg Loss: {avg_loss:.4f}")

        except Exception as exc:
            QMessageBox.critical(self, "Errore", str(exc))
            traceback.print_exc()

    def run_inference(self):
        import torch
        from skimage.io import imread
        from glob import glob
        import numpy as np
        from torch.utils.data import Dataset, DataLoader
        import torch.nn as nn
        import torch.nn.functional as F
        import os
        from PyQt5.QtWidgets import QMessageBox, QApplication

        if not self.check_inputs():
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        MODEL_DIR = self.model_save_path_edit.text().strip()
        model_path = os.path.join(MODEL_DIR, "trained_model.pt")
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Errore", "Modello non trovato. Devi prima addestrare il modello.")
            return

        class Simple2ClassNet(nn.Module):
            def __init__(self):
                super(Simple2ClassNet, self).__init__()
                self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.conv3 = nn.Conv2d(32, 2, 1)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = self.conv3(x)
                return x

        @torch.no_grad()
        def infer_mask(model, img):
            img_tensor = torch.from_numpy(img[np.newaxis, np.newaxis, ...]).to(device)
            logits = model(img_tensor)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            return pred.astype(np.int8)

        # Carica modello
        model = Simple2ClassNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Esegui inferenza sulle immagini RGB caricate
        rgb_path = Path(self.rgb_path_edit.text().strip())
        image_files = sorted(rgb_path.glob("layer_*.png"), key=lambda x: int(re.findall(r'\d+', x.stem)[0]))
        
        if not image_files:
            QMessageBox.warning(self, "Errore", "Nessuna immagine RGB trovata per l'inferenza.")
            return

        self.progress_bar.setMaximum(len(image_files))
        predictions = []
        for idx, img_file in enumerate(image_files):
            img = imread(str(img_file)).astype(np.float32) / 255.0
            prediction_mask = infer_mask(model, img)
            predictions.append(prediction_mask)
            self.progress_bar.setValue(idx + 1)
            QApplication.processEvents()

        prediction_stack = np.stack(predictions, axis=0)
        layer_name = self.prediction_name_edit.text().strip()

        # Apri risultati in napari
        viewer = napari.current_viewer()
        if viewer is None:
            viewer = napari.Viewer()
        viewer.add_labels(prediction_stack, name=layer_name, opacity=0.4)

        QMessageBox.information(self, "Inference", f"Inference completata, risultati nel layer '{layer_name}'.")


    def export_predictions(self):
        import torch
        from torch.utils.data import Dataset, DataLoader
        import torch.nn as nn
        import torch.nn.functional as F
        import numpy as np
        from skimage.io import imread
        import os
        from glob import glob
        from PyQt5.QtWidgets import QMessageBox, QApplication

        if not self.check_inputs():
            return

        export_path = Path(self.export_path_edit.text().strip())
        export_path.mkdir(parents=True, exist_ok=True)

        viewer = napari.current_viewer()
        layer_name = self.prediction_name_edit.text().strip()

        if viewer is None or layer_name not in viewer.layers:
            QMessageBox.warning(self, "Errore", f"Layer '{layer_name}' non trovato in Napari.")
            return

        predictions = viewer.layers[layer_name].data

        for idx, pred in enumerate(predictions):
            imsave(export_path / f"prediction_{idx}.png", (pred * 255).astype(np.uint8))

        QMessageBox.information(self, "Export", f"Predizioni esportate correttamente in '{export_path}'.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NapariTrainer()
    window.show()
    sys.exit(app.exec_())
