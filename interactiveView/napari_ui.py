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

        # Model Save Path
        model_save_layout = QHBoxLayout()
        model_save_layout.addWidget(QLabel("Model Save Path:"))
        model_save_layout.addWidget(self.model_save_path_edit)
        browse_model_save_btn = self._button("📂")
        browse_model_save_btn.clicked.connect(self.browse_model_save)
        model_save_layout.addWidget(browse_model_save_btn)
        main_layout.addLayout(model_save_layout)


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

    def browse_model_save(self):
        path = QFileDialog.getExistingDirectory(self, "Select Model Save Folder")
        if path:
            self.model_save_path_edit.setText(path)

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
    
    def save_manual_masks(self):
        viewer = napari.current_viewer()
        if viewer is None:
            QMessageBox.warning(self, "Errore", "Napari non aperto.")
            return

        if "Manual Labels" not in viewer.layers:
            QMessageBox.warning(self, "Errore", "Layer 'Manual Labels' non trovato.")
            return

        masks = viewer.layers["Manual Labels"].data

        save_dir = Path(self.model_save_path_edit.text().strip())
        save_dir.mkdir(parents=True, exist_ok=True)

        for idx, mask in enumerate(masks):
            mask_filename = save_dir / f"mask_{idx}.tiff"
            imsave(str(mask_filename), mask.astype(np.uint8))

        QMessageBox.information(self, "Salvataggio", f"Maschere salvate correttamente in '{save_dir}'.")

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
        from skimage.io import imread, imsave
        import os
        from glob import glob
        from PyQt5.QtWidgets import QMessageBox, QApplication
        import napari
        import re

        QApplication.processEvents()
        if not self.check_inputs():
            return

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        MODEL_DIR = self.model_save_path_edit.text().strip()
        if not MODEL_DIR:
            QMessageBox.warning(self, "Errore", "Specifica un percorso valido per salvare il modello.")
            return
        os.makedirs(MODEL_DIR, exist_ok=True)

        viewer = napari.current_viewer()
        if viewer is None:
            QMessageBox.warning(self, "Errore", "Napari non aperto.")
            return

        if "Manual Labels" not in viewer.layers:
            QMessageBox.warning(self, "Errore", "Layer 'Manual Labels' non trovato.")
            return

        manual_masks = viewer.layers["Manual Labels"].data

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

        num_layers = manual_masks.shape[0]

        # Raccogli solo layer annotati con 2 o 3
        annotated_layers = [z for z in range(num_layers) if np.any(np.isin(manual_masks[z], [2, 3]))]

        # Salva le maschere manuali annotate prima del training
        for z in annotated_layers:
            mask_filename = os.path.join(MODEL_DIR, f"mask_{z}.tiff")
            imsave(mask_filename, manual_masks[z].astype(np.uint8))


        if not annotated_layers:
            QMessageBox.warning(self, "Errore", "Nessun layer annotato trovato con etichette valide (2 o 3).")
            return

        # Prepara immagini e maschere
        all_images_to_train = []
        all_masks_to_train = []

        for z in annotated_layers:
            channels_images = []
            for channel_path in training_paths:
                # Trova il file corrispondente usando regex generico
                pattern = os.path.join(channel_path, f"*_{z}.tiff")
                matches = glob(pattern)
                if not matches:
                    QMessageBox.warning(self, "Errore", f"Nessuna immagine trovata per il layer {z} nel path {channel_path}.")
                    return
                # img = imread(matches[0]).astype(np.float32) / 255.0
                img = imread(matches[0]).astype(np.float32)
                if img.max() > 1.5:          # solo se uint8/uint16
                    img = img / 255.0
                channels_images.append(img)

            stacked_img = np.stack(channels_images, axis=0)  # (canali, H, W)
            mask = manual_masks[z]
            # processed_mask = np.where(mask == 2, 1, np.where(mask == 3, 0, -1)).astype(np.int64)  # 2→1, 3→0, 0→-1 (ignore)

            all_images_to_train.append(stacked_img)
            # all_masks_to_train.append(processed_mask)
            all_masks_to_train.append(mask.astype(np.int64))   # NIENTE rimappatura


        # Definisci la loss originale personalizzata
        def partial_cross_entropy(logits, labels, min_w=1.0, max_w=200.0):
            unlabeled = (labels == 0)
            # print(f"  Loss - Valori unici etichette input (lbls): {torch.unique(lbls)}") 
            target    = torch.where(labels == 2, 1, 0)          # 2→1 (fuoco), 3→0 (sfocato)
            # print(f"  Loss - Valori unici target (0/1): {torch.unique(target)}")

            # raddrizza tensori
            logits = logits.permute(0, 2, 3, 1).reshape(-1, 2)
            target = target.reshape(-1)
            mask   = (~unlabeled.reshape(-1))

            # print(f"  Loss - Numero pixel etichettati (mask.sum()): {mask.sum().item()}")

            if not mask.any():
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            # pixel etichettati
            logits_lab = logits[mask]
            target_lab = target[mask]
            # print(f"  Loss - Shape target_lab: {target_lab.shape}")

            # ---- P E S I  D Y N A M I C I -----------------------------------
            n_total = target_lab.numel()
            n_pos   = (target_lab == 1).sum()
            n_neg   = n_total - n_pos

            # print(f"  Loss - n_total: {n_total}, n_pos (in-focus): {n_pos.item()}, n_neg (out-of-focus): {n_neg.item()}") # <-- CONTROLLA QUI!

            # evita divisione per zero
            if n_pos == 0 or n_neg == 0:
                weight = torch.tensor([1.0, 1.0], device=logits.device)
            else:
                # peso inversamente proporzionale alla frequenza
                w_neg = torch.clamp(n_total / (2.0 * n_neg.float()), min=min_w, max=max_w)
                w_pos = torch.clamp(n_total / (2.0 * n_pos.float()), min=min_w, max=max_w)
                weight = torch.tensor([w_neg, w_pos], device=logits.device)
                # print(f"  Loss - Pesi calcolati: {weight}")

            loss = F.cross_entropy(logits_lab, target_lab, weight=weight, reduction='mean')
            # print(f"  Loss - Valore loss calcolato: {loss.item()}") # <-- CONTROLLA QUI!
            return loss

        class NapariMultiLayerDataset(Dataset):
            def __init__(self, images_list, masks_list):
                self.images = images_list
                self.masks = masks_list

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return torch.from_numpy(self.images[idx]), torch.from_numpy(self.masks[idx])

        num_input_channels = len(training_paths)

        class Simple2ClassNet(nn.Module):
            def __init__(self, num_channels):
                super(Simple2ClassNet, self).__init__()
                self.conv1 = nn.Conv2d(num_channels, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.conv3 = nn.Conv2d(32, 2, 1)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = self.conv3(x)
                return x

        dataset = NapariMultiLayerDataset(all_images_to_train, all_masks_to_train)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        model = Simple2ClassNet(num_input_channels).to(device)
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



    def run_inference(self):
        # Import necessari specifici per la funzione, se non già globali
        import torch
        from skimage.io import imread
        import numpy as np
        import torch.nn as nn
        import torch.nn.functional as F
        import os
        from glob import glob
        from PyQt5.QtWidgets import QMessageBox, QApplication
        import re # Assicurati sia importato
        from pathlib import Path # Assicurati sia importato
        import traceback # Assicurati sia importato

        QApplication.processEvents()
        if not self.check_inputs():
            return

        try: # Blocco try-except generale
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            MODEL_DIR = self.model_save_path_edit.text().strip()
            model_path = os.path.join(MODEL_DIR, "trained_model.pt")

            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Errore", "Modello non trovato. Devi prima addestrare il modello.")
                return

            # Raccogli i percorsi come stringhe
            training_paths_str = []
            for i in range(self.training_paths_layout.count()):
                layout = self.training_paths_layout.itemAt(i)
                if layout is not None:
                    path_edit = layout.itemAt(0).widget()
                    path_text = path_edit.text().strip()
                    if path_text:
                        training_paths_str.append(path_text)

            if not training_paths_str:
                QMessageBox.warning(self, "Errore", "Nessun percorso di training valido per l'inferenza.")
                return

            # Converti le stringhe in oggetti Path
            training_paths = [Path(p_str) for p_str in training_paths_str]
            actual_num_channels = len(training_paths)

            # Definizione del modello (assicurati che sia consistente con il training)
            class Simple2ClassNet(nn.Module):
                def __init__(self, num_input_channels_arg):
                    super(Simple2ClassNet, self).__init__()
                    self.conv1 = nn.Conv2d(num_input_channels_arg, 16, 3, padding=1)
                    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                    self.conv3 = nn.Conv2d(32, 2, 1)

                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    x = F.relu(self.conv2(x))
                    x = self.conv3(x)
                    return x

            @torch.no_grad()
            def infer_mask(model, img):
                img_tensor = torch.from_numpy(img[np.newaxis, ...]).to(device)
                logits = model(img_tensor)
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                return pred.astype(np.int8)

            model = Simple2ClassNet(actual_num_channels).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            channel_images = []
            for channel_path_obj in training_paths:
                if not channel_path_obj.is_dir():
                    QMessageBox.warning(self, "Errore", f"Il percorso {channel_path_obj} non è una directory valida.")
                    self.progress_bar.setValue(0)
                    return
                
                imgs = sorted(glob(str(channel_path_obj / "*_*.tiff")), key=lambda x: int(re.findall(r'\d+', Path(x).stem)[-1]))
                if not imgs:
                    QMessageBox.warning(self, "Attenzione", f"Nessun file .tiff trovato in {channel_path_obj} con il pattern '*_*.tiff'.")
                    self.progress_bar.setValue(0)
                    return
                channel_images.append(imgs)

            if not channel_images:
                QMessageBox.warning(self, "Errore", "Nessuna immagine trovata per l'inferenza.")
                self.progress_bar.setValue(0)
                return

            num_images_inference = len(channel_images[0])
            for i, ch_imgs in enumerate(channel_images):
                if len(ch_imgs) != num_images_inference:
                    QMessageBox.warning(self, "Errore", f"Numero immagini diverso nei canali. Canale 0 ha {num_images_inference}, canale {i} ha {len(ch_imgs)}.")
                    self.progress_bar.setValue(0)
                    return

            predictions = []
            self.progress_bar.setMaximum(num_images_inference)
            self.progress_bar.setValue(0) # Inizia da 0

            for idx in range(num_images_inference):
                current_image_channels_data = []
                for ch_idx in range(actual_num_channels):
                    image_path_to_load = channel_images[ch_idx][idx]
                    try:
                        # img_data = imread(image_path_to_load).astype(np.float32) / 255.0
                        img_data = imread(image_path_to_load).astype(np.float32)
                        if img_data.max() > 1.5:
                            img_data = img_data / 255.0
                        current_image_channels_data.append(img_data)
                    except Exception as e_read:
                        tb_str_read = traceback.format_exc()
                        QMessageBox.critical(self, "Errore Lettura Immagine", f"Impossibile leggere/processare: {image_path_to_load}\n{str(e_read)}\n\nTraceback:\n{tb_str_read}")
                        self.progress_bar.setValue(0)
                        return
                
                stacked_img = np.stack(current_image_channels_data, axis=0)
                prediction_mask = infer_mask(model, stacked_img)
                predictions.append(prediction_mask)
                self.progress_bar.setValue(idx + 1)
                QApplication.processEvents()

            prediction_stack = np.stack(predictions, axis=0)
            layer_name = self.prediction_name_edit.text().strip()

            viewer = napari.current_viewer()
            if viewer is None:
                viewer = napari.Viewer() # Crea un nuovo viewer se non esiste
            
            if layer_name in viewer.layers:
                # Rimuovi il layer esistente o chiedi all'utente
                confirm = QMessageBox.question(self, "Layer Esistente", 
                                               f"Il layer '{layer_name}' esiste già. Vuoi sovrascriverlo?",
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if confirm == QMessageBox.Yes:
                    viewer.layers.pop(layer_name)
                else:
                    QMessageBox.information(self, "Inference", "Inferenza annullata. Scegli un nome diverso per il layer.")
                    self.progress_bar.setValue(0)
                    return

            viewer.add_labels(prediction_stack, name=layer_name, opacity=0.4)
            QMessageBox.information(self, "Inference", f"Inference completata, risultati nel layer '{layer_name}'.")

        except Exception as e:
            tb_str = traceback.format_exc()
            QMessageBox.critical(self, "Errore Inatteso in run_inference",
                                 f"Si è verificato un errore:\n{str(e)}\n\nTraceback:\n{tb_str}")
            print(f"ERRORE in run_inference: {str(e)}\n{tb_str}")
            self.progress_bar.setValue(0)


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
