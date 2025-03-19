# file: my_2d_app/lib/trainers.py
import os
import torch
import monai
from monailabel.interfaces.tasks.train import TrainTask
from monailabel import TrainType
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelDimd, ScaleIntensityd, EnsureTyped
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet

class MySegmentationTrainer(TrainTask):
    def __init__(self):
        super().__init__(
            name="MySegTrainer",
            description="Simple 2D UNet Trainer",
            type=TrainType.SEGMENTATION
        )

    def train(self, datastore, device, **kwargs):
        # Obtaining the list of images with label
        labeled_images = datastore.get_labeled_images()  # dict: {id: {image, label}}
        data = []
        for image_id, meta in labeled_images.items():
            # "meta" has image and label
            data.append({"image": meta["image"], "label": meta["label"]})

        # Transformations
        train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelDimd(keys=["image", "label"]), 
            ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"]),
        ])
        dataset = Dataset(data=data, transform=train_transforms)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Initialization of net
        unet = UNet(
            dimensions=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=2
        ).to(device)

        # if existing model --> load that
        model_file = os.path.join(os.path.dirname(__file__), "model.pt")
        if os.path.exists(model_file):
            unet.load_state_dict(torch.load(model_file))

        # Loss and optimization function
        loss_function = monai.losses.DiceLoss(sigmoid=True)
        optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

        # Training loop base
        max_epochs = 5
        for epoch in range(max_epochs):
            unet.train()
            epoch_loss = 0
            for batch_data in loader:
                images = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)

                optimizer.zero_grad()
                outputs = unet(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            print(f"Epoch {epoch+1}/{max_epochs} - Loss: {epoch_loss:.4f}")

        # Save model
        torch.save(unet.state_dict(), model_file)
        return model_file
