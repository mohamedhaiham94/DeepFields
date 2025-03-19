# file: my_2d_app/lib/infers.py
import os
import torch
import monai
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType
)

class MySegmentationInfer(InferTask):
    def __init__(self):
        super().__init__(
            path=os.path.join(os.path.dirname(__file__), "model.pt"),
            type=InferType.SEGMENTATION
        )
        self.network = None

    def pre_transforms(self):
        # transformation before inference
        return Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            EnsureType(),
        ])

    def post_transforms(self):
        return Compose([
            EnsureType(),
        ])

    def inferer(self):
        return monai.inferers.SimpleInferer()

    def load(self, data=None):
        # Load network if not done
        if not self.network:
            unet = monai.networks.nets.UNet(
                dimensions=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64),
                strides=(2, 2),
                num_res_units=2
            )
            # if a model exists
            if os.path.exists(self.path):
                unet.load_state_dict(torch.load(self.path, map_location="cpu"))
            self.network = unet
        return self.network
