# file: my_2d_app/app.py
import os
from monailabel.interfaces.app import MONAILabelApp
from monailabel.datastore.local import LocalDatastore
from monailabel.interfaces.datastore import Datastore

# Importing our functions
from lib.infers import MySegmentationInfer
from lib.trainers import MySegmentationTrainer

class My2DApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        # Datastore (local) to point file .tiff
        datastore: Datastore = LocalDatastore(
            base_dir=studies,
            extensions=[".tiff", ".tif"], 
            auto_reload=True,
        )

        # Registering inference and training
        infers = [
            ("my_seg_infer", MySegmentationInfer()),
        ]
        trainers = [
            ("my_seg_trainer", MySegmentationTrainer()),
        ]

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            datastore=datastore,
            infers=infers,
            trainers=trainers,
        )

def create_app():
    return My2DApp(
        app_dir=os.path.dirname(__file__),
        studies=os.path.join(os.path.dirname(__file__), "min_results"),
        conf={"preload": True},
    )
