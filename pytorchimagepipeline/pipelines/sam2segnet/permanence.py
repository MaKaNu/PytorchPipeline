from dataclasses import dataclass
from pathlib import Path

import cv2
import torchvision
from torchvision.datasets import VisionDataset

from pytorchimagepipeline.abstractions import Permanence
from pytorchimagepipeline.pipelines.sam2segnet.utils import parse_voc_xml


@dataclass
class Datasets(Permanence):
    """
    Datasets class is a class which provides torch datasets.

    Example TOML Config:
    ```toml
    [permanences.data]
    type = "Datasets"
    params = { root = "data/datasets/pascal", format = "pascalvoc" }
    ```

    Attributes:
        root (Path): The root directory for the dataset.
        format (str): The format of the dataset.

    Methods:
        __post_init__(): Initializes the sam_dataset attribute with a VisionDataset instance.
        cleanup(): Placeholder method for cleanup operations.
    """

    root: Path
    format: str

    def __post_init__(self):
        self.sam_dataset: VisionDataset = SamDataset(self.root)

    def cleanup(self):
        return super().cleanup()


class SamDataset(VisionDataset):
    def __init__(self, root=None):
        self.root = Path(root)

        self.images = sorted([p for p in (self.root / "Images").iterdir() if p.suffix == ".png"])
        self.annotations = sorted([p for p in (self.root / "Annotations").iterdir() if p.suffix == ".xml"])

        with (self.root / "classes.json").open() as file_obj:
            self.class_idx = json.load(file_obj)

        self.target_location = self.root / "SegmentationClassSAM"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Read Image
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read filestem
        filestem = self.images[index].stem

        # Read Annotations
        annotation, error = parse_voc_xml(self.annotations[index])
        if error:
            raise error
        bboxes = [obj.bndbox for obj in annotation.objects]
        bbox_classes = [obj.name for obj in annotation.objects]

        return img, bboxes, bbox_classes, filestem

    def save_item(self, index, mask):
        mask = mask.squeeze()
        torchvision.utils.save_image(mask, str(self.target_location / self.images[index].name), "png")
