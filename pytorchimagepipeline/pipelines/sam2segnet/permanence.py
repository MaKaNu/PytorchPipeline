from dataclasses import dataclass
from pathlib import Path

from torchvision.datasets import VisionDataset

from pytorchimagepipeline.abstractions import Permanence


@dataclass
class Datasets(Permanence):
    root: Path
    format: str

    def __post_init__(self):
        self.sam_dataset: VisionDataset = SamDataset(self.root)

    def cleanup(self):
        return super().cleanup()


class SamDataset(VisionDataset):
    def __init__(self, root=None, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return super().__getitem__(index)
