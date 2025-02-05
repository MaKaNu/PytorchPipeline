import dataclasses
import importlib
import json
from dataclasses import dataclass, field
from logging import warning
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.datasets import VisionDataset
from torchvision.io import decode_image
from torchvision.tv_tensors import Mask

from pytorchimagepipeline.abstractions import Permanence
from pytorchimagepipeline.pipelines.sam2segnet.errors import (
    FormatNotSupportedError,
    MaskNotAvailable,
    MaskShapeError,
    ModeError,
    ModelNotSupportedError,
)
from pytorchimagepipeline.pipelines.sam2segnet.utils import parse_voc_xml


@dataclass
class PascalVocFormat:
    root: Path
    mean_std: dict = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)
    data: list[str] = field(default_factory=list)

    def __post_init__(self):
        with (self.root / "mean_std.json").open() as file_obj:
            self.mean_std = json.load(file_obj)

        with (self.root / "classes.json").open() as file_obj:
            self.classes = json.load(file_obj)

    def __len__(self):
        return len(self.data)

    def get_data(self, mode):
        data_file = self.root / f"ImageSets/Segmentation/{mode}.txt"

        if data_file.exists():
            with data_file.open() as file_obj:
                self.data = file_obj.read().split("\n")
        else:
            self.data = []


@dataclass
class Datasets(Permanence):
    """
    Datasets class is a class which provides torch datasets.

    Example TOML Config:
    ```toml
    [permanences.data]
    type = "Datasets"
    params = { root = "data/datasets/pascal", data_format = "pascalvoc" }
    ```

    Attributes:
        root (Path): The root directory for the dataset.
        data_format (str): The format of the dataset.

    Methods:
        __post_init__(): Initializes the sam_dataset attribute with a VisionDataset instance.
        cleanup(): Placeholder method for cleanup operations.
    """

    root: Path
    data_format: str

    def __post_init__(self):
        self.sam_dataset: VisionDataset = SamDataset(self.root)
        self.segnet_dataset_train: VisionDataset = SegnetDataset(self.root, data_format=self.data_format, mode="train")
        self.segnet_dataset_val: VisionDataset = SegnetDataset(self.root, data_format=self.data_format, mode="val")
        self.segnet_dataset_test: VisionDataset = SegnetDataset(self.root, data_format=self.data_format, mode="test")

    def cleanup(self):
        pass


@dataclass
class MaskCreator(Permanence):
    """
    MaskCreator is a class that performs various morphological operations on binary masks.

    Example TOML Config:
    ```toml
    [permanences.mask_creator]
    type = "MaskCreator"
    params = { morph_size = 1 , border_size = 1, ignore_value = 255 }
    ```

    Attributes:
        morph_size (int): Size of the morphological kernel. Default is 3.
        border_size (int): Size of the border to be created around the mask. Default is 4.
        ignore_value (int): Value to be used for ignored regions in the mask. Default is 255.
        current_masks (torch.Tensor): The current masks being processed.

    Methods:
        cleanup():
            Placeholder method for cleanup operations.

        set_current_masks(masks):
            Sets the current masks to the provided masks.

        create_mask(masks, masks_classes):
            Creates a mask by performing a series of morphological operations and merging masks with classes.

        merge_masks(mask_classes):
            Merges stacked binary masks where higher indices have priority.

        _create_border():
            Creates a border around the current masks using dilation.

        _erode(kernel_size=3, padding=1):
            Erodes the current masks using a max pooling operation.

        _dilate(kernel_size=3, padding=1):

        _opening():
            Performs an opening operation (erosion followed by dilation) on the current masks.

        _closing():
            Performs a closing operation (dilation followed by erosion) on the current masks.

        _get_kernel_size():
            Calculates and returns the kernel size based on the morphological size.
    """

    morph_size: int = 3
    border_size: int = 4
    ignore_value: int = 255
    current_masks: torch.Tensor = None

    def cleanup(self):
        pass

    def set_current_masks(self, masks: torch.ByteTensor) -> None:
        self.current_masks = masks

    def create_mask(
        self, masks: torch.FloatTensor | torch.cuda.FloatTensor
    ) -> torch.ByteTensor | torch.cuda.ByteTensor:
        """
        Creates and processes a mask tensor by applying a series of morphological operations and
        merging masks based on their position.

        Args:
            masks (torch.FloatTensor | torch.cuda.FloatTensor): A tensor containing the initial masks.

        Returns:
            torch.ByteTensor | torch.cuda.ByteTensor: The processed mask tensor after applying
            closing, opening, border creation and merging operations.
        """
        self.set_current_masks(masks)
        self._check_masks()
        self._closing()
        self._opening()
        self._create_border()
        self._merge_masks()
        return self.current_masks.type(torch.uint8)

    def _check_masks(self):
        """
        Checks the validity of the current masks.

        Raises:
            MaskNotAvailable: If `self.current_masks` is None.
            MaskShapeError: If `self.current_masks` does not have 4 dimensions.

        Warnings:
            If `self.current_masks` is not of type `torch.float`, a warning is issued and the masks are converted to float.
        """
        if self.current_masks is None:
            raise MaskNotAvailable()
        if len(self.current_masks.shape) != 4:
            raise MaskShapeError(self.current_masks.shape)
        if self.current_masks.dtype != torch.float32:
            warning(UserWarning("Masks are not in float32 format. Converting to float32."))
            self.set_current_masks(self.current_masks.type(torch.float32))

    def _merge_masks(self):
        """
        Merge stacked binary masks where higher indices have priority

        Args:
            stacked_masks (torch.Tensor): Shape (N, H, W) where N is number of masks

        Returns:
            torch.Tensor: Shape (H, W) merged mask
        """
        result = torch.zeros_like(self.current_masks[0])
        for mask in self.current_masks:
            result[mask > 0] = mask[mask > 0]
        self.set_current_masks(result)

    def _create_border(self):
        """
        Creates a border around the current masks by performing max pooling with a specified kernel size and padding.
        The mask + current mask are saved as the new current mask.

        Args:
            None

        Returns:
            None
        """
        kernel_size = 2 * self.border_size + 1
        padding = self.border_size

        # Perform max pooling to create the border effect
        dilated = self._dilate(kernel_size, padding)

        # Border is where dilated is 1 but mask is 0
        border = (dilated - self.current_masks).bool() * self.ignore_value

        # Add the border to the mask
        self.set_current_masks(self.current_masks + border)

    def _erode(self, kernel_size=3, padding=1):
        """
        Dilates the current masks using a max pooling operation.
        This Implementation is only used for binary masks.
        Morphological Operations for grayscale like described in the folowing articel are not implemented:
        https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm


        Args:
            kernel_size (int, optional): The size of the kernel to use for dilation. Default is 3.
            padding (int, optional): The amount of padding to add to the masks before dilation. Default is 1.

        Returns:
            torch.Tensor: The dilated masks.
        """
        masks = self.current_masks
        dilated = -F.max_pool2d(-masks, kernel_size=kernel_size, stride=1, padding=padding)
        return dilated

    def _dilate(self, kernel_size=3, padding=1):
        """
        Dilates the current masks using a max pooling operation.
        This Implementation is only used for binary masks.
        Morphological Operations for grayscale like described in the folowing articel are not implemented:
        https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm


        Args:
            kernel_size (int, optional): The size of the kernel to use for dilation. Default is 3.
            padding (int, optional): The amount of padding to add to the masks before dilation. Default is 1.

        Returns:
            torch.Tensor: The dilated masks.
        """
        masks = self.current_masks
        dilated = F.max_pool2d(masks, kernel_size=kernel_size, stride=1, padding=padding)
        return dilated

    def _opening(self):
        kernel_size = self._get_kernel_size()
        padding = self.morph_size
        self.set_current_masks(self._erode(kernel_size=kernel_size, padding=padding))
        self.set_current_masks(self._dilate(kernel_size=kernel_size, padding=padding))

    def _closing(self):
        kernel_size = self._get_kernel_size()
        padding = self.morph_size
        self.current_masks = self._dilate(kernel_size=kernel_size, padding=padding)
        self.current_masks = self._erode(kernel_size=kernel_size, padding=padding)

    def _get_kernel_size(self):
        return 2 * self.morph_size + 1


@dataclass
class Network(Permanence):
    """
    Network is a class that provides a network to perform semantic segmentation.

    Example TOML Config:
    ```toml
    [permanences.network]
    type = "Network"
    params = { model = "deeplabv3_resnet50", num_classes = 21, pretrained = True }
    ```

    Attributes:
        model (str): The name of the model to use.
        num_classes (int): The number of classes in the dataset.
        pretrained (bool): Whether to use a pretrained model or not.
        model_instance (torch.nn.Module): The instance of the model.

    Methods:
        cleanup():
            Placeholder method for cleanup operations.
    """

    model: str
    num_classes: int
    pretrained: bool

    model_instance: torch.nn.Module = None

    def __post_init__(self):
        self.implemented_models = [
            "fcn_resnet50",
            "fnc_resnet101",
            "deeplabv3_resnet50",
            "deeplabv3_resnet101",
            "deeplabv3_mobilenet_v3_large",
            "lsrap_mobilenet_v3_large",
        ]
        self._load_model()

    def cleanup(self):
        pass

    def _load_model(self):
        if self.model not in self.implemented_models:
            raise ModelNotSupportedError(self.model, self.implemented_models)
        get_model_func = importlib.import_module(f"torchvision.models.segmentation.{self.model}")

        self.model_instance = get_model_func(pretrained=self.pretrained, num_classes=self.num_classes)


class SamDataset(VisionDataset):
    def __init__(self, root=None):
        self.root = Path(root)

        self.images = sorted([p for p in (self.root / "JPEGImages").iterdir() if p.suffix == ".jpg"])
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


class SegnetDataset(VisionDataset):
    def __init__(self, root=None, transforms=None, mode="train", data_container=None):
        super().__init__(root, transforms=transforms)
        if mode not in ["train", "val", "test"]:
            raise ModeError(mode)
        self.root = Path(root)
        self.transforms = transforms
        self.mode = mode

        self.dataobj = type(data_container)(**dataclasses.asdict(data_container))
        self.dataobj.get_data(mode)

        self._get_data_func()

    def __len__(self):
        return len(self.dataobj)

    def __getitem__(self, index):
        file_name = self.dataobj.data[index]
        return self.get_data(file_name)

    def _get_data_func(self):
        def _get_train_data(file_name):
            # Read Image
            img = decode_image(self.root / "JPEGImages" / f"{file_name}.jpg")

            # Read Mask
            mask = Mask(decode_image(self.root / "SegmentationClassSAM" / f"{file_name}.png"))

            if self.transforms:
                img, mask = self.transforms(img, mask)
            return img, mask

        def _get_val_test_data(file_name):
            # Read Image
            img = decode_image(self.root / "JPEGImages" / f"{file_name}.jpg")

            if self.transforms:
                img = self.transforms(img)
            return img

        if self.mode == "train":
            self.get_data = _get_train_data
        else:  # val or test
            self.get_data = _get_val_test_data
