from dataclasses import dataclass, field
from pathlib import Path

import defusedxml.ElementTree as ET
import torch


class MissingBndBoxError(ValueError):
    def __init__(self):
        super().__init__("Missing <bndbox> element for an object in the XML file.")


class MissingSizeElementError(ValueError):
    def __init__(self):
        super().__init__("Missing <size> element in the XML file.")


@dataclass
class BndBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def to_tensor(self, device=None) -> torch.Tensor:
        """
        Converts the bounding box to a PyTorch tensor.
        Returns:
            torch.Tensor: A tensor of shape (4,) representing [xmin, ymin, xmax, ymax].
        """
        if device:
            return torch.as_tensor([self.xmin, self.ymin, self.xmax, self.ymax], dtype=torch.int64, device=device)
        return torch.as_tensor([self.xmin, self.ymin, self.xmax, self.ymax], dtype=torch.int64)


@dataclass
class ObjectAnnotation:
    name: str
    pose: str
    truncated: int
    difficult: int
    bndbox: BndBox


@dataclass
class ImageAnnotation:
    folder: str
    filename: str
    width: int
    height: int
    depth: int
    objects: list[ObjectAnnotation] = field(default_factory=list)


def parse_voc_xml(xml_file: Path) -> tuple[ImageAnnotation | None, None | Exception]:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Type safety: Check for None explicitly and provide fallbacks
    folder = root.findtext("folder") or ""
    filename = root.findtext("filename") or ""

    # Ensure 'size' exists
    size = root.find("size")
    if size is None:
        return None, MissingSizeElementError()

    # Extract size details safely
    width = int(size.findtext("width") or 0)
    height = int(size.findtext("height") or 0)
    depth = int(size.findtext("depth") or 0)

    # Parse objects safely
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name") or ""
        pose = obj.findtext("pose") or "Unspecified"
        truncated = int(obj.findtext("truncated") or 0)
        difficult = int(obj.findtext("difficult") or 0)

        bndbox = obj.find("bndbox")
        if bndbox is None:
            return None, MissingBndBoxError()

        bbox = BndBox(
            xmin=int(bndbox.findtext("xmin") or 0),
            ymin=int(bndbox.findtext("ymin") or 0),
            xmax=int(bndbox.findtext("xmax") or 0),
            ymax=int(bndbox.findtext("ymax") or 0),
        )
        objects.append(ObjectAnnotation(name, pose, truncated, difficult, bbox))

    return ImageAnnotation(folder, filename, width, height, depth, objects), None


def get_palette(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = torch.float32 if normalized else torch.uint8
    cmap = torch.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        cmap[i] = torch.tensor([r, g, b])
    cmap = cmap / 255 if normalized else cmap
    return cmap.flatten().tolist()
