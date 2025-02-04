from pathlib import Path

import torch
import torchvision
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from pytorchimagepipeline.abstractions import PipelineProcess
from pytorchimagepipeline.pipelines.sam2segnet.utils import get_palette


class PredictMasks(PipelineProcess):
    def __init__(self):
        sam_checkpoint = Path("data/models/sam_vit_h_4b8939.pth")
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    def execute(self, observer):
        device = observer.get_permanence("device").device
        self.sam.to(device)
        predictor = SamPredictor(self.sam)

        dataset = observer.get_permanence("data").sam_dataset

        bar = tqdm(len(dataset), desc="Predicting masks")
        for data in dataset:
            image, bboxes, bbox_classes, filestem = data
            bbox_classes_idx = torch.tensor(
                [observer.get_permanence("data").sam_dataset.class_idx[class_] for class_ in bbox_classes],
                dtype=torch.float32,
                device=device,
            )

            predictor.set_image(image)

            bboxes_tensor = torch.stack([bbox.to_tensor(device=predictor.device) for bbox in bboxes])
            transformed_boxes = predictor.transform.apply_boxes_torch(bboxes_tensor, image.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            masks_with_classes = masks * bbox_classes_idx.view(-1, 1, 1, 1)
            merged_mask = observer.get_permanence("mask_creator").create_mask(masks_with_classes)

            palette = get_palette()
            mask_as_pil = torchvision.transforms.functional.to_pil_image(merged_mask)
            mask_as_pil.putpalette(palette)
            mask_path = observer.get_permanence("data").sam_dataset.target_location / f"{filestem}.png"
            mask_as_pil.save(mask_path)

            bar.update(1)
            bar.refresh()
