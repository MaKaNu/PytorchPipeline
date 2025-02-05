from pathlib import Path

import torch
import torchvision
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from pytorchimagepipeline.abstractions import PipelineProcess
from pytorchimagepipeline.pipelines.sam2segnet.utils import get_palette


class PredictMasks(PipelineProcess):
    def __init__(self, observer: AbstractObserver, force: bool) -> None:
        super().__init__(observer, force)
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

    def skip(self):
        return torch.tensor(list(map(Path.exists, self.dataset.images)), dtype=bool).all() and not self.force


class TrainModel(PipelineProcess):
    def __init__(self, observer, force):
        super().__init__(observer, force)

    def skip(self):
        return False

    def execute(self):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader)}")

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_loader)}")

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

        print(f"Test Loss: {test_loss/len(test_loader)}")