from pathlib import Path

import torch
import torchvision
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from pytorchimagepipeline.abstractions import AbstractObserver, PipelineProcess
from pytorchimagepipeline.pipelines.sam2segnet.utils import get_palette


class PredictMasks(PipelineProcess):
    def __init__(self, observer: AbstractObserver, force: bool) -> None:
        super().__init__(observer, force)
        sam_checkpoint = Path("data/models/sam_vit_h_4b8939.pth")
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.device = observer.get_permanence("device").device
        self.dataset = observer.get_permanence("data").sam_dataset
        self.mask_creator = observer.get_permanence("mask_creator")

    def execute(self):
        self.sam.to(self.device)
        predictor = SamPredictor(self.sam)

        bar = tqdm(len(self.dataset), desc="Predicting masks")
        for data in self.dataset:
            image, bboxes, bbox_classes, filestem = data
            bbox_classes_idx = torch.tensor(
                [self.dataset.class_idx[class_] for class_ in bbox_classes],
                dtype=torch.float32,
                device=self.device,
            )

            predictor.set_image(image)

            if bboxes:
                bboxes_tensor = torch.stack([bbox.to_tensor(device=predictor.device) for bbox in bboxes])
                transformed_boxes = predictor.transform.apply_boxes_torch(bboxes_tensor, image.shape[:2])
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )

                masks_with_classes = masks * bbox_classes_idx.view(-1, 1, 1, 1)
                merged_mask = self.mask_creator.create_mask(masks_with_classes)
            else:
                merged_mask = torch.zeros(image.shape[:2], dtype=torch.uint8, device=self.device)

            palette = get_palette()
            mask_as_pil = torchvision.transforms.functional.to_pil_image(merged_mask)
            mask_as_pil.putpalette(palette)
            mask_path = self.dataset.target_location / f"{filestem}.png"
            mask_as_pil.save(mask_path)

            bar.update(1)
            bar.refresh()

    def skip(self):
        return self.dataset.all_created() and not self.force


class TrainModel(PipelineProcess):
    def __init__(self, observer, force):
        super().__init__(observer, force)
        self.device = observer.get_permanence("device").device
        self.model = observer.get_permanence("network").model_instance
        self.model.to(self.device)

        # Hyperparameters
        observer.get_permanence("hyperparams").calculate_batch_size(self.device)
        self.hyperparams = observer.get_permanence("hyperparams").hyperparams

        # Data
        self.datasets = observer.get_permanence("data")
        batch_size = self.hyperparams.get("batch_size", 20)
        trainset = self.datasets.segnet_dataset_train
        valset = self.datasets.segnet_dataset_val
        testset = self.datasets.segnet_dataset_test

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        # Training components
        components = observer.get_permanence("training_components")
        ignore_index = self.datasets.data_container.ignore
        self.criterion = components.Criterion(**self.hyperparams.get("criterion", {}), ignore_index=ignore_index)
        self.optimizer = components.Optimizer(self.model.parameters(), **self.hyperparams.get("optimizer", {}))
        self.scheduler = components.Scheduler(
            self.optimizer, **self.hyperparams.get("scheduler", components.scheduler_params)
        )
        self.num_epochs = self.hyperparams.get("num_epochs", 20)

    def skip(self):
        return False

    def execute(self):
        for epoch in range(self.num_epochs):
            running_loss = self._train_step()
            train_msg = f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {running_loss / len(self.train_loader)}"

            if self.datasets.val_available():
                val_loss = self._validate_step(epoch)
                val_msg = f"Epoch {epoch + 1}/{self.num_epochs}, Validation Loss: {val_loss / len(self.val_loader)}"

        if self.datasets.test_available():
            test_loss = self._test_step()
            test_msg = f"Test Loss: {test_loss / len(self.test_loader)}"

    def _train_step(self):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(inputs)
            main_pred, aux_pred = output.get("out"), output.get("aux", None)
            main_loss = self.criterion(main_pred, labels)
            aux_loss = self.criterion(aux_pred, labels) if aux_pred is not None else torch.zeros_like(main_loss)
            loss = main_loss + self.hyperparams.get("aux_lambda", 0.4) * aux_loss
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        self.scheduler.step()

        return running_loss

    def _validate_step(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
        return val_loss

    def _test_step(self):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
        return test_loss
