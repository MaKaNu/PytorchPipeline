from pytorchimagepipeline.pipelines.sam2segnet.permanence import (
    Datasets,
    HyperParameters,
    MaskCreator,
    Network,
    TrainingComponents,
)
from pytorchimagepipeline.pipelines.sam2segnet.processes import PredictMasks, TrainModel

permanences_to_register = {
    "Datasets": Datasets,
    "Network": Network,
    "MaskCreator": MaskCreator,
    "TrainingComponents": TrainingComponents,
    "HyperParameters": HyperParameters,
}
processes_to_register = {"PredictMasks": PredictMasks, "TrainModel": TrainModel}
