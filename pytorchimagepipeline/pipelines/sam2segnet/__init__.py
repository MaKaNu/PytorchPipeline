from pytorchimagepipeline.pipelines.sam2segnet.permanence import Datasets, MaskCreator
from pytorchimagepipeline.pipelines.sam2segnet.processes import PredictMasks

permanences_to_register = {"Datasets": Datasets, "MaskCreator": MaskCreator}
processes_to_register = {"PredictMasks": PredictMasks}
