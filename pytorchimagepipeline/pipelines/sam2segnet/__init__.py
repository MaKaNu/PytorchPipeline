from pytorchimagepipeline.pipelines.sam2segnet.permanence import Datasets
from pytorchimagepipeline.pipelines.sam2segnet.processes import DummyProcess

permanences_to_register = {"Datasets": Datasets}
processes_to_register = {"DummyProcess": DummyProcess}
