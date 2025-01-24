from pytorchimagepipeline.abstractions import PipelineProcess


class DummyProcess(PipelineProcess):
    def execute(self, observer):
        print(observer)
