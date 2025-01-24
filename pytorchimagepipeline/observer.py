from typing import Any, Optional

from pytorchimagepipeline.abstractions import AbstractObserver, Permanence, PipelineProcess
from pytorchimagepipeline.errors import (
    BuilderError,
    ErrorCode,
    ExecutionError,
    InstTypeError,
    PermanenceKeyError,
)


class Observer(AbstractObserver):
    def __init__(self, permanent_objects: dict[str, Permanence]):
        self._permanent_objects = permanent_objects
        self._processes: list[PipelineProcess] = []
        self._current_process: Optional[PipelineProcess] = None

    def add_process(self, process: PipelineProcess) -> Optional[Exception]:
        """Add a process to the pipeline"""
        if not isinstance(process, PipelineProcess):
            return InstTypeError(process)
        self._processes.append(process)

    def run(self) -> None:
        """Execute the pipeline with error handling"""
        for process in self._processes:
            self._current_process = process
            error = process.execute(self)
            if error:
                self._handle_error(error)
            self._current_process = None

    def _handle_error(self, error: Exception) -> None:
        """Handle execution errors"""
        if isinstance(error, BuilderError):
            raise error

        process_name = self._current_process.__class__.__name__
        raise ExecutionError(process_name, error)

    def get_permanent_object(self, name: str) -> Any:
        """Retrieve permanent object with validation"""
        if name not in self._permanent_objects:
            raise PermanenceKeyError(ErrorCode.PERMA_KEY, key=name)
        return self._permanent_objects[name]
