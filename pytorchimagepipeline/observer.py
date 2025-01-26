"""This module defines an Observer responsible for managing a pipeline of processes and handling
potential errors that occur during their execution.

Classes:
    Observer: Manages a pipeline of processes and handles errors that occur during.

Copyright (C) 2025 Matti Kaupenjohann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from typing import Any, Optional

from pytorchimagepipeline.abstractions import AbstractObserver, Permanence, PipelineProcess
from pytorchimagepipeline.errors import (
    BuilderError,
    ErrorCode,
    ExecutionError,
    PermanenceKeyError,
)


class Observer(AbstractObserver):
    def __init__(self, permanences: dict[str, Permanence]):
        """
        Initializes the Observer with the given permanences.

        Args:
            permanences (dict[str, Permanence]): A dictionary mapping string keys to Permanence objects.
        """
        self._permanences = permanences
        self._processes: list[PipelineProcess] = []
        self._current_process: Optional[PipelineProcess] = None

    def add_process(self, process: PipelineProcess) -> None:
        """Adds a process to the pipeline.

        Args:
            process (PipelineProcess): The process to add.
        """
        self._processes.append(process)

    def run(self) -> None:
        """
        Executes each process in the list of processes.

        Iterates over the processes, sets the current process, and executes it.
        If an error occurs during the execution of a process, it handles the error.
        Resets the current process to None after each execution.

        Returns:
            None
        """
        for process in self._processes:
            self._current_process = process
            error = process.execute(self)
            if error:
                self._handle_error(error)
            self._current_process = None

    def _handle_error(self, error: Exception) -> None:
        """
        Handles errors that occur during the execution of a process.

        Args:
            error (Exception): The exception that was raised.

        Raises:
            BuilderError: If the error is an instance of BuilderError.
            ExecutionError: If the error is not an instance of BuilderError,
                            raises an ExecutionError with the current process name and the original error.
        """
        if isinstance(error, BuilderError):
            raise error

        process_name = self._current_process.__class__.__name__
        raise ExecutionError(process_name, error)

    def get_permanences(self, name: str) -> Any:
        """
        Retrieve the permanence value associated with the given name.

        Args:
            name (str): The key name for which to retrieve the permanence value.
        Returns:
            Any: The permanence value associated with the given name.
        Raises:
            PermanenceKeyError: If the given name is not found in the permanences dictionary.
        """

        if name not in self._permanences:
            raise PermanenceKeyError(ErrorCode.PERMA_KEY, key=name)
        return self._permanences[name]
