"""This module defines abstract base classes for the Pytorch image pipeline.

Classes:
    AbstractObserver: Base class for the Observer class.
    Permanence: Base class for objects that persist through the entire pipeline lifecycle.
    PipelineProcess: Abstract base class for pipeline processes.

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

from abc import ABC, abstractmethod
from typing import Optional


class AbstractObserver(ABC):
    """Base class for the Observer class"""

    @abstractmethod
    def run(self) -> None:
        """Executes the observer's processes.

        This method runs the specific processes defined by the observer implementation.
        The execution details depend on the concrete observer class. #todo: add different observers

        Returns:
            None: This method doesn't return any value
        """
        ...


class Permanence(ABC):
    """Base class for objects that persist through the entire pipeline lifecycle"""

    @abstractmethod
    def cleanup(self) -> Optional[Exception]:
        """Cleans up data from RAM or VRAM.

        Since the objects are permanent, it might be necessary to call a cleanup.
        This will be executed by the observer.

        Returns:
            Optional[Exception]: An exception if an error occurs during cleanup, otherwise None.
        """
        ...


class PipelineProcess(ABC):
    """Abstract base class for pipeline processes"""

    def __init__(self, observer: AbstractObserver, force: bool) -> None:
        """
        Initializes the instance with the given observer.

        When overriding this method, make sure to call the super().__init__(observer, force) method.
        In genereal instead of creating a new instance of the observer, the observer should be passed as an argument.
        The same applies to the force parameter.

        Args:
            observer (AbstractObserver): The observer to be assigned to the instance.
        """

        self.observer = observer
        self.force = force

    @abstractmethod
    def execute(self) -> Optional[Exception]:
        """Executes the process.

        Args:
            observer (AbstractObserver): The observer instance managing the pipeline.

        Returns:
            Optional[Exception]: An exception if an error occurs during execution, otherwise None.
        """
        ...
