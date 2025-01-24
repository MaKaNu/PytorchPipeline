from abc import ABC, abstractmethod
from typing import Optional


class AbstractObserver(ABC):
    """Base class for the Observer class"""

    @abstractmethod
    def run(self) -> None: ...


class Permanence(ABC):
    """Base class for objects that persist through the entire pipeline lifecycle"""

    @abstractmethod
    def cleanup(self) -> Optional[Exception]:
        """Cleanup data from ram or vram

        Since the objects are permanent it might be necessary to call a cleanup.
        Will be executed by the observer.
        """
        ...


class PipelineProcess(ABC):
    """Abstract base class for pipeline processes"""

    @abstractmethod
    def execute(self, observer: AbstractObserver) -> Optional[Exception]:
        """Execute the process with access to observer and long-living objects"""
        ...
