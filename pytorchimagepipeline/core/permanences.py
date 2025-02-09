from dataclasses import dataclass
from functools import wraps
from logging import warning

import torch
from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from pytorchimagepipeline.abstractions import Permanence


class VRAMUsageError(RuntimeError):
    def __init__(self):
        super().__init__("All devices are using more than 80% of VRAM")


@dataclass
class DeviceWithVRAM:
    device: torch.device
    vram_usage: float


class Device(Permanence):
    """
    Device class for managing CUDA device selection based on VRAM usage.

    This class inherits from the Permanence class and is responsible for calculating
    and setting the best available CUDA device for computation based on VRAM usage.

    Example TOML Config:
        ```toml
        [permanences.network]
        type = "Network"
        params = { model = "deeplabv3_resnet50", num_classes = 21, pretrained = true }
        ```

    Methods:
        __init__() -> None:

        _calculate_best_device() -> None:
            Raises VRAMUsageError if the VRAM usage of the best device exceeds 80%.

        cleanup() -> None:
            Cleans up the device instance. This method doesn't perform any cleanup operations.
    """

    def __init__(self) -> None:
        """
        Initializes the instance and calculates the best device for computation.
        """
        self._calculate_best_device()

    def _calculate_best_device(self) -> None:
        """
        Calculate and set the best available CUDA device based on VRAM usage.

        This method iterates over all available CUDA devices, calculates their VRAM usage,
        and selects the device with the lowest VRAM usage. If the VRAM usage of the best
        device exceeds 80%, a VRAMUsageError is raised.

        Raises:
            VRAMUsageError: If the VRAM usage of the best device exceeds 80%.

        Attributes:
            device (torch.device): The CUDA device with the lowest VRAM usage.
        """
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        vram_usage = [
            DeviceWithVRAM(
                device,
                torch.cuda.memory_reserved(device) / torch.cuda.get_device_properties(device).total_memory,
            )
            for device in devices
        ]

        best_device = min(vram_usage, key=lambda x: x.vram_usage)

        if best_device.vram_usage > 0.8:
            raise VRAMUsageError()

        self.device = best_device.device

    def cleanup(self) -> None:
        """
        Cleans up the device instance.

        This method doesn't perform any cleanup operations for the device instance.

        Returns:
            None: This method doesn't return any value.
        """
        pass


class ProgressManager(Permanence):
    """
    Manages progress tracking for tasks using the Rich library.

    Example TOML Config:
        ```toml
        [permanences.progress_manager]
        type = "ProgressManager"
        param
        ```

    Attributes:
        progress_dict (dict): A dictionary to store progress objects.
        live (Live): A Live object to manage live updates.

    Methods:
        __init__(direct=True):
            Initializes the ProgressManager with an optional direct live update.

        _create_progress(color="#F55500"):
            Creates and returns a Progress object with specified color.

        _init_live():
            Initializes the live update group with the progress objects.

        progress_task(task_name, visible=True):
            A decorator to wrap functions for progress tracking.

        cleanup():
            not used
    """

    def __init__(self, console=None, direct=False):
        """
        Initializes the instance of the class.

        Args:
            direct (bool): If True, initializes live progress. Defaults to False.
        """
        self.console = console
        self.progress_dict = {
            "overall": self._create_progress(),
        }
        if direct:
            self._init_live()

    def _create_progress(self, color="#F55500"):
        """
        Create a progress bar with specified color.

        Args:
            color (str): The color to use for the progress bar. Default is "#F55500".

        Returns:
            Progress: A Progress object configured with the specified color.
        """
        return Progress(
            TextColumn(f"[bold{color}]" + "{task.description}"),
            BarColumn(style="#333333", complete_style=color, finished_style="#22FF55"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=self.console,
        )

    def _init_live(self):
        """
        Initializes the live attribute with a Live object.
        This method creates a Group object using the values from the
        progress_dict attribute and then initializes the live attribute
        with a Live object that takes the Group object as an argument.
        """
        group = Group(*self.progress_dict.values())
        self.live = Live(group, console=self.console)

    def progress_task(self, task_name, visible=True):
        """
        A decorator to add a progress tracking task to a function.

        Args:
            task_name (str): The name of the task to be tracked.
            visible (bool, optional): Whether the task should be visible when done. Defaults to True.

        Returns:
            function: The decorated function with progress tracking.

        The decorated function should have the following signature:
            func(task_id, total, progress, *args, **kwargs)

        The decorator will:
            - Create a progress task if it does not already exist.
            - Add the task to the progress tracker.
            - Call the decorated function with the task_id, total, progress, and any additional arguments.
            - Update the task visibility when done.
        """

        def decorator(func):
            @wraps(func)
            def wrapper(total, *args, **kwargs):
                progress_key = next((key for key in self.progress_dict if task_name.lower() in key), None)
                if progress_key is None:
                    raise NotImplementedError(f"Progress for {task_name} not found")
                progress = self.progress_dict[task_name]
                # Add task to progress
                task_id = progress.add_task(task_name, total=total)

                # Call the function with task_id
                result = func(task_id, total, progress, *args, **kwargs)

                if not progress.finished:
                    warning(UserWarning("Progress not completed, Wrong total provided or advance steps to small"))

                # Hide task when done
                progress.update(task_id, visible=visible)
                return result

            return wrapper

        return decorator

    def cleanup(self):
        pass
