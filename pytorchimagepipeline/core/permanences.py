from dataclasses import dataclass

import torch

from pytorchimagepipeline.abstractions import Permanence


class VRAMUsageError(RuntimeError):
    def __init__(self):
        super().__init__("All devices are using more than 80% of VRAM")


@dataclass
class DeviceWithVRAM:
    device: torch.device
    vram_usage: float


class Device(Permanence):
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
