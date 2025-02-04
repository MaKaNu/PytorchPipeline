from unittest.mock import MagicMock, patch

import pytest
import torch

from pytorchimagepipeline.core.permanences import Device, VRAMUsageError


@pytest.fixture
def mock_torch_cuda():
    with (
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.memory_reserved", side_effect=[100, 200]),
        patch(
            "torch.cuda.get_device_properties", side_effect=[MagicMock(total_memory=1000), MagicMock(total_memory=1000)]
        ),
    ):
        yield


def test_calculate_best_device(mock_torch_cuda):
    device = Device()
    assert device.device == torch.device("cuda:0")


def test_calculate_best_device_vram_usage_error(mock_torch_cuda):
    with patch("torch.cuda.memory_reserved", side_effect=[900, 950]), pytest.raises(VRAMUsageError):
        Device()
