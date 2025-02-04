import logging

import pytest
import torch

from pytorchimagepipeline.pipelines.sam2segnet.permanence import MaskCreator, MaskNotAvailable, MaskShapeError


@pytest.fixture
def mask_creator():
    return MaskCreator(morph_size=1, border_size=1, ignore_value=255)


@pytest.fixture
def masks(request):
    if request.param == "small":
        masks = torch.zeros((4, 1, 4, 7), dtype=torch.float32)
        masks[0, 0, 3, 6] = 7
        masks[1, 0, 0:2, 1:3] = 5
        masks[2, 0, 1:3, 4:6] = 3
        masks[3, 0, 1:, :2] = 9
        return masks
    elif request.param == "large":
        masks = torch.zeros((4, 1, 40, 70), dtype=torch.float32)
        masks[0, 0, 30:, 60:] = 7
        masks[1, 0, :10, 20:30] = 5
        masks[2, 0, 10:30, 40:60] = 3
        masks[3, 0, 10:, :20] = 9
        return masks


@pytest.mark.parametrize("masks", ["small", "large"], indirect=True)
def test_create_mask_shape(masks, mask_creator):
    expected_shape = masks.shape[1:]
    result = mask_creator.create_mask(masks)
    assert result.shape == expected_shape, "The shape of the result mask is incorrect."


@pytest.mark.parametrize("masks", ["small", "large"], indirect=True)
def test_create_mask_values(masks, mask_creator, request):
    result = mask_creator.create_mask(masks)
    assert torch.all(result >= 0) and torch.all(result <= 255), (
        "The values in the result mask are out of expected range."
    )
    assert result.dtype == torch.uint8, "The dtype of the result mask is incorrect."
    scale = 200 / result.shape[1]
    mask_img_0 = torch.nn.functional.interpolate(
        masks[0].view(1, *result.shape).type(torch.uint8), scale_factor=scale
    ).squeeze()
    mask_img_1 = torch.nn.functional.interpolate(
        masks[1].view(1, *result.shape).type(torch.uint8), scale_factor=scale
    ).squeeze()
    mask_img_2 = torch.nn.functional.interpolate(
        masks[2].view(1, *result.shape).type(torch.uint8), scale_factor=scale
    ).squeeze()
    mask_img_3 = torch.nn.functional.interpolate(
        masks[3].view(1, *result.shape).type(torch.uint8), scale_factor=scale
    ).squeeze()
    result_img = torch.nn.functional.interpolate(result.view(1, *result.shape), scale_factor=scale).squeeze()
    request.node.user_properties.append(("image_tensor", "Mask 0", mask_img_0))
    request.node.user_properties.append(("image_tensor", "Mask 1", mask_img_1))
    request.node.user_properties.append(("image_tensor", "Mask 2", mask_img_2))
    request.node.user_properties.append(("image_tensor", "Mask 3", mask_img_3))
    request.node.user_properties.append(("image_tensor", "Combined Mask", result_img))


def test_check_masks_none(mask_creator):
    with pytest.raises(MaskNotAvailable):
        mask_creator._check_masks()


def test_check_masks_invalid_shape(mask_creator):
    mask_creator.set_current_masks(torch.zeros(3, 224, 224))  # 3D tensor instead of 4D
    with pytest.raises(MaskShapeError):
        mask_creator._check_masks()


def test_check_masks_invalid_dtype(mask_creator, caplog):
    mask_creator.set_current_masks(torch.zeros(1, 3, 224, 224, dtype=torch.uint8))  # int32 instead of float
    with caplog.at_level(logging.WARNING):
        mask_creator._check_masks()

    assert "Masks are not in float32 format. Converting to float32." in caplog.text
    assert mask_creator.current_masks.dtype == torch.float32


def test_check_masks_valid(mask_creator):
    mask_creator.set_current_masks(torch.zeros(1, 3, 224, 224, dtype=torch.float32))
    mask_creator._check_masks()  # Should not raise any exceptions


@pytest.mark.parametrize(
    "masks, kernel_size, padding, expected",
    [
        (
            torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=torch.float32),
            3,
            1,
            torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32),
        ),
        (
            torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=torch.float32),
            3,
            0,
            torch.tensor([[[[0]]]], dtype=torch.float32),
        ),
        (
            torch.tensor(
                [[[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]]],
                dtype=torch.float32,
            ),
            3,
            1,
            torch.tensor(
                [[[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
                dtype=torch.float32,
            ),
        ),
        (
            torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=torch.float32),
            5,
            2,
            torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32),
        ),
        (
            torch.zeros((1, 1, 3, 3), dtype=torch.float32),
            3,
            1,
            torch.zeros((1, 1, 3, 3), dtype=torch.float32),
        ),
        (
            torch.ones((1, 1, 3, 3), dtype=torch.float32),
            3,
            1,
            torch.ones((1, 1, 3, 3), dtype=torch.float32),
        ),
    ],
    ids=[
        "3x3_input_3x3_kernel_with_padding",
        "3x3_input_3x3_kernel_no_padding",
        "5x5_input_3x3_kernel_with_padding",
        "3x3_input_5x5_kernel_with_padding",
        "3x3_kernel_zeros",
        "3x3_kernel_ones",
    ],
)
def test_erode(mask_creator, masks, kernel_size, padding, expected, request):
    mask_creator.set_current_masks(masks)
    eroded = mask_creator._erode(kernel_size=kernel_size, padding=padding)
    assert torch.equal(eroded, expected), f"Expected {expected}, but got {eroded}"
    scale = 200 / masks.shape[2]
    mask_img = torch.nn.functional.interpolate(masks.type(torch.uint8), scale_factor=scale).squeeze()
    result_img = torch.nn.functional.interpolate(eroded.type(torch.uint8), scale_factor=scale).squeeze()
    request.node.user_properties.append(("image_tensor", "Mask", mask_img))
    request.node.user_properties.append(("image_tensor", "Result", result_img))


@pytest.mark.parametrize(
    "masks, kernel_size, padding, expected",
    [
        (
            torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32),
            3,
            1,
            torch.tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], dtype=torch.float32),
        ),
        (
            torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32),
            3,
            0,
            torch.tensor([[[[1]]]], dtype=torch.float32),
        ),
        (
            torch.tensor(
                [[[[0, 0, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0]]]],
                dtype=torch.float32,
            ),
            3,
            1,
            torch.tensor(
                [[[[0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1]]]],
                dtype=torch.float32,
            ),
        ),
        (
            torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32),
            5,
            2,
            torch.tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], dtype=torch.float32),
        ),
        (
            torch.zeros((1, 1, 3, 3), dtype=torch.float32),
            3,
            1,
            torch.zeros((1, 1, 3, 3), dtype=torch.float32),
        ),
        (
            torch.ones((1, 1, 3, 3), dtype=torch.float32),
            3,
            1,
            torch.ones((1, 1, 3, 3), dtype=torch.float32),
        ),
    ],
    ids=[
        "3x3_input_3x3_kernel_with_padding",
        "3x3_input_3x3_kernel_no_padding",
        "5x5_input_3x3_kernel_with_padding",
        "3x3_input_5x5_kernel_with_padding",
        "3x3_kernel_zeros",
        "3x3_kernel_ones",
    ],
)
def test_dilate(mask_creator, masks, kernel_size, padding, expected, request):
    mask_creator.set_current_masks(masks)
    dilated = mask_creator._dilate(kernel_size=kernel_size, padding=padding)
    assert torch.equal(dilated, expected), f"Expected {expected}, but got {dilated}"
    scale = 200 / masks.shape[2]
    mask_img = torch.nn.functional.interpolate(masks.type(torch.uint8), scale_factor=scale).squeeze()
    result_img = torch.nn.functional.interpolate(dilated.type(torch.uint8), scale_factor=scale).squeeze()
    request.node.user_properties.append(("image_tensor", "Mask", mask_img))
    request.node.user_properties.append(("image_tensor", "Result", result_img))
