class MaskNotAvailable(Exception):
    def __init__(self):
        super().__init__("Masks are not set by user. Please set the masks using set_current_masks method.")


class MaskShapeError(Exception):
    def __init__(self, shape):
        super().__init__(f"The masks should be a 4D tensor with shape (N, C, H, W). Got shape: {shape}")


class ModelNotSupportedError(Exception):
    def __init__(self, model_name, supported_models):
        super().__init__(
            f"Model {model_name} is not supported. Please choose one of the supported models: {supported_models}"
        )


class FormatNotSupportedError(Exception):
    def __init__(self, format_name, supported_formats):
        super().__init(
            f"Format {format_name} is not supported. Please choose one of the supported formats: {supported_formats}"
        )
