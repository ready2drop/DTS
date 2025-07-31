from typing import Any, Tuple

from .. import DTS

class ModelHub:
    def __init__(self) -> None:
        pass

    def __call__(self, model_name: str, **kwargs: Any) -> Any:
        if model_name == "DTS":
            model = DTS(
                image_size=self.parse_image_size(**kwargs),
                in_channels=kwargs['in_channels'],
                out_channels=kwargs['out_channels'],
                feature_size=48,
                extract_features=kwargs['extract_features'],
                freeze=kwargs['freeze'],
            )
        else:
            raise ValueError(f"Invalid model type: {model_name}")

        return model

    def parse_image_size(self, **kwargs) -> Tuple[int, int, int]:
        return (kwargs['spatial_size'], kwargs['image_size'], kwargs['image_size'])