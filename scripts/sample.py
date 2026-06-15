import torch

__all__ = ["ReconModel", "check_input"]


class ReconModel(torch.nn.Module):
    """
    Decode latent tensors back to image space using a trained autoencoder.

    Attributes:
        autoencoder (torch.nn.Module): The trained autoencoder with a decode_stage_2_outputs method.
        scale_factor (float): Scaling factor applied to latents before decoding.
    """

    def __init__(self, autoencoder: torch.nn.Module, scale_factor: float):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode scaled latents to images."""
        return self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)


def check_input(
    body_region,
    anatomy_list,
    label_dict_json,
    output_size,
    spacing,
    controllable_anatomy_size=None,
):
    """
    Basic validation for output size and spacing used in inference.

    Only output_size and spacing are validated; other parameters are accepted
    for backward compatibility.
    """
    controllable_anatomy_size = controllable_anatomy_size or []

    if output_size[0] != output_size[1]:
        raise ValueError(f"The first two components of output_size need to be equal, got {output_size}.")
    if (output_size[0] not in [256, 384, 512]) or (output_size[2] not in [128, 256, 384, 512, 640, 768]):
        raise ValueError(
            f"output_size[0] must be in [256, 384, 512] and output_size[2] in [128, 256, 384, 512, 640, 768], got {output_size}."
        )

    if spacing[0] != spacing[1]:
        raise ValueError(f"The first two components of spacing need to be equal, got {spacing}.")
    if spacing[0] < 0.5 or spacing[0] > 3.0 or spacing[2] < 0.5 or spacing[2] > 5.0:
        raise ValueError(
            f"spacing[0] must be between 0.5 and 3.0 mm, spacing[2] between 0.5 and 5.0 mm, got {spacing}."
        )

    return True
