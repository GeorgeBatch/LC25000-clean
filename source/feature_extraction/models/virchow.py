import torch
from torch import nn

import timm
from timm.layers import SwiGLUPacked
# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform


class VirchowFeatureExtractor(nn.Module):
    """
    VirchowFeatureExtractor is a feature extractor model based on different versions of the ViTModel.

    Args:
        version (str): The version of the model to use. Options are "v1" and "v2".

    Raises:
        ValueError: If an invalid version is provided.

    Version Details:
        - v1 (paige-ai/Virchow):
            - output tensor [batch_size, 257, 1280]
                - batch_size images
                - 257 tokens = 1 cls token + 16*16 patch tokens
                - 1280 features
        - v2 (paige-ai/Virchow2):
            - output tensor [batch_size, 261, 1280]
                - batch_size images
                - 261 tokens = 1 cls token + 4 DINOv2 register tokens + 16*16 patch tokens
                - 1280 features
    """
    def __init__(self, version):
        super().__init__()
        self.version = version

        if version == "v1":
            self.model = timm.create_model(
                "hf-hub:paige-ai/Virchow",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        elif version == "v2":
            self.model = timm.create_model(
                "hf-hub:paige-ai/Virchow2",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        else:
            raise ValueError(f"Invalid version: {version}")

    def forward(self, x):
        output = self.model(x)
        cls_token = output[:, 0, :]

        # if self.version == "v1":
        #     patch_tokens = output[:, 1:, :]
        # elif self.version == "v2":
        #     register_tokens = output[:, 1:5, :]
        #     patch_tokens = output[:, 5:, :]

        # embedding = torch.cat([cls_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        # return embedding

        return cls_token
