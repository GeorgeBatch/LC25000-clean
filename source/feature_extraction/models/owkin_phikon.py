import torch
from torch import nn
from transformers import ViTModel, AutoModel


class OwkinPhikonFeatureExtractor(nn.Module):
    """
    OwkinPhikonFeatureExtractor is a feature extractor model based on different versions of the ViTModel.

    Args:
        version (str): The version of the model to use. Options are "v1" and "v2".

    Raises:
        ValueError: If an invalid version is provided.

    Version Details:
        - v1 (owkin/phikon):
            - outputs
                - last_hidden_state [batch_size, 197, 768]
                    - batch_size images
                    - 197 tokens = 1 cls token + 14*14 patch tokens
                    - 768 features
        - v2 (owkin/phikon-v2):
            - outputs
                - pooler_output [batch_size, 1024] - the same as cls token at index 0
                - last_hidden_state [batch_size, 197, 1024]
                    - batch_size images
                    - 197 tokens = 1 cls token + 14*14 patch tokens
                    - 1024 features
    """
    def __init__(self, version):
        super().__init__()
        self.version = version

        if version == "v1":
            self.model = ViTModel.from_pretrained(
                "owkin/phikon", add_pooling_layer=False)
        
        elif version == "v2":
            self.model = AutoModel.from_pretrained("owkin/phikon-v2")
        else:
            raise ValueError(f"Invalid version: {version}")

    def forward(self, x):
        outputs = self.model(x)
        
        cls_token = outputs.last_hidden_state[:, 0, :]
        # patch_tokens = outputs.last_hidden_state[:, 1:, :]
        
        # if self.version == "v2":
        #     assert torch.equal(cls_token, outputs.pooler_output), "Pooler output and cls token should be the same for version={self.version}."
        return cls_token
