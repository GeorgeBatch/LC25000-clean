from torch import nn
from transformers import AutoModel


class HibouFeatureExtractor(nn.Module):
    """
    HibouFeatureExtractor is a feature extractor model based on different versions of the ViTModel.

    Args:
        version (str): The version of the model to use. Options are "b" and "L".

    Raises:
        ValueError: If an invalid version is provided.

    Version Details:
        - b (histai/hibou-b):
            - outputs:
                - pooler_output [batch_size, 768] - the same as cls token at index 0
                - last_hidden_state [batch_size, 261, 768]
                    - batch_size images
                    - 261 tokens = 1 cls token + 4 DINOv2 register tokens + 16*16 patch tokens
                    - 768 features
        - L (histai/hibou-L):
            - outputs:
                - pooler_output [batch_size, 1024] - the same as cls token at index 0
                - last_hidden_state [batch_size, 261, 1024]
                    - batch_size images
                    - 261 tokens = 1 cls token + 4 DINOv2 register tokens + 16*16 patch tokens
                    - 1024 features
    """
    def __init__(self, version):
        super().__init__()

        if version == "b":
            self.model = AutoModel.from_pretrained("histai/hibou-b", trust_remote_code=True)
        elif version == "L":
            self.model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
        else:
            raise ValueError(f"Invalid version: {version}")

    def forward(self, x):
        outputs = self.model(x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        # register_tokens = outputs.last_hidden_state[:, 1:5, :]
        # patch_tokens = outputs.last_hidden_state[:, 5:, :]

        # assert torch.equal(cls_token, outputs.pooler_output), "Pooler output and cls token should be the same."
        return cls_token
