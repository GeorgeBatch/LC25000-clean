from torch import nn
from transformers import ViTModel


class OwkinPhikonFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained(
            "owkin/phikon", add_pooling_layer=False)

    def forward(self, x):
        # outputs: [batch_size, 197, 768]:
        #   batch_size images,
        #   197 tokens = 1 cls token + 14*14 patch tokens,
        #   768 features
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 0, :]
        return features
