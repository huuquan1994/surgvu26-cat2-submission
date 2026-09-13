"""EfficientNetV2-S classifiers for tool and organ inference."""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s


class SurgToolClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_v2_s(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 12)

    def load_from_ckpt(self, ckpt_path: str, device="cuda"):
        self.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)["state_dict"],
            strict=True,
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))


class OrganClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_v2_s(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 8)

    def load_from_ckpt(self, ckpt_path: str, device="cuda"):
        self.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)["state_dict"],
            strict=True,
        )

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)
