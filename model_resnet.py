import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class WeldResNet(nn.Module):

    def __init__(self):

        super(WeldResNet, self).__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Replace final classification layer
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            4
        )

    def forward(self, x):
        return self.model(x)
