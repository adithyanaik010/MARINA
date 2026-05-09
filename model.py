import torch
import torch.nn as nn
import torch.nn.functional as F

class WeldCNN(nn.Module):

    def __init__(self):
        super(WeldCNN, self).__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # Second convolution layer
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):

        # Conv layer 1
        x = self.pool(F.relu(self.conv1(x)))

        # Conv layer 2
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = F.relu(self.fc1(x))

        # Output layer
        x = self.fc2(x)

        return x
