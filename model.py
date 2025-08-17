import torch.nn as nn
from torchvision import models

class FishClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FishClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
