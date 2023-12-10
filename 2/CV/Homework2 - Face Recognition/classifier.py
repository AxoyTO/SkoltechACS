import gdown
from pathlib import Path
import torch
from torch import nn
import numpy as np
from torchvision import models
from torchvision.models import ResNet50_Weights

class ResNet50Model(nn.Module):
    def __init__(self, num_classes=10572, n_features=512):
        super(ResNet50Model, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(self.model.children())[:-1], nn.Flatten())
        self.n_features = n_features

    def forward(self, x):
        x = self.model(x)
        return x, self.features(x)

class Classifier:
    def __init__(self, num_classes=10572):
        self.model = ResNet50Model(num_classes)

    def load_model(self):
        model_path = Path('/autograder/submission/model5.pth')
        if not model_path.exists():
            gdown.download(url="https://drive.google.com/u/0/uc?id=1Hd0iOJT9bG7vVWQzyJZWY7JhHATsuEX6&export=download",
                           output=str(model_path), quiet=False)
        model_state_dict = torch.load(str(model_path), map_location=torch.device('cpu'))
        self.model.load_state_dict(model_state_dict, strict=False)
        self.model.eval()

    def calculate_descriptors(self, img):
        img = torch.from_numpy(np.float32(img).transpose(2, 0, 1)[None] / 255. * 2 - 1)
        return self.model.features(img).numpy(force=True).ravel()
