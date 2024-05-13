import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

model_dict = {
    # Imagenet variants
    'resnet18': [torchvision.models.resnet18, 512],
    'resnet34': [torchvision.models.resnet34, 512],
    'resnet50': [torchvision.models.resnet50, 2048],
    'resnet101': [torchvision.models.resnet101, 2048],
}

class ResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=2, normalize=False):
        super().__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.encoder.conv1.weight.data = self.encoder.conv1.weight.data[:, :1]
        self.encoder.conv1.in_channels = 1
        
        if hasattr(self.encoder, 'fc'):
            self.encoder.fc = nn.Identity()
        self.fc = nn.Linear(dim_in, num_classes)
        self.normalize = normalize

    def forward(self, x):
        feats = self.encoder(x)
        if self.normalize:
            feats = F.normalize(feats, dim=1)
        return self.fc(feats), feats