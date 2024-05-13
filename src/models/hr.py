import torch
import torchvision
import torch.nn.functional as F

# Model creation
"""
No Finding                   TOP   0    
Enlarged Cardiomediastinum   TOP   1    
Cardiomegaly                       2    
Lung Opacity                 TOP   3    
Lung Lesion                        4    
Edema                              5    
Consolidation                      6    
Pneumonia                          7    
Atelectasis                        8    
Pneumothorax                 TOP   9    
Pleural Effusion             TOP   10   
Pleural Other                TOP   11   
Fracture                     TOP   12   
Support Devices              TOP   13
"""

chexpert_classes = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'Support Devices'
]


class HierarchicalResidual(torch.nn.Module):
    def __init__(self, encoder='resnet18', pretrained=True):
        super().__init__()

        self.encoder_name = encoder
        self.encoder = None
        self.num_ft = 0

        if 'resnet' in encoder:
            resnets = {
                'resnet18': torchvision.models.resnet18,
                'resnet34': torchvision.models.resnet34,
                'resnet50': torchvision.models.resnet50,
                'resnet101': torchvision.models.resnet101,
            }

            self.encoder = resnets[encoder](pretrained=pretrained)
            self.encoder.conv1.weight.data = self.encoder.conv1.weight.data[:, :1]
            self.encoder.conv1.in_channels = 1
            self.num_ft = self.encoder.fc.in_features
            self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1])
        
        elif 'densenet' in encoder:
            self.encoder = torch.hub.load('pytorch/vision:v0.6.0', encoder, pretrained=pretrained)
            self.encoder.features.conv0.weight.data = self.encoder.features.conv0.weight.data[:, :1]
            self.encoder.features.conv0.in_channels = 1
            self.num_ft = self.encoder.classifier.in_features
            self.encoder = torch.nn.Sequential(
                self.encoder.features,
                torch.nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d((1, 1))
            )
           
        else:
            print(f'Unkown encoder {encoder}')
            exit(1)

        """parent classes [
            No Finding, Enlarged Cardiomediastinum, Lung Opacity, 
            Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support devices
        ]
        """
        self.fc1 = torch.nn.Linear(in_features=self.num_ft, out_features=8, bias=True)

        """child classes [
            Cardiomegaly, Lung Lesion, Edema, Consolidation, Pneumonia, Atelactasis
        ]
        """
        self.fc2 = torch.nn.Linear(in_features=self.num_ft+8, out_features=6, bias=True)
        
        # Sort output with correct label order
        output_order = torch.tensor([2, 4, 5, 6, 7, 8, 0, 1, 3, 9, 10, 11, 12, 13])
        self.out_idx = torch.argsort(output_order)
    
    def forward(self, x):
        x = self.encoder(x)

        x = torch.flatten(x, 1)

        parent = self.fc1(x)
        x = torch.cat((x, parent), dim=1)
        x = F.relu(x)
        x = self.fc2(x)

        x = torch.cat((x, parent), dim=1)
        return x[:, self.out_idx]
    

class HierarchicalResidualFT(HierarchicalResidual):
    """ For finetuning"""
    def forward(self, x):
        feats = self.encoder(x)
        feats = torch.flatten(feats, 1)
        x = self.fc1(feats)
        x = F.relu(x)
        x = self.fc2(x)
        return x, feats
    
class HierarchicalResidualFTNorm(HierarchicalResidual):
    """ For finetuning"""
    def forward(self, x):
        feats = self.encoder(x)
        feats = torch.flatten(feats, 1)
        feats = F.normalize(feats, dim=1)
        x = self.fc1(feats)
        x = F.relu(x)
        x = self.fc2(x)
        return x, feats
    

class HierarchicalResidualFTNorm2(HierarchicalResidual):
    """ For finetuning"""
    def forward(self, x):
        feats = self.encoder(x)
        feats = torch.flatten(feats, 1)
        feats = self.fc1(feats)
        feats = F.normalize(feats, dim=1)
        x = self.fc2(feats)
        return x, feats