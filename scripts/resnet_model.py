import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet_Model(nn.Module):
    """
    Encoder, that outputs 768 dim embedding(normalized)... 
    """
    def __init__(self, name, in_channels=3):
        super(ResNet_Model, self).__init__()
        self.model = models.__dict__[name.lower()]()
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.flatten = nn.Flatten()
        print(f'Using {name}, In Dim: {self.model.fc.in_features}')
        self.fc = nn.Linear(self.model.fc.in_features, 768) # 768 is embedding dimension of CLIP token emebddings
    
    def forward(self,x, normalize = True):
        out = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = self.model.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        if normalize:
            out = F.normalize(out, p=2, dim=1)

        return out
    