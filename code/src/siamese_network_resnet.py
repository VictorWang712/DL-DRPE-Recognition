import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SiameseNetworkResNet(nn.Module):
    def __init__(self, embedding_dim=256):
        super(SiameseNetworkResNet, self).__init__()
        from torchvision.models import ResNet34_Weights
        # 推荐用 weights 参数
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature = nn.Sequential(*list(resnet.children())[:-1])  # (batch, 512, 1, 1)
        self.fc = nn.Linear(512, embedding_dim)

    def forward_once(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2
