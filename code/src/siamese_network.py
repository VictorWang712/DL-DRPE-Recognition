import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2), nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16, 512), nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward_once(self, x):
        x = self.conv(x)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2
