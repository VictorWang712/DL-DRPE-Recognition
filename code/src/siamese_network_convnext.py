import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SiameseNetworkConvNeXt(nn.Module):
    def __init__(self, embedding_dim=256, convnext_variant='base', pretrained=True):
        super(SiameseNetworkConvNeXt, self).__init__()

        # 选择不同的 ConvNeXt 变体（使用新版 weights 参数）
        if convnext_variant == 'tiny':
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = models.convnext_tiny(weights=weights)
            out_dim = 768
        elif convnext_variant == 'small':
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            model = models.convnext_small(weights=weights)
            out_dim = 768
        elif convnext_variant == 'base':
            weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            model = models.convnext_base(weights=weights)
            out_dim = 1024
        elif convnext_variant == 'large':
            weights = models.ConvNeXt_Large_Weights.DEFAULT if pretrained else None
            model = models.convnext_large(weights=weights)
            out_dim = 1536
        else:
            raise ValueError("convnext_variant must be one of ['tiny', 'small', 'base', 'large']")

        # 修改输入通道为1（灰度图）
        model.features[0][0] = nn.Conv2d(
            1,
            model.features[0][0].out_channels,
            kernel_size=model.features[0][0].kernel_size,
            stride=model.features[0][0].stride,
            padding=model.features[0][0].padding,
            bias=False
        )

        # 提取特征部分
        self.feature = model.features
        self.avgpool = model.avgpool
        self.fc = nn.Linear(out_dim, embedding_dim)

    def forward_once(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2
