import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ViTEncoder(nn.Module):
    def __init__(self, embedding_dim=256, vit_variant='b_16', pretrained=True):
        super().__init__()
        # torchvision ViT
        if vit_variant == 'b_16':
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.vit_b_16(weights=weights)
            out_dim = 768
        elif vit_variant == 'l_16':
            weights = models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.vit_l_16(weights=weights)
            out_dim = 1024
        else:
            raise ValueError("vit_variant must be 'b_16' or 'l_16'")

        # 修改输入通道为1
        model.conv_proj = nn.Conv2d(
            1,
            model.conv_proj.out_channels,
            kernel_size=model.conv_proj.kernel_size,
            stride=model.conv_proj.stride,
            padding=model.conv_proj.padding,
            bias=False
        )
        # 去掉分类头，输出为 [B, embed_dim]
        model.heads = nn.Identity()
        self.vit = model
        self.fc = nn.Linear(out_dim, embedding_dim)

    def forward(self, x):
        x = self.vit(x)  # 输出为 [B, embed_dim]
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

class DualViTCLIP(nn.Module):
    """
    共享编码器结构：原图和加密图都用同一个ViTEncoder
    """
    def __init__(self, embedding_dim=256, vit_variant='b_16', pretrained=True):
        super().__init__()
        self.encoder = ViTEncoder(embedding_dim, vit_variant, pretrained)

    def forward(self, grey, enc):
        z_grey = self.encoder(grey)
        z_enc = self.encoder(enc)
        return z_grey, z_enc

    def encode_grey(self, x):
        return self.encoder(x)

    def encode_enc(self, x):
        return self.encoder(x)
