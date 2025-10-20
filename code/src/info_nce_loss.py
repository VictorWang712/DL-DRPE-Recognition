import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_grey, z_enc):
        # z_grey: (N, D), z_enc: (N, D)
        z_grey = F.normalize(z_grey, dim=1)
        z_enc = F.normalize(z_enc, dim=1)
        logits = torch.matmul(z_grey, z_enc.t()) / self.temperature  # (N, N)
        labels = torch.arange(z_grey.size(0), device=z_grey.device)
        loss_g2e = F.cross_entropy(logits, labels)
        loss_e2g = F.cross_entropy(logits.t(), labels)
        return (loss_g2e + loss_e2g) / 2
