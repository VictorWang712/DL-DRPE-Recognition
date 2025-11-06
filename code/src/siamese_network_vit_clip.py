import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ViTEncoder(nn.Module):  # Define a Vision Transformer encoder module 
    def __init__(self, embedding_dim=256, vit_variant='b_16', pretrained=True):  # Initialize encoder with embedding size, variant, and pretrained flag 
        super().__init__()  # Call parent constructor 
        if vit_variant == 'b_16':  # Check if using base ViT 
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None  # Load pretrained weights if needed 
            model = models.vit_b_16(weights=weights)  # Instantiate ViT base model 
            out_dim = 768  # Set output dimension for base model 
        elif vit_variant == 'l_16':  # Check if using large ViT 
            weights = models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None  # Load pretrained weights if needed 
            model = models.vit_l_16(weights=weights)  # Instantiate ViT large model 
            out_dim = 1024  # Set output dimension for large model 
        else:  # Handle invalid variant 
            raise ValueError("vit_variant must be 'b_16' or 'l_16'")  # Raise error for invalid variant 

        model.conv_proj = nn.Conv2d(  # Replace input projection to accept single-channel input 
            1,  # Set input channels to 1 (grayscale) 
            model.conv_proj.out_channels,  # Keep output channels unchanged 
            kernel_size=model.conv_proj.kernel_size,  # Use original kernel size 
            stride=model.conv_proj.stride,  # Use original stride 
            padding=model.conv_proj.padding,  # Use original padding 
            bias=False  # No bias term 
        )
        model.heads = nn.Identity()  # Remove classification head for feature extraction 
        self.vit = model  # Store modified ViT model 
        self.fc = nn.Linear(out_dim, embedding_dim)  # Add linear layer to map to embedding_dim 

    def forward(self, x):  # Forward pass for encoder 
        x = self.vit(x)  # Pass input through ViT backbone 
        x = self.fc(x)  # Project features to embedding space 
        x = F.normalize(x, dim=1)  # Normalize embeddings along feature dimension 
        return x  # Return normalized embeddings 

class DualViTCLIP(nn.Module):  # Define a dual-branch Siamese network using ViT encoder 
    def __init__(self, embedding_dim=256, vit_variant='b_16', pretrained=True):  # Initialize with embedding size, variant, and pretrained flag 
        super().__init__()  # Call parent constructor 
        self.encoder = ViTEncoder(embedding_dim, vit_variant, pretrained)  # Instantiate shared ViT encoder 

    def forward(self, grey, enc):  # Forward pass for two inputs 
        z_grey = self.encoder(grey)  # Encode the first input (e.g., grayscale image) 
        z_enc = self.encoder(enc)  # Encode the second input (e.g., encrypted image) 
        return z_grey, z_enc  # Return both embeddings 

    def encode_grey(self, x):  # Encode a single grayscale image 
        return self.encoder(x)  # Return embedding for input 

    def encode_enc(self, x):  # Encode a single encrypted image 
        return self.encoder(x)  # Return embedding for input 
