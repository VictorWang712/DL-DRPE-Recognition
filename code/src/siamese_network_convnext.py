import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SiameseNetworkConvNeXt(nn.Module): # Define a Siamese network using ConvNeXt backbone 
    def __init__(self, embedding_dim=256, convnext_variant='base', pretrained=True): # Initialize with embedding dimension, variant, and pretrained flag 
        super(SiameseNetworkConvNeXt, self).__init__() # Call parent constructor 

        if convnext_variant == 'tiny': # Check if using tiny variant 
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None # Set weights for pretrained or not 
            model = models.convnext_tiny(weights=weights) # Load ConvNeXt tiny model 
            out_dim = 768 # Output dimension for tiny variant 
        elif convnext_variant == 'small': # Check if using small variant 
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None # Set weights for pretrained or not 
            model = models.convnext_small(weights=weights) # Load ConvNeXt small model 
            out_dim = 768 # Output dimension for small variant 
        elif convnext_variant == 'base': # Check if using base variant 
            weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None # Set weights for pretrained or not 
            model = models.convnext_base(weights=weights) # Load ConvNeXt base model 
            out_dim = 1024 # Output dimension for base variant 
        elif convnext_variant == 'large': # Check if using large variant 
            weights = models.ConvNeXt_Large_Weights.DEFAULT if pretrained else None # Set weights for pretrained or not 
            model = models.convnext_large(weights=weights) # Load ConvNeXt large model 
            out_dim = 1536 # Output dimension for large variant 
        else: # If variant is not recognized 
            raise ValueError("convnext_variant must be one of ['tiny', 'small', 'base', 'large']") # Raise error for invalid variant 

        model.features[0][0] = nn.Conv2d( # Replace first convolution to accept single-channel input 
            1, # Set input channels to 1 for grayscale images 
            model.features[0][0].out_channels, # Keep output channels unchanged 
            kernel_size=model.features[0][0].kernel_size, # Use original kernel size 
            stride=model.features[0][0].stride, # Use original stride 
            padding=model.features[0][0].padding, # Use original padding 
            bias=False # No bias for convolution 
        )

        self.feature = model.features # Store feature extractor part of model 
        self.avgpool = model.avgpool # Store average pooling layer 
        self.fc = nn.Linear(out_dim, embedding_dim) # Linear layer to project to embedding dimension 

    def forward_once(self, x): # Forward pass for a single input 
        x = self.feature(x) # Extract features from input 
        x = self.avgpool(x) # Apply average pooling to features 
        x = torch.flatten(x, 1) # Flatten the output except batch dimension 
        x = self.fc(x) # Project features to embedding space 
        x = F.normalize(x, dim=1) # Normalize embeddings along feature dimension 
        return x # Return normalized embedding 

    def forward(self, input1, input2): # Forward pass for a pair of inputs 
        out1 = self.forward_once(input1) # Get embedding for first input 
        out2 = self.forward_once(input2) # Get embedding for second input 
        return out1, out2 # Return both embeddings as output 
