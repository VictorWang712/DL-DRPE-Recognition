import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SiameseNetworkResNet(nn.Module):
    def __init__(self, embedding_dim=256): # Initialize the Siamese Network with embedding dimension
        super(SiameseNetworkResNet, self).__init__() # Call the parent class constructor
        from torchvision.models import ResNet34_Weights # Import weights for ResNet34
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT) # Load pre-trained ResNet34 model
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # Change input channels to 1 for grayscale
        self.feature = nn.Sequential(*list(resnet.children())[:-1]) # Extract all layers except the last FC layer
        self.fc = nn.Linear(512, embedding_dim) # Linear layer to get the desired embedding dimension

    def forward_once(self, x): # Forward pass for a single input
        x = self.feature(x) # Pass input through feature extractor
        x = x.view(x.size(0), -1) # Flatten the output to (batch, 512)
        x = self.fc(x) # Pass through the fully connected layer
        x = F.normalize(x, dim=1) # Normalize the embedding vector
        return x # Return the normalized embedding

    def forward(self, input1, input2): # Forward pass for two inputs (Siamese)
        out1 = self.forward_once(input1) # Get embedding for the first input
        out2 = self.forward_once(input2) # Get embedding for the second input
        return out1, out2 # Return both embeddings as a tuple
