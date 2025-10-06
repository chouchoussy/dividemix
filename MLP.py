"""
MLP model for DivideMix with embedding input
Architecture:
  Block 1: Linear(in_dim → 512) → BatchNorm1d(512) → ReLU → Dropout(0.3)
  Block 2: Linear(512 → 256) → BatchNorm1d(256) → ReLU → Dropout(0.3)
  Output: Linear(256 → num_classes)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self, in_dim, num_classes):
        """
        Args:
            in_dim: dimension of input embedding
            num_classes: number of output classes
        """
        super(MLPNet, self).__init__()
        
        # Block 1: in_dim → 512
        self.fc1 = nn.Linear(in_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.3)
        
        # Block 2: 512 → 256
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.3)
        
        # Output layer: 256 → num_classes
        self.fc_out = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensor of shape (batch_size, in_dim)
        Returns:
            logits of shape (batch_size, num_classes)
        """
        # Block 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Output
        x = self.fc_out(x)
        
        return x


def create_model(in_dim, num_classes):
    """
    Factory function to create MLP model
    Args:
        in_dim: dimension of input embedding
        num_classes: number of output classes
    Returns:
        MLPNet model
    """
    model = MLPNet(in_dim=in_dim, num_classes=num_classes)
    return model
