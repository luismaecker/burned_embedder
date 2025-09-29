import torch
import torch.nn as nn

class OptimalMLP(nn.Module):
    """Optimized MLP based on experiment results"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.5, use_batch_norm=True):
        super(OptimalMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization before activation
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x).squeeze()
    
class DifferenceMLP(nn.Module):
    """Simple MLP for binary classification on embedding differences"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.4):
        """
        Args:
            input_dim: Dimension of input embeddings
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(DifferenceMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x).squeeze()


def get_model(input_dim, hidden_dims=[256, 128], dropout=0.4, device='cuda'):
    """Create and initialize model"""
    model = DifferenceMLP(input_dim, hidden_dims, dropout)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model