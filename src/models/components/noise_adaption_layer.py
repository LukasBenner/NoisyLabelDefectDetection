"""Noise Adaption Layer for handling noisy labels.

The Noise Adaption Layer learns a confusion matrix that models the label noise,
allowing the model to adapt its predictions to noisy training data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseAdaptionLayer(nn.Module):
    """
    Noise Adaption Layer that learns label noise transition matrix.
    
    This layer learns a confusion matrix T where T[i,j] represents the probability
    that a sample with true label i is labeled as j. The layer is applied after
    the main network to adapt predictions for noisy training data.
    
    Args:
        num_classes: Number of classes in the classification task
        init_value: Initial value for the transition matrix diagonal (default: 0.9)
        trainable: Whether the transition matrix should be trainable (default: True)
    """
    
    def __init__(
        self,
        num_classes: int,
        init_value: float = 0.9
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Initialize transition matrix
        # Start with identity-like matrix (assuming low noise initially)
        init_matrix = torch.eye(num_classes) * init_value
        init_matrix += (1.0 - init_value) / (num_classes - 1) * (1 - torch.eye(num_classes))
        
        self.transition_matrix = nn.Parameter(init_matrix)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply noise adaption layer to logits.
        
        Args:
            logits: Input logits from the base network [batch_size, num_classes]
            
        Returns:
            Adapted logits [batch_size, num_classes]
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Ensure transition matrix is a valid probability distribution
        # Normalize each row to sum to 1
        transition_matrix = F.softmax(self.transition_matrix, dim=1)
        
        # Apply noise transition: p_noisy = p_clean @ T^T
        adapted_probs = torch.matmul(probs, transition_matrix.t())
        
        # Convert back to logits for loss computation
        adapted_probs = torch.clamp(adapted_probs, min=1e-7, max=1.0)
        adapted_logits = torch.log(adapted_probs)
        
        return adapted_logits
    
    def get_transition_matrix(self) -> torch.Tensor:
        """Get the normalized transition matrix."""
        with torch.no_grad():
            return F.softmax(self.transition_matrix, dim=1)
    
    def extra_repr(self) -> str:
        """Extra information for print statement."""
        return f'num_classes={self.num_classes}'


class NoiseAdaptionNet(nn.Module):
    """
    Wrapper that adds a noise adaption layer to any base network.
    
    Args:
        base_net: The base classification network
        num_classes: Number of classes
        init_value: Initial value for noise transition matrix diagonal
        trainable: Whether the transition matrix should be trainable
    """
    
    def __init__(
        self,
        base_net: nn.Module,
        num_classes: int,
        init_value: float = 0.9
    ):
        super().__init__()
        self.base_net = base_net
        self.noise_layer = NoiseAdaptionLayer(
            num_classes=num_classes,
            init_value=init_value
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base network and noise adaption layer."""
        logits = self.base_net(x)
        adapted_logits = self.noise_layer(logits)
        return adapted_logits
    
    def get_clean_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get output from base network without noise adaption."""
        return self.base_net(x)
    
    def get_transition_matrix(self) -> torch.Tensor:
        """Get the learned noise transition matrix."""
        return self.noise_layer.get_transition_matrix()
