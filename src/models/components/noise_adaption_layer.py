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


class InstanceNoiseAdaptionNet(nn.Module):
    """Noise adaption with instance-dependent transition via a small MLP."""

    def __init__(
        self,
        base_net: nn.Module,
        num_classes: int,
        init_value: float = 0.9,
        feature_dim: int | None = None,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.base_net = base_net
        self.num_classes = num_classes

        if feature_dim is None:
            feature_dim = self._infer_feature_dim(base_net, num_classes)

        self.base_transition = nn.Parameter(
            self._init_transition(num_classes=num_classes, init_value=init_value)
        )

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes * num_classes),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _init_transition(num_classes: int, init_value: float) -> torch.Tensor:
        init_matrix = torch.eye(num_classes) * init_value
        init_matrix += (1.0 - init_value) / (num_classes - 1) * (1 - torch.eye(num_classes))
        return init_matrix

    @staticmethod
    def _infer_feature_dim(base_net: nn.Module, num_classes: int) -> int:
        if hasattr(base_net, "feature_dim"):
            return int(getattr(base_net, "feature_dim"))
        if hasattr(base_net, "out_dim"):
            return int(getattr(base_net, "out_dim"))
        if hasattr(base_net, "model") and hasattr(base_net.model, "classifier"):
            classifier = base_net.model.classifier
            if isinstance(classifier, nn.Sequential) and len(classifier) > 0:
                first = classifier[0]
                if isinstance(first, nn.Linear):
                    return int(first.in_features)
        return int(num_classes)

    def _extract_features(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if hasattr(self.base_net, "forward_features"):
            return self.base_net.forward_features(x)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_net(x)
        probs = F.softmax(logits, dim=1)

        features = self._extract_features(x, logits)
        delta = self.mlp(features).view(-1, self.num_classes, self.num_classes)
        transition_logits = self.base_transition.unsqueeze(0) + delta
        transition = F.softmax(transition_logits, dim=2)

        adapted_probs = torch.bmm(probs.unsqueeze(1), transition.transpose(1, 2)).squeeze(1)
        adapted_probs = torch.clamp(adapted_probs, min=1e-7, max=1.0)
        adapted_logits = torch.log(adapted_probs)
        return adapted_logits

    def get_clean_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_net(x)

    def get_transition_matrix(self) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.base_transition, dim=1)
