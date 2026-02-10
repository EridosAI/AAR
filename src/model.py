"""Association MLP -- transforms passage embeddings into association space.

Architecture: 4-layer MLP with LayerNorm, GELU activations, and a learned
residual connection. The output is L2-normalised to lie on the unit hypersphere.

    f(x) = normalize(alpha * x + (1 - alpha) * g(x))

where g is the MLP transformation and alpha is a learned scalar.
"""

import torch
import torch.nn as nn


class AssociationMLP(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=1024, num_layers=4):
        super().__init__()
        layers = []
        # Input projection
        layers.append(nn.Linear(embedding_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        # Output projection
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        self.net = nn.Sequential(*layers)
        # Learned residual blend weight (initialised to 0.5 via sigmoid(0))
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        Args:
            x: passage or query embedding, shape [batch, embedding_dim]
        Returns:
            association-space embedding, shape [batch, embedding_dim], L2-normalised
        """
        transformed = self.net(x)
        alpha = torch.sigmoid(self.residual_weight)
        out = alpha * x + (1 - alpha) * transformed
        return nn.functional.normalize(out, dim=-1)
