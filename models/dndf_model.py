import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DNDT(nn.Module):
    """
    Deep Neural Decision Tree (DNDT) using continuous, differentiable soft routing.
    Computes split decisions through soft binning and constructs leaf class distributions.
    """
    def __init__(self, num_features: int, num_classes: int, depth: int = 4, temperature: float = 1.0):
        super(DNDT, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.temperature = temperature
        
        # Linear routing projection for split decision nodes
        self.feature_weights = nn.Parameter(torch.empty(num_features, self.num_leaves))
        self.feature_bias = nn.Parameter(torch.empty(self.num_leaves))
        
        # Leaf class probability logits
        self.leaf_distributions = nn.Parameter(torch.empty(self.num_leaves, num_classes))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.feature_weights, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.feature_weights)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.feature_bias, -bound, bound)
        nn.init.normal_(self.leaf_distributions, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Routing probabilities: [batch_size, num_leaves]
        routing_logits = (torch.matmul(x, self.feature_weights) + self.feature_bias) / self.temperature
        leaf_probs = F.softmax(routing_logits, dim=-1)
        
        # Leaf class distributions: [num_leaves, num_classes]
        class_probs_per_leaf = F.softmax(self.leaf_distributions, dim=-1)
        
        # Expected class probabilities: [batch_size, num_classes]
        class_probs = torch.matmul(leaf_probs, class_probs_per_leaf)
        return class_probs


class DNDF(nn.Module):
    """
    Deep Neural Decision Forest (DNDF).
    Ensemble of differentiable decision trees with optional dense feature backbone.
    """
    def __init__(self, num_features: int, num_classes: int, num_trees: int = 10, depth: int = 4, 
                 temperature: float = 1.0, use_feature_extractor: bool = False, hidden_dim: int = 128):
        super(DNDF, self).__init__()
        self.num_trees = num_trees
        self.use_feature_extractor = use_feature_extractor
        
        if use_feature_extractor:
            self.feature_extractor = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            tree_input_dim = hidden_dim
        else:
            self.feature_extractor = nn.Identity()
            tree_input_dim = num_features

        self.trees = nn.ModuleList([
            DNDT(num_features=tree_input_dim, num_classes=num_classes, depth=depth, temperature=temperature)
            for _ in range(num_trees)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        tree_outputs = [tree(features) for tree in self.trees]
        # Average probability predictions across all ensemble trees
        forest_output = torch.mean(torch.stack(tree_outputs, dim=0), dim=0)
        return forest_output
