import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepNeuralDecisionTree(nn.Module):
    """
    Deep Neural Decision Tree (DNDT) based on Yang et al. (2018).
    Implements soft per-feature binning via cutpoints and differentiable outer-product routing.
    """
    def __init__(self, in_features: int, num_classes: int = 2, num_cutpoints: int = 1, temperature: float = 1.0):
        super(DeepNeuralDecisionTree, self).__init__()
        assert in_features <= 12, f"DNDT requires in_features <= 12 to avoid leaf explosion, got {in_features}"
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_cutpoints = num_cutpoints
        self.temperature = temperature
        self.num_leaves = (num_cutpoints + 1) ** in_features

        # Initialize trainable cutpoints per feature
        init_cuts = torch.linspace(-1.0, 1.0, num_cutpoints).unsqueeze(0).repeat(in_features, 1)
        self.cutpoints = nn.Parameter(init_cuts)
        
        # Scaling parameter for binning slope steepness
        self.scales = nn.Parameter(torch.ones(in_features, 1))

        # Trainable leaf class parameter distributions
        self.leaf_logits = nn.Parameter(torch.randn(self.num_leaves, num_classes) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # x: [batch_size, in_features]
        # Reshape for broadcasting cutpoints: [batch_size, in_features, 1]
        x_exp = x.unsqueeze(-1)
        
        # Sigmoid soft bin routing: [batch_size, in_features, num_cutpoints]
        sigmoid_outputs = torch.sigmoid((x_exp - self.cutpoints) * self.scales / self.temperature)
        
        # Construct soft bin indicator probabilities per feature: [batch_size, in_features, num_cutpoints + 1]
        zeros = torch.zeros(batch_size, self.in_features, 1, device=x.device)
        ones = torch.ones(batch_size, self.in_features, 1, device=x.device)
        
        extended = torch.cat([zeros, sigmoid_outputs, ones], dim=-1)
        bin_probs = extended[:, :, 1:] - extended[:, :, :-1]
        
        # Joint leaf routing via continuous tensor product across features
        leaf_probs = bin_probs[:, 0, :]
        for d in range(1, self.in_features):
            # Outer product routing
            leaf_probs = torch.bmm(leaf_probs.unsqueeze(-1), bin_probs[:, d, :].unsqueeze(1)).view(batch_size, -1)

        # Per-leaf class probability normalization
        class_distribution = F.softmax(self.leaf_logits, dim=-1)
        
        # Final output distribution: [batch_size, num_classes]
        output_probs = torch.matmul(leaf_probs, class_distribution)
        return output_probs


class DecisionTree(nn.Module):
    """
    Base oblique differentiable decision tree for DNDF (Kontschieder et al., 2015).
    """
    def __init__(self, in_features: int, num_classes: int = 2, depth: int = 4, temperature: float = 1.0):
        super(DecisionTree, self).__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.num_leaves = 2 ** depth
        self.num_internals = self.num_leaves - 1
        self.temperature = temperature

        self.decision_layer = nn.Linear(in_features, self.num_internals)
        self.leaf_logits = nn.Parameter(torch.randn(self.num_leaves, num_classes) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Decision split probabilities: [batch_size, num_internals]
        d_probs = torch.sigmoid(self.decision_layer(x) / self.temperature)
        
        # Construct paths from root (node 0) to all 2^depth leaves
        leaf_probs = torch.ones(batch_size, self.num_leaves, device=x.device)
        for leaf_idx in range(self.num_leaves):
            curr_idx = leaf_idx
            node = 0
            for d in range(self.depth):
                split_val = d_probs[:, node]
                # Determine binary branch direction
                is_right = (curr_idx >> (self.depth - 1 - d)) & 1
                if is_right:
                    leaf_probs[:, leaf_idx] = leaf_probs[:, leaf_idx] * split_val
                    node = 2 * node + 2
                else:
                    leaf_probs[:, leaf_idx] = leaf_probs[:, leaf_idx] * (1.0 - split_val)
                    node = 2 * node + 1
                    
        class_dist = F.softmax(self.leaf_logits, dim=-1)
        output_probs = torch.matmul(leaf_probs, class_dist)
        return output_probs


class DeepNeuralDecisionForest(nn.Module):
    """
    Deep Neural Decision Forest (DNDF) ensemble.
    """
    def __init__(self, in_features: int, num_classes: int = 2, num_trees: int = 12, depth: int = 4, temperature: float = 1.0):
        super(DeepNeuralDecisionForest, self).__init__()
        self.trees = nn.ModuleList([
            DecisionTree(in_features=in_features, num_classes=num_classes, depth=depth, temperature=temperature)
            for _ in range(num_trees)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tree_outputs = [tree(x) for tree in self.trees]
        # Ensemble average: [batch_size, num_classes]
        return torch.mean(torch.stack(tree_outputs, dim=0), dim=0)
