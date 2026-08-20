import torch
from app.models.deep_trees import DeepNeuralDecisionTree, DeepNeuralDecisionForest

def test_dndt_shapes_and_gradients():
    batch_size = 16
    in_features = 8
    num_classes = 2
    
    x = torch.randn(batch_size, in_features, requires_grad=True)
    dndt = DeepNeuralDecisionTree(in_features=in_features, num_classes=num_classes, num_cutpoints=1)
    
    out = dndt(x)
    assert out.shape == (batch_size, num_classes), f"DNDT Output shape mismatch: {out.shape}"
    assert torch.allclose(out.sum(dim=-1), torch.ones(batch_size), atol=1e-5), "DNDT Probabilities do not sum to 1"
    
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "DNDT Gradients did not backprop to input"

def test_dndf_shapes_and_gradients():
    batch_size = 16
    in_features = 32
    num_classes = 2
    
    x = torch.randn(batch_size, in_features, requires_grad=True)
    dndf = DeepNeuralDecisionForest(in_features=in_features, num_classes=num_classes, num_trees=4, depth=3)
    
    out = dndf(x)
    assert out.shape == (batch_size, num_classes), f"DNDF Output shape mismatch: {out.shape}"
    assert torch.allclose(out.sum(dim=-1), torch.ones(batch_size), atol=1e-5), "DNDF Probabilities do not sum to 1"
    
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "DNDF Gradients did not backprop to input"

if __name__ == "__main__":
    test_dndt_shapes_and_gradients()
    test_dndf_shapes_and_gradients()
    print("[SUCCESS] All shape, forward, and backward gradient assertions passed!")
