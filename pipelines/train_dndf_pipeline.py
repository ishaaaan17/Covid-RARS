import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from models.dndf_model import DNDF
from evaluation.metrics_engine import compute_classification_metrics, bootstrap_confidence_intervals

def load_data_from_path(dataset_path: str):
    X_path = os.path.join(dataset_path, "cough_X_features_np.npy")
    y_path = os.path.join(dataset_path, "cough_y_features_np.npy")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing feature arrays in: {dataset_path}")
        
    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    else:
        y = y.ravel()
        
    return X, y

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        
        probs = model(batch_X)
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        loss = criterion(torch.log(probs), batch_y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
        
    return total_loss / len(loader.dataset)

def evaluate_model(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            probs = model(batch_X)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(batch_y.numpy())
            
    all_probs = np.vstack(all_probs)
    all_targets = np.concatenate(all_targets)
    return all_probs, all_targets

def run_cross_validation(dataset_dir: str, num_trees: int = 15, depth: int = 5, n_splits: int = 5,
                         epochs: int = 50, batch_size: int = 32, lr: float = 0.001, device: str = "cpu"):
    print(f"[+] Loading Dataset from: {dataset_dir}")
    X, y = load_data_from_path(dataset_dir)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_oof_probs = np.zeros((len(y), 2))
    all_oof_targets = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Running Fold {fold + 1}/{n_splits} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = DNDF(num_features=X.shape[1], num_classes=2, num_trees=num_trees, depth=depth, use_feature_extractor=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.NLLLoss()
        
        best_auroc = -1.0
        best_probs = None
        
        for epoch in range(epochs):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            probs, targets = evaluate_model(model, val_loader, device)
            m = compute_classification_metrics(targets, probs[:, 1])
            
            if m["auroc"] > best_auroc:
                best_auroc = m["auroc"]
                best_probs = probs
                
        all_oof_probs[val_idx] = best_probs
        all_oof_targets[val_idx] = y_val
        
        fold_m = compute_classification_metrics(y_val, best_probs[:, 1])
        print(f"Fold {fold + 1} Best AUROC: {fold_m['auroc']:.4f} | AUPRC: {fold_m['auprc']:.4f} | Sens: {fold_m['sensitivity']:.4f} | Spec: {fold_m['specificity']:.4f}")
        
    print("\n================== Overall Out-Of-Fold Evaluation ==================")
    overall_metrics = compute_classification_metrics(all_oof_targets, all_oof_probs[:, 1])
    ci = bootstrap_confidence_intervals(all_oof_targets, all_oof_probs[:, 1], n_bootstraps=1000)
    
    print(f"OOF AUROC: {overall_metrics['auroc']:.4f} (95% CI: {ci['auroc_ci_lower']:.4f} - {ci['auroc_ci_upper']:.4f})")
    print(f"OOF AUPRC: {overall_metrics['auprc']:.4f} (95% CI: {ci['auprc_ci_lower']:.4f} - {ci['auprc_ci_upper']:.4f})")
    print(f"Sensitivity: {overall_metrics['sensitivity']:.4f} | Specificity: {overall_metrics['specificity']:.4f}")
    
    return overall_metrics, ci

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="Extracted Features/Coswara")
    parser.add_argument("--num_trees", type=int, default=15)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_cross_validation(args.dataset_dir, args.num_trees, args.depth, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)
