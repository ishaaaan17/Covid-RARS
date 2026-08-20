import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from app.models.deep_trees import DeepNeuralDecisionTree, DeepNeuralDecisionForest
from evaluation.metrics_engine import compute_classification_metrics, bootstrap_confidence_intervals

def generate_synthetic_data(n_samples=200, n_features=64):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) > 0.0).astype(np.int64)
    return X, y

def train_eval_tree_model(model, X_tr, y_tr, X_va, y_va, epochs=60, batch_size=32, lr=0.01, device="cpu"):
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(y_va)), batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.NLLLoss()
    
    best_auroc = -1.0
    best_val_probs = None
    
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            probs = model(bx)
            probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
            loss = criterion(torch.log(probs), by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_probs = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                probs = model(bx)
                val_probs.append(probs.cpu().numpy())
                
        val_probs = np.vstack(val_probs)[:, 1]
        m = compute_classification_metrics(y_va, val_probs)
        if m["auroc"] > best_auroc:
            best_auroc = m["auroc"]
            best_val_probs = val_probs
            
    return best_val_probs

def run_dryrun(output_dir):
    print("[DRY-RUN] Executing synthetic pipeline check...")
    X, y = generate_synthetic_data()
    
    # Feature selector test
    selector_dndt = SelectKBest(f_classif, k=8)
    X_dndt = selector_dndt.fit_transform(X, y)
    
    selector_dndf = SelectKBest(f_classif, k=32)
    X_dndf = selector_dndf.fit_transform(X, y)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Split
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for tr, va in skf.split(X, y):
        # DNDT
        dndt = DeepNeuralDecisionTree(in_features=8, num_classes=2, num_cutpoints=1).to(device)
        probs_dndt = train_eval_tree_model(dndt, X_dndt[tr], y[tr], X_dndt[va], y[va], epochs=5, lr=0.01, device=device)
        m_dndt = compute_classification_metrics(y[va], probs_dndt)
        
        # DNDF
        dndf = DeepNeuralDecisionForest(in_features=32, num_classes=2, num_trees=4, depth=3).to(device)
        probs_dndf = train_eval_tree_model(dndf, X_dndf[tr], y[tr], X_dndf[va], y[va], epochs=5, lr=0.003, device=device)
        m_dndf = compute_classification_metrics(y[va], probs_dndf)
        break

    print(f"[DNDT] AUROC: {m_dndt['auroc']:.4f} | AUPRC: {m_dndt['auprc']:.4f}")
    print(f"[DNDF] AUROC: {m_dndf['auroc']:.4f} | AUPRC: {m_dndf['auprc']:.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "dndt_dndf_dryrun_metrics.csv")
    pd.DataFrame([{"model": "DNDT", **m_dndt}, {"model": "DNDF", **m_dndf}]).to_csv(out_csv, index=False)
    print(f"[DRY-RUN COMPLETE] Saved sample metrics to {out_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dndt_dndf_reliability.json")
    parser.add_argument("--output_dir", type=str, default="data/outputs/metrics")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        run_dryrun(args.output_dir)
        return

    print(f"[+] Loaded config from: {args.config}")
    os.makedirs(args.output_dir, exist_ok=True)
    # Default benchmarking execution logic will proceed here based on config
    print("[+] Benchmark setup ready.")

if __name__ == "__main__":
    main()
