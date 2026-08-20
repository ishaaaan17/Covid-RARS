import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix

def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Computes comprehensive clinical validation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, y_prob)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "brier_score": brier
    }

def bootstrap_confidence_intervals(y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, alpha: float = 0.05, seed: int = 42) -> dict:
    """Computes 95% bootstrap confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    boot_metrics = {"auroc": [], "auprc": [], "sensitivity": [], "specificity": [], "accuracy": []}
    
    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        sample_y_true = y_true[idx]
        sample_y_prob = y_prob[idx]
        
        if len(np.unique(sample_y_true)) < 2:
            continue
            
        m = compute_classification_metrics(sample_y_true, sample_y_prob)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])
            
    ci_results = {}
    lower_p = (alpha / 2.0) * 100
    upper_p = (1.0 - alpha / 2.0) * 100
    
    for k, values in boot_metrics.items():
        if len(values) > 0:
            ci_results[f"{k}_mean"] = float(np.mean(values))
            ci_results[f"{k}_ci_lower"] = float(np.percentile(values, lower_p))
            ci_results[f"{k}_ci_upper"] = float(np.percentile(values, upper_p))
            
    return ci_results
