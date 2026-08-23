from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import RFECV, SelectKBest, SelectPercentile, f_classif, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler


class NeuralDecisionTree(nn.Module):
    """Differentiable Neural Decision Tree (DNDT).

    Implements a soft, differentiable binary decision tree of depth D.
    - Number of inner decision nodes: 2^D - 1
    - Number of leaf nodes: 2^D
    - Routing is computed via logistic sigmoid projections on input features.
    - Leaf predictions are learned probability distributions over classes.

    Reference:
        Kontschieder et al. (ICCV 2015), "Deep Neural Decision Forests".
        Islam, Chowdhury, & Kabir (ESWA 2026), "Robust COVID-19 detection from cough sounds using DNDT and DNDF".
    """

    def __init__(
        self,
        depth: int = 4,
        num_features: int = 800,
        used_features_rate: float = 1.0,
        num_classes: int = 2,
        temperature: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"Depth must be >= 1, got {depth}")
        if not (0.0 < used_features_rate <= 1.0):
            raise ValueError(f"used_features_rate must be in (0, 1], got {used_features_rate}")

        if num_features < 1:
            raise ValueError(f"NeuralDecisionTree requires num_features >= 1, got {num_features}")

        self.depth = depth
        self.num_features = num_features
        self.used_features_rate = used_features_rate
        self.num_classes = num_classes
        self.temperature = float(temperature)
        self.num_leaves = 2**depth
        self.num_inner_nodes = self.num_leaves - 1

        # Feature subsampling indices (feature bagging)
        used_feature_count = max(1, min(num_features, int(math.ceil(num_features * used_features_rate))))
        self.used_feature_count = used_feature_count

        if random_state is not None:
            rng = np.random.RandomState(random_state)
            chosen_indices = rng.choice(num_features, size=used_feature_count, replace=False)
        else:
            chosen_indices = np.random.choice(num_features, size=used_feature_count, replace=False)

        self.register_buffer("feature_indices", torch.tensor(chosen_indices, dtype=torch.long))

        # Feature normalization and linear decision routing layer
        self.input_norm = nn.LayerNorm(used_feature_count)
        self.decision_layer = nn.Linear(used_feature_count, self.num_inner_nodes)

        # Trainable leaf distribution logits [num_leaves, num_classes]
        self.leaf_logits = nn.Parameter(
            torch.empty(self.num_leaves, num_classes).uniform_(-0.1, 0.1)
        )

        # Build path routing matrix M: shape [num_leaves, num_inner_nodes]
        # M[l, j] = +1 if path to leaf l takes left branch at node j
        # M[l, j] = -1 if path to leaf l takes right branch at node j
        # M[l, j] =  0 if node j is not on path to leaf l
        path_matrix = torch.zeros(self.num_leaves, self.num_inner_nodes, dtype=torch.float32)
        for leaf_idx in range(self.num_leaves):
            curr_node = 0
            for d in range(self.depth):
                bit = (leaf_idx >> (self.depth - 1 - d)) & 1
                if bit == 0:
                    path_matrix[leaf_idx, curr_node] = 1.0  # left branch (probability p)
                    curr_node = 2 * curr_node + 1
                else:
                    path_matrix[leaf_idx, curr_node] = -1.0  # right branch (probability 1 - p)
                    curr_node = 2 * curr_node + 2
        self.register_buffer("path_matrix", path_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing class probabilities for input batch x.

        Args:
            x: Tensor of shape [batch_size, num_features]

        Returns:
            Tensor of shape [batch_size, num_classes] containing predicted probabilities.
        """
        # Subsample features
        x_sub = torch.index_select(x, dim=1, index=self.feature_indices)
        x_norm = self.input_norm(x_sub)

        # Compute decision node routing probabilities in (0, 1)
        decision_logits = self.decision_layer(x_norm) / self.temperature
        d_prob = torch.sigmoid(decision_logits)  # [B, num_inner_nodes]
        d_prob = torch.clamp(d_prob, min=1e-7, max=1.0 - 1e-7)

        # Log routing probabilities
        log_d = torch.log(d_prob)
        log_1_minus_d = torch.log(1.0 - d_prob)

        # Compute log path probabilities for each leaf
        # M_pos = (path_matrix == 1.0), M_neg = (path_matrix == -1.0)
        pos_mask = (self.path_matrix > 0).float()   # [num_leaves, num_inner_nodes]
        neg_mask = (self.path_matrix < 0).float()   # [num_leaves, num_inner_nodes]

        # [B, num_leaves] = [B, num_inner_nodes] @ [num_inner_nodes, num_leaves]
        log_leaf_prob = torch.matmul(log_d, pos_mask.t()) + torch.matmul(log_1_minus_d, neg_mask.t())
        leaf_prob = F.softmax(log_leaf_prob, dim=-1)  # [B, num_leaves]

        # Leaf class distributions [num_leaves, num_classes]
        pi = F.softmax(self.leaf_logits, dim=-1)

        # Output class probabilities [B, num_classes]
        out = torch.matmul(leaf_prob, pi)
        return out


class NeuralDecisionForest(nn.Module):
    """Deep Neural Decision Forest (DNDF).

    An ensemble of differentiable NeuralDecisionTree models.
    The ensemble prediction is the arithmetic mean of individual tree predictions.

    Reference:
        Kontschieder et al. (ICCV 2015).
        Islam et al. (ESWA 2026).
    """

    def __init__(
        self,
        num_trees: int = 20,
        depth: int = 4,
        num_features: int = 800,
        used_features_rate: float = 0.8,
        num_classes: int = 2,
        temperature: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        if num_trees < 1:
            raise ValueError(f"num_trees must be >= 1, got {num_trees}")

        self.num_trees = num_trees
        self.depth = depth
        self.num_features = num_features
        self.used_features_rate = used_features_rate
        self.num_classes = num_classes
        self.temperature = float(temperature)

        self.trees = nn.ModuleList([
            NeuralDecisionTree(
                depth=depth,
                num_features=num_features,
                used_features_rate=used_features_rate,
                num_classes=num_classes,
                temperature=temperature,
                random_state=(random_state + i) if random_state is not None else None,
            )
            for i in range(num_trees)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass averaging class probabilities across all trees.

        Args:
            x: Tensor of shape [batch_size, num_features]

        Returns:
            Tensor of shape [batch_size, num_classes]
        """
        tree_outputs = [tree(x) for tree in self.trees]
        stacked = torch.stack(tree_outputs, dim=0)  # [num_trees, batch_size, num_classes]
        return torch.mean(stacked, dim=0)


@dataclass
class DNDFHyperparameters:
    """Hyperparameters for DNDT and DNDF classifiers."""
    model_type: str = "dndf"  # "dndt" or "dndf"
    num_trees: int = 20
    depth: int = 4
    used_features_rate: float = 0.8
    temperature: float = 1.0
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 10
    use_smote: bool = True
    feature_selection: str = "none"  # "none", "rfecv_extratrees", "f_classif"
    n_selected_features: int | None = None
    threshold: float = 0.5
    device: str = "auto"
    random_state: int = 42


class DNDFClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-Learn compatible estimator for DNDT / DNDF models with SMOTE,

    feature selection (RFECV / ExtraTrees / F-score), Bayesian/Grid tuning hooks,
    and threshold optimization matching Islam et al. (ESWA 2026).
    """

    def __init__(
        self,
        model_type: str = "dndf",
        num_trees: int = 20,
        depth: int = 4,
        used_features_rate: float = 0.8,
        temperature: float = 1.0,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 50,
        patience: int = 10,
        use_smote: bool = True,
        feature_selection: str = "none",
        n_selected_features: int | None = None,
        threshold: float = 0.5,
        device: str = "auto",
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.num_trees = num_trees
        self.depth = depth
        self.used_features_rate = used_features_rate
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.use_smote = use_smote
        self.feature_selection = feature_selection
        self.n_selected_features = n_selected_features
        self.threshold = threshold
        self.device = device
        self.random_state = random_state

        self.model_: nn.Module | None = None
        self.classes_: np.ndarray = np.array([0, 1])
        self.device_: torch.device = torch.device("cpu")
        self.scaler_: StandardScaler | None = None
        self.feature_selector_: Any = None
        self.best_threshold_: float = threshold
        self.train_history_: list[dict[str, float]] = []

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _apply_feature_selection_train(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        k = self.n_selected_features or min(80, X.shape[1])
        if self.feature_selection == "rfecv_extratrees":
            estimator = ExtraTreesClassifier(
                n_estimators=50, random_state=self.random_state, n_jobs=-1
            )
            min_features = self.n_selected_features or max(10, X.shape[1] // 10)
            selector = RFECV(
                estimator=estimator,
                step=0.1,
                cv=3,
                scoring="roc_auc",
                min_features_to_select=min_features,
                n_jobs=-1,
            )
            X_sel = selector.fit_transform(X, y)
            self.feature_selector_ = selector
            return X_sel
        elif self.feature_selection in ("f_classif", "anova") and k < X.shape[1]:
            selector = SelectKBest(score_func=f_classif, k=k)
            X_sel = selector.fit_transform(X, y)
            self.feature_selector_ = selector
            return X_sel
        elif self.feature_selection == "mutual_info" and k < X.shape[1]:
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_sel = selector.fit_transform(X, y)
            self.feature_selector_ = selector
            return X_sel
        else:
            self.feature_selector_ = None
            return X

    def _apply_feature_selection_transform(self, X: np.ndarray) -> np.ndarray:
        if self.feature_selector_ is not None:
            return self.feature_selector_.transform(X)
        return X

    def fit(
        self,
        X: np.ndarray | Sequence[Sequence[float]],
        y: np.ndarray | Sequence[int],
        X_val: np.ndarray | Sequence[Sequence[float]] | None = None,
        y_val: np.ndarray | Sequence[int] | None = None,
        optimize_threshold: bool = True,
    ) -> "DNDFClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.device_ = self._resolve_device()

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)

        # Standard Scaling fitted strictly on training data
        self.scaler_ = StandardScaler()
        X_arr = self.scaler_.fit_transform(X_arr)

        # Optional SMOTE oversampling on scaled training data
        if self.use_smote and len(np.unique(y_arr)) > 1:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(k_neighbors=min(3, max(1, np.sum(y_arr == 1) - 1)), random_state=self.random_state)
                X_arr, y_arr = smote.fit_resample(X_arr, y_arr)
            except Exception:
                pass

        # Feature Selection
        X_arr = self._apply_feature_selection_train(X_arr, y_arr)

        num_features = X_arr.shape[1]
        num_classes = 2

        # Class weights for balanced loss
        unique_classes, counts = np.unique(y_arr, return_counts=True)
        if len(unique_classes) == 2 and counts[0] > 0 and counts[1] > 0:
            weights = len(y_arr) / (2.0 * counts)
            class_weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device_)
        else:
            class_weights_t = None

        # Initialize DNDT or DNDF model
        if self.model_type.lower() in ("dndt", "tree"):
            self.model_ = NeuralDecisionTree(
                depth=self.depth,
                num_features=num_features,
                used_features_rate=1.0,
                num_classes=num_classes,
                temperature=self.temperature,
                random_state=self.random_state,
            ).to(self.device_)
        else:
            self.model_ = NeuralDecisionForest(
                num_trees=self.num_trees,
                depth=self.depth,
                num_features=num_features,
                used_features_rate=self.used_features_rate,
                num_classes=num_classes,
                temperature=self.temperature,
                random_state=self.random_state,
            ).to(self.device_)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.max_epochs), eta_min=1e-5
        )

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.long),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            drop_last=False,
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_scaled = self.scaler_.transform(np.asarray(X_val, dtype=np.float32))
            X_val_sel = self._apply_feature_selection_transform(X_val_scaled)
            X_val_tensor = torch.tensor(X_val_sel, dtype=torch.float32, device=self.device_)
            y_val_arr = np.asarray(y_val, dtype=np.int64)

        best_loss = float("inf")
        best_state = None
        patience_count = 0
        self.train_history_ = []

        for epoch in range(self.max_epochs):
            self.model_.train()
            epoch_loss = 0.0
            total_samples = 0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)

                optimizer.zero_grad()
                probs = self.model_(batch_x)
                # Compute Cross Entropy via NLL on log probabilities with class weighting
                log_probs = torch.log(torch.clamp(probs, min=1e-7, max=1.0))
                loss = F.nll_loss(log_probs, batch_y, weight=class_weights_t)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch_y)
                total_samples += len(batch_y)

            scheduler.step()
            mean_train_loss = epoch_loss / max(1, total_samples)
            epoch_record = {"epoch": epoch + 1, "train_loss": mean_train_loss}

            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    val_probs = self.model_(X_val_tensor)
                    val_log_probs = torch.log(torch.clamp(val_probs, min=1e-7, max=1.0))
                    val_loss = F.nll_loss(
                        val_log_probs,
                        torch.tensor(y_val_arr, device=self.device_),
                        weight=class_weights_t,
                    ).item()

                epoch_record["val_loss"] = val_loss
                if (epoch + 1) % 10 == 0 or epoch == self.max_epochs - 1:
                    print(f"    Epoch {epoch+1:02d}/{self.max_epochs:02d} - Train Loss: {mean_train_loss:.4f} | Val Loss: {val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self.patience:
                        break
            else:
                if (epoch + 1) % 10 == 0 or epoch == self.max_epochs - 1:
                    print(f"    Epoch {epoch+1:02d}/{self.max_epochs:02d} - Train Loss: {mean_train_loss:.4f}")

            self.train_history_.append(epoch_record)

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # Optimize threshold on validation data if requested
        if optimize_threshold and has_val:
            val_probs_eval = self.predict_proba(X_val)[:, 1]
            try:
                from covid_rars.metrics import best_threshold_by_balanced_accuracy
                self.best_threshold_ = float(best_threshold_by_balanced_accuracy(y_val_arr, val_probs_eval))
            except Exception:
                self.best_threshold_ = self.threshold
        else:
            self.best_threshold_ = self.threshold

        return self

    def predict_proba(self, X: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        if self.scaler_ is not None:
            X_arr = self.scaler_.transform(X_arr)
        X_sel = self._apply_feature_selection_transform(X_arr)
        X_tensor = torch.tensor(X_sel, dtype=torch.float32, device=self.device_)
        with torch.no_grad():
            probs = self.model_(X_tensor).cpu().numpy()
        return np.clip(probs, 0.0, 1.0)

    def predict(self, X: np.ndarray | Sequence[Sequence[float]], threshold: float | None = None) -> np.ndarray:
        t = threshold if threshold is not None else self.best_threshold_
        proba = self.predict_proba(X)[:, 1]
        return (proba >= t).astype(int)
