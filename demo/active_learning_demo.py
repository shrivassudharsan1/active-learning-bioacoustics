#!/usr/bin/env python3
"""
Minimal active learning demo: multiclass classification with entropy-based sampling.

Uses fully synthetic data (sklearn.make_classification). No real audio or private data.

Pipeline:
  1. Generate a feature matrix and labels (stand-in for fixed embeddings from audio).
  2. Hold out a test set; the rest is the pool for labeling.
  3. Start with a small random labeled set; everything else is "unlabeled" for querying.
  4. Each round: train RandomForest, score unlabeled points by prediction entropy,
     add the top-k most uncertain to the labeled set, retrain.
  5. Report test accuracy after each round.

Entropy: for each sample, H = -sum_c p_c log(p_c) over class probabilities p_c.
Higher H = more uncertain = better to label next (uncertainty sampling).
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def entropy_from_proba(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy per row from predict_proba output, shape (n_samples, n_classes)."""
    eps = 1e-12
    p = np.clip(proba, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def run_demo(
    *,
    n_samples: int = 2000,
    n_features: int = 32,
    n_classes: int = 5,
    test_size: float = 0.2,
    initial_labeled: int = 40,
    budget_per_round: int = 40,
    n_rounds: int = 10,
    random_state: int = 42,
) -> None:
    rng = np.random.default_rng(random_state)

    # --- 1. Synthetic data (reproducible) ---
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(20, n_features),
        n_redundant=max(0, n_features // 4),
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=1.2,
        random_state=random_state,
    )

    # --- 2. Fixed test set; pool is only for training / querying ---
    X_pool, X_test, y_pool, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    n_pool = len(X_pool)
    indices = np.arange(n_pool)
    rng.shuffle(indices)

    # Labeled: first `initial_labeled` shuffled indices; unlabeled: the rest
    labeled_idx = indices[:initial_labeled].copy()
    unlabeled_idx = indices[initial_labeled:].copy()

    print("=== Active learning demo (synthetic data) ===")
    print(f"Pool: {n_pool} samples | Test: {len(X_test)} | Classes: {n_classes}")
    print(f"Start: {len(labeled_idx)} labeled, {len(unlabeled_idx)} unlabeled")
    print(f"Each round: add {budget_per_round} by highest entropy | Rounds: {n_rounds}\n")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
    )

    for r in range(1, n_rounds + 1):
        X_lab = X_pool[labeled_idx]
        y_lab = y_pool[labeled_idx]

        model.fit(X_lab, y_lab)
        y_pred_test = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_test)

        print(f"Round {r:2d} | labeled={len(labeled_idx):4d} | test accuracy = {acc:.4f}")

        if len(unlabeled_idx) == 0:
            print("  (unlabeled pool empty — stopping)")
            break
        if r == n_rounds:
            break

        # Entropy on unlabeled only
        proba = model.predict_proba(X_pool[unlabeled_idx])
        ent = entropy_from_proba(proba)
        k = min(budget_per_round, len(unlabeled_idx))
        # Indices within unlabeled_idx array of top-k entropy
        local_order = np.argsort(-ent)[:k]
        picked = unlabeled_idx[local_order]

        labeled_idx = np.concatenate([labeled_idx, picked])
        # Remove picked from unlabeled (mask)
        mask = np.ones(len(unlabeled_idx), dtype=bool)
        mask[local_order] = False
        unlabeled_idx = unlabeled_idx[mask]

    print("\nDone.")


if __name__ == "__main__":
    run_demo()
