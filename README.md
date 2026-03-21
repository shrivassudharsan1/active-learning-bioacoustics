# Active learning for bioacoustics — minimal demo

A tiny, **standalone** Python example of **pool-based active learning** for **multiclass classification**, written for researchers who work with **bird / bioacoustic** data in the field.

In practice, you often:

1. Turn recordings into **fixed embeddings** (e.g. from a pretrained audio model).
2. Train a **lightweight classifier** on a **small** set of labeled segments.
3. Use **active learning** to choose which unlabeled clips to annotate next—e.g. by **uncertainty** (high entropy of predicted class probabilities).

This repo **does not** use real audio or any private dataset. It uses **synthetic tabular data** from scikit-learn so you can run and share the idea without leaking sensitive data.

## What this demo does

| Step | Description |
|------|-------------|
| Data | `sklearn.datasets.make_classification` → features + labels |
| Split | Train **pool** + held-out **test** (test never used for querying) |
| Start | Random **small labeled** subset; rest of pool is **unlabeled** |
| Loop | Train `RandomForestClassifier` → `predict_proba` on unlabeled → **entropy** per sample → label top‑**k** most uncertain → add to labeled set → retrain |
| Metric | **Accuracy on the fixed test set** printed each round |

**Entropy sampling:** for each unlabeled point, compute  
\(H = -\sum_c p_c \log p_c\) over class probabilities \(p_c\). Higher \(H\) ⇒ more uncertain ⇒ prioritize for labeling.

## Requirements

- Python 3.10+ recommended  
- See `requirements.txt`

## Run

```bash
cd active-learning-bioacoustics
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python demo/active_learning_demo.py
```

## Example output

```
=== Active learning demo (synthetic data) ===
Pool: 1600 samples | Test: 400 | Classes: 5
Start: 40 labeled, 1560 unlabeled
Each round: add 40 by highest entropy | Rounds: 10

Round  1 | labeled=  40 | test accuracy = 0.4550
Round  2 | labeled=  80 | test accuracy = 0.4950
Round  3 | labeled= 120 | test accuracy = 0.5350
Round  4 | labeled= 160 | test accuracy = 0.5750
Round  5 | labeled= 200 | test accuracy = 0.5750
Round  6 | labeled= 240 | test accuracy = 0.6175
Round  7 | labeled= 280 | test accuracy = 0.6600
Round  8 | labeled= 320 | test accuracy = 0.6875
Round  9 | labeled= 360 | test accuracy = 0.6800
Round 10 | labeled= 400 | test accuracy = 0.6675

Done.
```

(Exact numbers depend on `random_state` and sklearn version.)

## Limitations (intentional)

- **Synthetic** data only; not a substitute for real bioacoustic benchmarks.
- **Uncertainty sampling** is one strategy; others include margin, diversity, or hybrid criteria.
- **Accuracy** is shown for clarity; imbalanced species distributions often need **macro-F1**, per-class recall, etc.

## Push to GitHub

From this folder (after you create an empty repo on GitHub, or use your existing `active-learning-bioacoustics` repo):

```bash
git remote add origin https://github.com/<your-username>/active-learning-bioacoustics.git
git push -u origin main
```

If your default branch on GitHub is `master`, use `git push -u origin main` only if `main` exists locally; otherwise rename with `git branch -M main` first.

## License

Use freely for teaching and demos. No warranty.
