# Active Learning for Bioacoustics

An active learning system for bioacoustic multi-label classification using pretrained BirdNET embeddings, built to improve sample efficiency in scenarios where labeling wildlife recordings is expensive.

---

## What This Project Does

Labeling bioacoustic recordings is time-consuming and requires domain expertise. This project implements a pool-based active learning loop that:

1. Takes raw audio segments and extracts fixed embeddings using a pretrained audio model (BirdNET/Perch2)
2. Trains a lightweight classifier on a small labeled set
3. Uses uncertainty sampling (entropy-based) to select the most informative unlabeled samples for annotation
4. Iteratively retrains as new labels are added

The result: you need far fewer labeled examples to reach strong classification performance compared to random sampling.

---

## My Work

**Active Learning Loop with Multiple Classification Heads (demo/ and compare_all_strategies.py)**
- Built the core active learning loop: pool-based uncertainty sampling with entropy scoring, multi-round training and evaluation, support for multiple classifier heads (Random Forest, logistic regression, MLP)
- Implemented and compared multiple active learning strategies to benchmark which head + query strategy combination performs best on bioacoustic data
- Designed the loop to operate directly on BirdNET embeddings rather than raw audio, enabling fast iteration without recomputing features each round

**BirdNET Embedding Pipeline (standalone Perch2/BirdNET pipeline)**
- Wrote a standalone embedding pipeline that takes 5-second audio segments and generates BirdNET/Perch2 embeddings ready for the active learning loop
- Handles segment extraction, model inference, and output formatting for downstream classification

---

## Results (Synthetic Baseline)

The demo/ folder contains a synthetic tabular version of the experiment for reproducibility without requiring raw audio data or private datasets:

    Active learning demo (synthetic data)
    Pool: 1600 samples | Test: 400 | Classes: 5
    Round 1 | labeled= 40  | test accuracy = 0.4550
    Round 5 | labeled= 200 | test accuracy = 0.5750
    Round 10| labeled= 400 | test accuracy = 0.6675

With real BirdNET embeddings, the active learning loop shows much stronger early-round gains due to the richer feature space.

---

## Setup

    cd active-learning-bioacoustics
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python demo/active_learning_demo.py

---

## Tech Stack

- Python 3.10+
- BirdNET / Perch2 pretrained audio embeddings
- scikit-learn (RandomForest, entropy sampling)
- NumPy
