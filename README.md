# Amazon Book Recommendation System

**🏆 3rd Place** on class leaderboard (~20 competing teams) · CMPE 256 Recommender Systems · SJSU

A comprehensive comparison of four recommendation architectures — linear, neural, and graph-based — on a large-scale implicit-feedback book interaction dataset.

## Results at a Glance

| Model | NDCG@20 | Notes |
|---|---|---|
| **EASE + IDF** | **0.0517** | **Best linear baseline — my implementation** |
| LightGCN | 0.0517 | Graph-based, matched EASE |
| NCF | ~0.041 | Neural collaborative filtering |
| GraphGPS | ~0.038 | Graph Transformer |
| BPR-MF | ~0.022 | Classical baseline |

## Dataset

| Property | Value |
|---|---|
| Users | 31,668 |
| Items | 38,048 |
| Interactions | 1,237,259 |
| Matrix Density | ~0.10% |
| Feedback Type | Implicit (no ratings) |
| Task | Top-N Ranking (NDCG@20) |

Implicit feedback means we only observe what users interacted with, not what they disliked. This makes it a top-N ranking problem, not a rating-prediction task.

## Training Curves

### LightGCN Loss (Grid Search)
![LightGCN Training Curves](lightgcn_training_curves.png)

### EASE Hyperparameter Search
![EASE Lambda Ablation](ease_lambda_ablation.png)

### EASE IDF Ablation
![EASE IDF Ablation](ease_idf_ablation.png)

### EASE Architecture
![EASE Structure](ease_structure.png)

## Architecture Overview

```
Implicit Interaction Data  (user_id → [item_id, item_id, ...])
           │
           ▼
┌──────────────────────────────────────────────────────┐
│              PREPROCESSING PIPELINE                  │
│  • Item k-core filtering (item_min=3)                │
│  • Warm per-user 80/20 stratified split              │
│  • Warm-eval constraint (items appear in training)   │
│  • BPR-MF: heavy-user cap at 300 interactions        │
└──────────────────────────────────────────────────────┘
           │
     ┌─────┴──────┬──────────────┬──────────────┐
     ▼            ▼              ▼              ▼
  EASE         LightGCN        NCF          GraphGPS
  (Linear)    (GCN-based)   (Neural CF)  (Graph Trans.)
     │
     ▼
  Gram matrix:  G = XᵀX + λI
  Item weights: B = I - diag(G⁻¹) / diag(G⁻¹)
  Predict:      Ŷ = X · B
```

## My Contributions

### EASE Implementation & Tuning
EASE (Embarrassingly Shallow Autoencoders for Sparse Data) is a closed-form linear model that inverts a 38,048 × 38,048 item-item Gram matrix.

**Challenge:** Inverting a 38k×38k dense matrix is memory- and numerically-intensive.

**How I solved it:**
- Migrated from SciPy/NumPy to **PyTorch float32 CPU** pipeline → stable numerics
- Full λ hyperparameter sweep: λ ∈ {100, 200, 300, 400, 500, 600, 700, 1000, 2000}
- Optimal: **λ = 500**

**IDF Ablation Discovery:**
```
Without IDF weighting:  NDCG@20 = 0.0394
With IDF weighting:     NDCG@20 = 0.0517   (+31% relative improvement)
```

### BPR-MF Baseline
- Implemented NumPy-based Matrix Factorization with Bayesian Personalized Ranking (BPR) loss
- Used truncated SVD to identify optimal latent dimension (~200 components = 90% variance)
- Searched `n_factors` ∈ {96, 128, 160, 192, 224}; monitored NDCG@20 every 5 epochs
- Applied heavy-user capping (≤300 interactions) to prevent SGD bias

### Preprocessing & Evaluation Pipeline
- Designed fair, reproducible offline evaluation preventing data leakage
- Stratified per-user 80/20 split maintaining both training coverage and evaluation integrity

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Core Libraries | PyTorch (CPU), NumPy, SciPy, pandas |
| Evaluation | NDCG@20 (top-N ranking) |
| EASE | Custom PyTorch implementation |
| LightGCN | PyTorch Geometric |
| NCF | PyTorch |
| GraphGPS | PyG + custom extensions |

## How to Run

```bash
# EASE (main implementation)
python EASE.py

# EASE vanilla baseline
python EASE_vanilla.py

# Matrix Factorization baseline
python MF_submission.py

# LightGCN
cd lightGCN/
python lightGCN.py
```

## Repository Structure

```
AmazonBook_Recommendation/
├── README.md
├── CHANGELOG.md
├── EASE.py                       ← Full EASE with IDF + tuning
├── EASE_vanilla.py               ← Baseline EASE without modifications
├── MF_submission.py              ← BPR-MF implementation
├── lightGCN/
│   ├── lightGCN.py
│   ├── lightgcn_loss_curves.csv  ← Training data (used for plots)
│   └── lightgcn_grid_history.csv ← Grid search results
├── ease_ndcg_vs_lambda.png       ← Original λ sweep chart
├── ease_idf_ablation.png         ← IDF comparison chart
├── ease_structure.png            ← EASE architecture diagram
├── ease_lambda_ablation.png      ← Generated λ ablation plot
├── lightgcn_training_curves.png  ← Generated LightGCN loss curves
├── LeaderBoardRank.png           ← 3rd place leaderboard screenshot
├── CMPE_256_Final_Report.pdf     ← Full project report
└── Group8.pptx                   ← Presentation slides
```

---

*Academic Project — CMPE 256 (SJSU) · Python · PyTorch · Recommender Systems · Graph Neural Networks*
