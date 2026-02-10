# Models

Trained model checkpoints are not included in this repository due to their dependence on the specific embedding index. Models should be reproduced by running the training scripts.

## Reproducing the Transductive Model (Paper Results)

```bash
# 1. Prepare data (if not already done)
python -c "from src.utils import prepare_data; prepare_data()"

# 2. Train transductive model (combined train+val pairs, ~20,742 pairs)
python -m src.train
```

This produces `models/association_mlp.pt` — a 4-layer MLP with 4,204,545 parameters. Training takes ~2 minutes on an RTX 4080 Super.

**Expected training accuracy:** ~97% (HotpotQA)

## Reproducing the Inductive Model

```bash
python -m src.train_true_inductive
```

This trains on training-split pairs only (~8,758 pairs) and saves to `models/inductive_train_only.pt`.

**Expected training accuracy:** ~94.5% (HotpotQA)

## Architecture

```
AssociationMLP(
  embedding_dim=1024,
  hidden_dim=1024,
  num_layers=4
)
```

- 4-layer MLP: Linear(1024, 1024) + LayerNorm + GELU, repeated, + Linear(1024, 1024)
- Learned residual: `alpha * input + (1 - alpha) * MLP(input)`, L2-normalised
- Total parameters: 4,204,545

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Batch size | 512 |
| Temperature | 0.05 |
| Epochs | 100 |
| Optimizer | AdamW |
| Scheduler | Cosine annealing |
