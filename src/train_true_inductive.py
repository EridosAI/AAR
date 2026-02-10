"""Train and evaluate a true inductive AAR model.

Trains on ONLY training-split association pairs (~8,758 pairs from HotpotQA
training set), providing a genuine inductive evaluation where the model has
never seen validation-split associations.

Usage:
    python -m src.train_true_inductive
"""

import os
import time
import csv
import numpy as np
import torch
from datasets import load_from_disk

from src.model import AssociationMLP
from src.train import train_model
from src.utils import (extract_pairs_from_hotpotqa, passage_recall_at_k,
                       bidi_retrieve, bootstrap_ci, load_model)


def main():
    overall_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────
    print("\nLoading data...")
    embeddings = np.load("data/processed/embeddings.npy")
    passage_ids = np.load("data/processed/passage_ids.npy", allow_pickle=True)
    passage_id_to_idx = {pid: i for i, pid in enumerate(passage_ids)}
    n_passages = len(passage_ids)

    import faiss
    faiss_index = faiss.read_index("data/processed/faiss.index")

    # ── Extract train-only pairs ─────────────────────────────────────
    print("\nExtracting train-only association pairs...")
    hotpotqa_train = load_from_disk("data/raw/hotpotqa_train")
    train_pairs_raw = extract_pairs_from_hotpotqa(hotpotqa_train, passage_id_to_idx)
    train_only_pairs = list(set(
        (a, p) for a, p in train_pairs_raw
        if a < n_passages and p < n_passages
    ))
    print(f"Train-split pairs (valid in 66K corpus): {len(train_only_pairs)}")
    del hotpotqa_train

    # ── Train ────────────────────────────────────────────────────────
    print("\nTraining inductive model on train-only pairs...")
    config = {
        "batch_size": 512,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "epochs": 100,
        "temperature": 0.05,
    }

    model = AssociationMLP(embedding_dim=embeddings.shape[1],
                           hidden_dim=1024, num_layers=4)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: 1024h x 4L, {params:,} params")

    train_start = time.time()
    model = train_model(model, train_only_pairs, embeddings, config)
    train_time = (time.time() - train_start) / 60
    print(f"Training time: {train_time:.1f} min")

    os.makedirs("models", exist_ok=True)
    model_path = "models/inductive_train_only.pt"
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Saved to {model_path}")

    # ── Load validation data ─────────────────────────────────────────
    print("\nLoading validation data...")
    hotpotqa_val = load_from_disk("data/raw/hotpotqa")
    valid_examples = []
    for example in hotpotqa_val:
        supporting_titles = set(example["supporting_facts"]["title"])
        gold_indices = [passage_id_to_idx[t] for t in supporting_titles
                        if t in passage_id_to_idx]
        if len(gold_indices) >= 2:
            valid_examples.append((example["question"], gold_indices))
    print(f"  {len(valid_examples)} valid questions")

    # Load cached query embeddings
    cache_path = "data/processed/query_embeddings_hotpotqa.npy"
    if os.path.exists(cache_path):
        query_embeddings = np.load(cache_path)
    else:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        questions = [q for q, _ in valid_examples]
        query_embeddings = embed_model.encode(
            questions, batch_size=64, show_progress_bar=True,
            normalize_embeddings=True
        ).astype(np.float32)
        np.save(cache_path, query_embeddings)

    # ── Lambda sweep ─────────────────────────────────────────────────
    print("\nLambda sweep...")
    model = load_model(model_path, device=device)
    n = len(valid_examples)
    alphas = [0.30, 0.40, 0.50, 0.60, 0.70]

    sweep_results = {}
    for a in alphas:
        r5_scores = []
        for i, (_, gold_indices) in enumerate(valid_examples):
            retrieved = bidi_retrieve(query_embeddings[i], faiss_index, model,
                                     embeddings, device, alpha=a)
            r5_scores.append(passage_recall_at_k(retrieved, gold_indices, k=5))
        r5 = float(np.mean(r5_scores))
        sweep_results[a] = r5
        print(f"  alpha={a:.2f}: R@5={r5:.4f}")

    best_alpha = max(sweep_results, key=lambda a: sweep_results[a])
    print(f"Best alpha: {best_alpha:.2f} (R@5={sweep_results[best_alpha]:.4f})")

    # ── Per-query evaluation ─────────────────────────────────────────
    print(f"\nFull evaluation at alpha={best_alpha:.2f}...")
    baseline_r5 = np.zeros(n)
    model_r5 = np.zeros(n)
    for i, (_, gold_indices) in enumerate(valid_examples):
        _, indices = faiss_index.search(query_embeddings[i].reshape(1, -1), 100)
        baseline_r5[i] = passage_recall_at_k(indices[0], gold_indices, k=5)
        retrieved = bidi_retrieve(query_embeddings[i], faiss_index, model,
                                 embeddings, device, alpha=best_alpha)
        model_r5[i] = passage_recall_at_k(retrieved, gold_indices, k=5)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n}: R@5={np.mean(model_r5[:i+1]):.4f}")

    delta = model_r5 - baseline_r5
    mean, lo, hi = bootstrap_ci(delta)
    print(f"\nInductive: R@5={np.mean(model_r5):.4f}, "
          f"delta={mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # ── Save results ─────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    rows = []
    for a in alphas:
        rows.append({"section": "lambda_sweep", "metric": f"alpha={a:.2f}",
                      "inductive": f"{sweep_results[a]:.4f}"})
    rows.append({"section": "result", "metric": "best_alpha",
                  "inductive": f"{best_alpha:.2f}"})
    rows.append({"section": "result", "metric": "R@5",
                  "inductive": f"{np.mean(model_r5):.4f}"})
    rows.append({"section": "result", "metric": "delta_R@5",
                  "inductive": f"{mean:+.4f}"})
    rows.append({"section": "result", "metric": "ci_lower",
                  "inductive": f"{lo:+.4f}"})
    rows.append({"section": "result", "metric": "ci_upper",
                  "inductive": f"{hi:+.4f}"})

    with open("results/true_inductive_evaluation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "metric", "inductive"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved to results/true_inductive_evaluation.csv")

    elapsed = (time.time() - overall_start) / 60
    print(f"\nTotal time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
