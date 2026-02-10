"""Evaluate AAR retrieval on HotpotQA or MuSiQue.

Runs the full pipeline: dense baseline, then AAR reranking at one or more
lambda values. Reports R@5, R@10, R@20, easy/hard split, and bootstrap CIs.

Usage:
    python -m src.evaluate --model models/association_mlp.pt
    python -m src.evaluate --model models/association_mlp.pt --alpha 0.60
    python -m src.evaluate --model models/association_mlp.pt --alpha-sweep
"""

import os
import argparse
import csv
import numpy as np
import torch
import faiss
from datasets import load_from_disk

from src.utils import (passage_recall_at_k, bidi_retrieve, bootstrap_ci,
                       load_model)


def evaluate(model_path, dataset_name="hotpotqa", alpha=0.50,
             alpha_sweep=False, processed_dir="data/processed"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    embeddings = np.load(os.path.join(processed_dir, "embeddings.npy"))
    passage_ids = np.load(os.path.join(processed_dir, "passage_ids.npy"),
                          allow_pickle=True)
    passage_id_to_idx = {pid: i for i, pid in enumerate(passage_ids)}
    faiss_index = faiss.read_index(os.path.join(processed_dir, "faiss.index"))

    # Load dataset
    print(f"Loading {dataset_name}...")
    dataset = load_from_disk(f"data/raw/{dataset_name}")
    valid_examples = []
    for example in dataset:
        supporting_titles = set(example["supporting_facts"]["title"])
        gold_indices = [passage_id_to_idx[t] for t in supporting_titles
                        if t in passage_id_to_idx]
        if len(gold_indices) >= 2:
            valid_examples.append((example["question"], gold_indices))
    print(f"  {len(valid_examples)} valid questions")

    # Load cached query embeddings
    cache_path = os.path.join(processed_dir, f"query_embeddings_{dataset_name}.npy")
    if os.path.exists(cache_path):
        query_embeddings = np.load(cache_path)
    else:
        print("Encoding queries (will cache for future runs)...")
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        questions = [q for q, _ in valid_examples]
        query_embeddings = embed_model.encode(
            questions, batch_size=64, show_progress_bar=True,
            normalize_embeddings=True
        ).astype(np.float32)
        np.save(cache_path, query_embeddings)

    n = len(valid_examples)

    # Dense baseline
    print("\nDense baseline...")
    baseline_scores = {k: [] for k in [5, 10, 20]}
    for i, (_, gold_indices) in enumerate(valid_examples):
        _, indices = faiss_index.search(query_embeddings[i].reshape(1, -1), 100)
        for k in [5, 10, 20]:
            baseline_scores[k].append(
                passage_recall_at_k(indices[0], gold_indices, k=k))
    baseline = {k: float(np.mean(v)) for k, v in baseline_scores.items()}
    print(f"  R@5={baseline[5]:.4f}, R@10={baseline[10]:.4f}, R@20={baseline[20]:.4f}")

    # Load model
    model = load_model(model_path, device=device)

    # Lambda sweep or single alpha
    alphas = [0.30, 0.40, 0.50, 0.60, 0.70] if alpha_sweep else [alpha]

    for a in alphas:
        print(f"\nAAR (alpha={a:.2f})...")
        r5_arr = np.zeros(n)
        scores = {k: [] for k in [5, 10, 20]}

        for i, (_, gold_indices) in enumerate(valid_examples):
            retrieved = bidi_retrieve(query_embeddings[i], faiss_index, model,
                                     embeddings, device, alpha=a)
            for k in [5, 10, 20]:
                scores[k].append(
                    passage_recall_at_k(retrieved, gold_indices, k=k))
            r5_arr[i] = scores[5][-1]
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{n}: R@5={np.mean(r5_arr[:i+1]):.4f}")

        results = {k: float(np.mean(v)) for k, v in scores.items()}
        print(f"  R@5={results[5]:.4f}, R@10={results[10]:.4f}, R@20={results[20]:.4f}")
        print(f"  Delta R@5: {results[5] - baseline[5]:+.4f}")

        # Bootstrap CI
        baseline_r5_arr = np.array(baseline_scores[5])
        delta = r5_arr - baseline_r5_arr
        mean, lo, hi = bootstrap_ci(delta)
        print(f"  95% CI: [{lo:+.4f}, {hi:+.4f}]")

        # Easy/hard split
        easy_idx = [i for i in range(n) if baseline_r5_arr[i] == 1.0]
        hard_idx = [i for i in range(n) if baseline_r5_arr[i] < 1.0]
        if hard_idx:
            hard_base = float(np.mean(baseline_r5_arr[hard_idx]))
            hard_model = float(np.mean(r5_arr[hard_idx]))
            print(f"  Easy (n={len(easy_idx)}): delta={np.mean(r5_arr[easy_idx]) - 1.0:+.4f}")
            print(f"  Hard (n={len(hard_idx)}): base={hard_base:.4f}, "
                  f"model={hard_model:.4f}, delta={hard_model - hard_base:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AAR retrieval")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", default="hotpotqa",
                        choices=["hotpotqa", "musique"])
    parser.add_argument("--alpha", type=float, default=0.50,
                        help="Scoring blend parameter lambda")
    parser.add_argument("--alpha-sweep", action="store_true",
                        help="Sweep alpha from 0.30 to 0.70")
    args = parser.parse_args()
    evaluate(args.model, dataset_name=args.dataset, alpha=args.alpha,
             alpha_sweep=args.alpha_sweep)
