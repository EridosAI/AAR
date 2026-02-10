"""Shared utilities for AAR: data loading, metrics, retrieval helpers."""

import os
import numpy as np
import torch
import faiss
from datasets import load_from_disk, load_dataset
from sentence_transformers import SentenceTransformer

from src.model import AssociationMLP


# ── Metrics ──────────────────────────────────────────────────────────

def passage_recall_at_k(retrieved_indices, gold_indices, k=5):
    """Fraction of gold passages found in the top-k retrieved passages."""
    retrieved_set = set(int(i) for i in retrieved_indices[:k])
    gold_set = set(int(i) for i in gold_indices)
    if len(gold_set) == 0:
        return 1.0
    return len(retrieved_set & gold_set) / len(gold_set)


def bootstrap_ci(scores, n_boot=10000, ci=0.95):
    """Paired bootstrap confidence interval. Returns (mean, lower, upper)."""
    scores = np.array(scores)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(scores, size=len(scores), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return (float(np.mean(scores)),
            float(np.percentile(boot_means, 100 * alpha)),
            float(np.percentile(boot_means, 100 * (1 - alpha))))


# ── Data loading ─────────────────────────────────────────────────────

def download_datasets(data_dir="data/raw"):
    """Download HotpotQA and MuSiQue datasets."""
    os.makedirs(data_dir, exist_ok=True)

    print("Downloading HotpotQA (distractor, validation)...")
    hotpotqa = load_dataset("hotpot_qa", "distractor", split="validation")
    hotpotqa.save_to_disk(os.path.join(data_dir, "hotpotqa"))
    print(f"  Saved {len(hotpotqa)} examples")

    print("Downloading HotpotQA (distractor, train)...")
    hotpotqa_train = load_dataset("hotpot_qa", "distractor", split="train")
    hotpotqa_train.save_to_disk(os.path.join(data_dir, "hotpotqa_train"))
    print(f"  Saved {len(hotpotqa_train)} examples")

    print("Downloading MuSiQue (validation)...")
    musique = load_dataset("bdsaglam/musique", "answerable", split="validation")
    musique.save_to_disk(os.path.join(data_dir, "musique"))
    print(f"  Saved {len(musique)} examples")

    print("Done.")


def extract_passages_hotpotqa(dataset):
    """Extract unique passages from HotpotQA. Returns dict: title -> text."""
    passages = {}
    for example in dataset:
        titles = example["context"]["title"]
        sentences_list = example["context"]["sentences"]
        for title, sentences in zip(titles, sentences_list):
            text = " ".join(sentences)
            if title not in passages:
                passages[title] = text
    return passages


def embed_passages(passages, model_name="BAAI/bge-large-en-v1.5", batch_size=64):
    """Embed all passages with a sentence transformer. Returns (ids, embeddings)."""
    model = SentenceTransformer(model_name)
    ids = list(passages.keys())
    texts = [passages[pid] for pid in ids]
    print(f"Embedding {len(texts)} passages with {model_name}...")
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(ids), np.array(embeddings, dtype=np.float32)


def build_faiss_index(embeddings):
    """Build FAISS IndexFlatIP (inner product on normalised vectors = cosine)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def prepare_data(data_dir="data/raw", processed_dir="data/processed"):
    """Full data pipeline: download, extract passages, embed, build index."""
    download_datasets(data_dir)
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading HotpotQA from disk...")
    hotpotqa = load_from_disk(os.path.join(data_dir, "hotpotqa"))
    passages = extract_passages_hotpotqa(hotpotqa)
    print(f"Extracted {len(passages)} unique passages")

    passage_ids, embeddings = embed_passages(passages)
    np.save(os.path.join(processed_dir, "passage_ids.npy"), passage_ids)
    np.save(os.path.join(processed_dir, "embeddings.npy"), embeddings)

    index = build_faiss_index(embeddings)
    faiss.write_index(index, os.path.join(processed_dir, "faiss.index"))
    print(f"Saved {len(passage_ids)} passage IDs, embeddings {embeddings.shape}, FAISS index")


# ── Pair extraction ──────────────────────────────────────────────────

def extract_pairs_from_hotpotqa(dataset, passage_id_to_idx):
    """Extract (anchor, positive) association pairs from HotpotQA supporting facts."""
    pairs = set()
    for example in dataset:
        supporting_titles = set(example["supporting_facts"]["title"])
        supporting_idxs = [passage_id_to_idx[t] for t in supporting_titles
                           if t in passage_id_to_idx]
        if len(supporting_idxs) < 2:
            continue
        for i in range(len(supporting_idxs)):
            for j in range(len(supporting_idxs)):
                if i != j:
                    pairs.add((supporting_idxs[i], supporting_idxs[j]))
    return list(pairs)


# ── Retrieval ────────────────────────────────────────────────────────

def bidi_retrieve(query_emb, faiss_index, model, embeddings, device,
                  alpha=0.50, initial_k=100):
    """Bi-directional AAR retrieval.

    score(q, p) = (1 - alpha) * cos(q, p) + alpha * 0.5 * [f(q).p + f(p).q]

    Returns reranked passage indices (np.ndarray of length initial_k).
    """
    distances, indices = faiss_index.search(query_emb.reshape(1, -1), initial_k)
    indices = indices[0]
    cosine_scores = distances[0]

    query_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0).to(device)
    cand_embs = torch.tensor(embeddings[indices], dtype=torch.float32).to(device)

    with torch.no_grad():
        assoc_query = model(query_tensor)  # f(q)
        forward_scores = torch.mm(assoc_query, cand_embs.t()).squeeze(0)  # f(q).p
        assoc_cands = model(cand_embs)  # f(p)
        backward_scores = torch.mm(assoc_cands, query_tensor.t()).squeeze(1)  # f(p).q
        assoc_scores = ((forward_scores + backward_scores) / 2).cpu().numpy()

    combined = (1 - alpha) * cosine_scores + alpha * assoc_scores
    order = np.argsort(-combined)
    return indices[order]


# ── Model loading ────────────────────────────────────────────────────

def load_model(model_path, device="cuda", embedding_dim=1024, hidden_dim=1024,
               num_layers=4):
    """Load a trained AssociationMLP from a checkpoint."""
    model = AssociationMLP(
        embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers,
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {hidden_dim}h x {num_layers}L, {params:,} params")
    return model
