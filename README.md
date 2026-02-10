# Association ≠ Similarity: Learning Corpus-Specific Associations for Multi-Hop Retrieval

**Jason Dury** — Eridos AI, Perth, Australia

Dense retrieval ranks passages by embedding similarity to a query, but multi-hop questions require passages that are *associatively related* through shared reasoning chains rather than semantically similar. Association-Augmented Retrieval (AAR) trains a lightweight MLP (4.2M parameters) to learn passage-to-passage associations from co-occurrence annotations using CLIP-style contrastive learning, then reranks dense retrieval results via bi-directional association scoring. AAR is transductive by design: it learns associations over the target corpus, mirroring how RAG systems are deployed in practice. On HotpotQA, AAR improves Recall@5 by +8.6 points without evaluation-set tuning, with a +28.5 point gain on the hardest questions. On MuSiQue it achieves +10.1 points. An inductive variant shows no significant improvement, consistent with corpus-specific co-occurrence learning. The method trains in under two minutes on a single GPU, adds 3.7ms per query, and requires no LLM-based indexing.

**Paper:** [arXiv (forthcoming)]() | **PAM framework:** [Zenodo](https://doi.org/10.5281/zenodo.18595537)

## Key Results

| Setting | Dataset | R@5 | Delta R@5 | 95% CI |
|---------|---------|-----|-----------|--------|
| Dense baseline | HotpotQA | 0.831 | — | — |
| **AAR transductive** | **HotpotQA** | **0.916** | **+8.6** | [+8.1, +9.0] |
| AAR inductive | HotpotQA | 0.832 | +0.1 | [-0.3, +0.5] |
| Dense baseline | MuSiQue | 0.387 | — | — |
| **AAR transductive** | **MuSiQue** | **0.488** | **+10.1** | — |

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- FAISS (`faiss-gpu` or `faiss-cpu`)
- sentence-transformers (for BGE-large-en-v1.5 embeddings)
- datasets (HuggingFace)

```bash
pip install torch faiss-gpu sentence-transformers datasets numpy
```

## Quick Start

### 1. Prepare data

```bash
# Download HotpotQA and MuSiQue, embed passages, build FAISS index
python -c "from src.utils import prepare_data; prepare_data()"
```

This downloads the datasets, extracts ~66K unique passages from HotpotQA, embeds them with BGE-large-en-v1.5, and builds a FAISS index. Takes ~15 minutes (mostly embedding).

### 2. Train (transductive)

```bash
python -m src.train
```

Trains on combined train+validation association pairs (~20,742 pairs). Completes in ~2 minutes on an RTX 4080 Super. Saves to `models/association_mlp.pt`.

### 3. Evaluate

```bash
python -m src.evaluate --model models/association_mlp.pt --alpha 0.50
python -m src.evaluate --model models/association_mlp.pt --alpha-sweep
```

### 4. Train inductive variant

```bash
python -m src.train_true_inductive
```

Trains on training-split pairs only (~8,758), evaluates on validation set.

## Repository Structure

```
AAR/
├── README.md
├── LICENSE
├── paper/
│   └── aar_paper_submission.md       # Full paper (markdown)
├── src/
│   ├── model.py                      # AssociationMLP architecture
│   ├── train.py                      # Main training script (transductive)
│   ├── train_true_inductive.py       # Inductive training script
│   ├── evaluate.py                   # Evaluation / retrieval pipeline
│   └── utils.py                      # Data loading, metrics, retrieval
├── results/
│   ├── retrieval_matched_hp.csv      # Main results (Table 2)
│   ├── true_inductive_evaluation.csv # Inductive evaluation (Table 4)
│   ├── scoring_ablation.csv          # Scoring method ablation (Table 1)
│   ├── bm25_baseline.csv             # BM25 comparison (Table 7)
│   ├── qa_sanity_check.csv           # Downstream QA (Table 8)
│   ├── answer_coverage_matched_hp.csv# Answer coverage (Table 9)
│   ├── candidate_pool_sensitivity.csv# FAISS expansion depth (Table B1)
│   ├── latency_breakdown.csv         # Latency (Table 10)
│   ├── iteration_log.csv             # Development iteration log
│   └── bootstrap_ci.csv             # Bootstrap confidence intervals
├── data/
│   └── README.md                     # Data acquisition instructions
└── models/
    └── README.md                     # Model reproduction instructions
```

## Method Overview

1. **Candidate retrieval:** FAISS top-100 by cosine similarity
2. **Association reranking:** For each candidate, compute blended score:

   `score(q, p) = (1 - lambda) * cos(q, p) + lambda * a(q, p)`

   where `a(q, p) = 0.5 * [f(q) . p + f(p) . q]` is the bi-directional association score and `f` is the trained MLP.

3. Return top-k by blended score.

## Citation

```bibtex
@misc{dury2026aar,
  author       = {Dury, Jason},
  title        = {Association $\neq$ Similarity: Learning Corpus-Specific
                  Associations for Multi-Hop Retrieval},
  year         = {2026},
  note         = {arXiv preprint (forthcoming)}
}

@misc{dury2026pam,
  author       = {Dury, Jason},
  title        = {Predictive Associative Memory: Unified Retrieval, Imagination,
                  and Creative Recombination Through Predictive Traversal of
                  Meaning Space},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18595537}
}
```

## License

MIT
