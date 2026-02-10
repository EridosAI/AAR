# Data

AAR uses two multi-hop QA datasets. Both are downloaded automatically by the data preparation script.

## Automatic Setup

```bash
python -c "from src.utils import prepare_data; prepare_data()"
```

This will:
1. Download HotpotQA (distractor setting, train + validation splits) from HuggingFace
2. Download MuSiQue (answerable, validation split) from HuggingFace
3. Extract ~66,581 unique passages from HotpotQA
4. Embed all passages with BGE-large-en-v1.5 (1024-dimensional, L2-normalised)
5. Build a FAISS IndexFlatIP index for exact cosine similarity search

## Manual Setup

If you prefer to download datasets manually:

**HotpotQA** (Yang et al., 2018):
```python
from datasets import load_dataset
hotpotqa_val = load_dataset("hotpot_qa", "distractor", split="validation")
hotpotqa_train = load_dataset("hotpot_qa", "distractor", split="train")
```

**MuSiQue** (Trivedi et al., 2022):
```python
from datasets import load_dataset
musique = load_dataset("bdsaglam/musique", "answerable", split="validation")
```

## Directory Structure After Preparation

```
data/
├── raw/
│   ├── hotpotqa/           # HotpotQA validation (7,405 questions)
│   ├── hotpotqa_train/     # HotpotQA training (90,447 questions)
│   └── musique/            # MuSiQue validation (2,417 questions)
└── processed/
    ├── passage_ids.npy     # Passage titles, shape (66581,)
    ├── embeddings.npy      # BGE-large-en-v1.5 embeddings, shape (66581, 1024)
    └── faiss.index         # FAISS IndexFlatIP for cosine retrieval
```

## Embedding Model

We use [BGE-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) (Xiao et al., 2024) via the `sentence-transformers` library. All embeddings are L2-normalised, so inner product equals cosine similarity.
