"""Train the AAR association model (transductive setting).

Trains a 4-layer MLP with CLIP-style in-batch contrastive loss on passage
co-occurrence pairs from HotpotQA. By default, uses combined train+validation
association pairs (~20,742 pairs) for the transductive setting described in
the paper. Pass --no-train-split to use validation pairs only.

Usage:
    python -m src.train                          # transductive (combined)
    python -m src.train --no-train-split         # val-only pairs
    python -m src.train --epochs 200 --batch-size 512
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.model import AssociationMLP
from src.utils import extract_pairs_from_hotpotqa


class PairDataset(Dataset):
    """Dataset of (anchor_embedding, positive_embedding) pairs."""

    def __init__(self, pairs, embeddings_tensor):
        self.pairs = pairs
        self.embeddings = embeddings_tensor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor_idx, positive_idx = self.pairs[idx]
        return self.embeddings[anchor_idx], self.embeddings[positive_idx]


def clip_loss(anchor_transformed, positives, temperature=0.05):
    """CLIP-style symmetric in-batch contrastive loss.

    With batch size B, each positive pair is contrasted against B-1 in-batch
    negatives via a B x B similarity matrix.
    """
    logits = torch.mm(anchor_transformed, positives.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return (loss_a + loss_b) / 2


def train_model(model, pairs, embeddings, config):
    """Training loop with CLIP-style in-batch negatives."""
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    dataset = PairDataset(pairs, embeddings_tensor)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    print(f"Batch size: {config['batch_size']} -> {config['batch_size'] - 1} in-batch negatives")
    model = model.to(device)

    best_acc = 0
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for anchors, positives in loader:
            anchors = anchors.to(device)
            positives = positives.to(device)
            anchor_transformed = model(anchors)
            loss = clip_loss(anchor_transformed, positives, temperature=config["temperature"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            B = anchors.size(0)
            total_loss += loss.item() * B
            with torch.no_grad():
                sim = torch.mm(anchor_transformed, positives.t())
                preds = sim.argmax(dim=1)
                labels = torch.arange(B, device=device)
                correct += (preds == labels).sum().item()
                total += B

        avg_loss = total_loss / len(dataset)
        accuracy = correct / total
        if accuracy > best_acc:
            best_acc = accuracy
            marker = " *"
        else:
            marker = ""

        print(f"Epoch {epoch + 1:3d}/{config['epochs']} -- "
              f"Loss: {avg_loss:.4f} -- Accuracy: {accuracy:.4f}{marker}")
        scheduler.step()

    print(f"Best accuracy: {best_acc:.4f}")
    return model


def run(processed_dir="data/processed", output_dir="models", epochs=100,
        hidden_dim=1024, num_layers=4, batch_size=512, use_train_split=True):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading embeddings...")
    embeddings = np.load(os.path.join(processed_dir, "embeddings.npy"))
    passage_ids = np.load(os.path.join(processed_dir, "passage_ids.npy"),
                          allow_pickle=True)
    passage_id_to_idx = {pid: i for i, pid in enumerate(passage_ids)}

    # Collect pairs from validation set
    from datasets import load_from_disk
    print("Loading validation pairs...")
    hotpotqa_val = load_from_disk("data/raw/hotpotqa")
    val_pairs = extract_pairs_from_hotpotqa(hotpotqa_val, passage_id_to_idx)
    print(f"  Validation pairs: {len(val_pairs)}")

    all_pairs = set(val_pairs)
    if use_train_split and os.path.exists("data/raw/hotpotqa_train"):
        print("Loading training split pairs...")
        hotpotqa_train = load_from_disk("data/raw/hotpotqa_train")
        train_pairs = extract_pairs_from_hotpotqa(hotpotqa_train, passage_id_to_idx)
        valid_train_pairs = [(a, p) for a, p in train_pairs
                             if a < len(passage_ids) and p < len(passage_ids)]
        all_pairs.update(valid_train_pairs)
        print(f"  Training split pairs (valid): {len(valid_train_pairs)}")

    pairs = list(all_pairs)
    print(f"Total unique pairs: {len(pairs)}")

    config = {
        "batch_size": batch_size,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "epochs": epochs,
        "temperature": 0.05,
    }

    model = AssociationMLP(embedding_dim=embeddings.shape[1],
                           hidden_dim=hidden_dim, num_layers=num_layers)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {embeddings.shape[1]}d -> {hidden_dim}h x {num_layers}L, {params:,} params")

    model = train_model(model, pairs, embeddings, config)

    model_path = os.path.join(output_dir, "association_mlp.pt")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Saved model to {model_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AAR association model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--no-train-split", action="store_true",
                        help="Use validation pairs only (not transductive)")
    args = parser.parse_args()
    run(epochs=args.epochs, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        batch_size=args.batch_size, use_train_split=not args.no_train_split)
