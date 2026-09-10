"""
Standalone reimplementation of the TANGERINE prediction head, for use when you
already have extracted encoder embeddings and want to skip the ViT-Large
encoder / raw CT volumes entirely.

Mirrors models_vit.py from:
    https://github.com/niccolo246/3D-MAE-MedImaging

Two head variants are provided, matching the two model classes there:
    - LinearHead -> matches `VisionTransformer`    (used by vit_large_patch16_yo)
    - MLPHead    -> matches `VisionTransformerMod` (used by vit_large_patch16_power_2_yo,
                                                      hidden_layers=[64] in the paper)

Both heads are: fc_norm (LayerNorm over 2*embed_dim) -> Linear (or MLP) -> logits.

--------------------------------------------------------------------------
IMPORTANT: which embedding_type did you extract with extract_encoder_embeddings.py?
--------------------------------------------------------------------------
    "class"       : CLS token only                    [B, 1024]   -- NOT what the head expects
    "pooled"      : mean patch token only              [B, 1024]   -- NOT what the head expects
    "concat"      : raw [CLS ; mean-patch]             [B, 2048]   -- fc_norm NOT applied yet
    "pre_logits"  : fc_norm(concat)  (this is the DEFAULT) [B, 2048]   -- fc_norm ALREADY applied

If you have "pre_logits" embeddings -> set apply_fc_norm=False (skip renormalizing).
If you have "concat" embeddings     -> set apply_fc_norm=True  (this script will LayerNorm them).
If you only have "class" or "pooled" -> you don't have what the released head was trained on;
    re-extract with --embedding_type concat or pre_logits.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Head definitions (faithful ports of models_vit.py)
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    """Matches VisionTransformer's global-pool head: fc_norm -> single Linear."""

    def __init__(self, embed_dim: int = 1024, num_classes: int = 1, apply_fc_norm: bool = True):
        super().__init__()
        self.apply_fc_norm = apply_fc_norm
        self.fc_norm = nn.LayerNorm(embed_dim * 2)
        self.head = nn.Linear(embed_dim * 2, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2*embed_dim] concatenated [cls_token ; mean_patch_tokens]
        if self.apply_fc_norm:
            x = self.fc_norm(x)
        return self.head(x)


class MLPHead(nn.Module):
    """Matches VisionTransformerMod's head: fc_norm -> [Linear -> LeakyReLU -> BatchNorm1d]* -> Linear.

    The paper's Cancer-NLST-ACRIN / Cancer-Duke head used hidden_layers=[64].
    """

    def __init__(self, embed_dim: int = 1024, num_classes: int = 1,
                 hidden_layers: Optional[List[int]] = None, apply_fc_norm: bool = True):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64]
        self.apply_fc_norm = apply_fc_norm
        self.fc_norm = nn.LayerNorm(embed_dim * 2)

        mlp_layers = []
        input_dim = embed_dim * 2
        for h in hidden_layers:
            mlp_layers.append(nn.Linear(input_dim, h))
            mlp_layers.append(nn.LeakyReLU())
            mlp_layers.append(nn.BatchNorm1d(h))
            input_dim = h
        mlp_layers.append(nn.Linear(input_dim, num_classes))
        self.head = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_fc_norm:
            x = self.fc_norm(x)
        return self.head(x)


def build_head(head_type: str, embed_dim: int, num_classes: int,
                apply_fc_norm: bool, hidden_layers: Optional[List[int]] = None) -> nn.Module:
    if head_type == "linear":
        return LinearHead(embed_dim=embed_dim, num_classes=num_classes, apply_fc_norm=apply_fc_norm)
    elif head_type == "mlp":
        return MLPHead(embed_dim=embed_dim, num_classes=num_classes,
                        hidden_layers=hidden_layers, apply_fc_norm=apply_fc_norm)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")


def load_head_weights_from_checkpoint(head: nn.Module, checkpoint_path: str) -> None:
    """Load only fc_norm.* and head.* keys from a fine-tuned TANGERINE checkpoint
    (a full ViT state_dict) into this standalone head module.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt
    for key in ("model", "model_state", "state_dict", "model_state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state_dict = ckpt[key]
            break

    relevant = {k: v for k, v in state_dict.items() if k.startswith("fc_norm.") or k.startswith("head.")}
    msg = head.load_state_dict(relevant, strict=False)
    print(f"Loaded head weights from {checkpoint_path}: {msg}")


# ---------------------------------------------------------------------------
# Dataset: loads pre-extracted embeddings (+ optional labels) from a CSV
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """Reads embeddings produced by extract_encoder_embeddings.py.

    Supports either:
      - a single JSON-list column (default output of extract_encoder_embeddings.py,
        e.g. column "Embedding" containing "[0.01, -0.3, ...]"), or
      - flattened numeric columns (if extract_encoder_embeddings.py was run with
        --flatten_columns, e.g. "Embedding_0", "Embedding_1", ...).
    """

    def __init__(self, csv_path: str, embedding_col: str = "Embedding",
                 label_col: Optional[str] = None, flattened: bool = False):
        self.df = pd.read_csv(csv_path)
        self.embedding_col = embedding_col
        self.label_col = label_col
        self.flattened = flattened

        if flattened:
            self.emb_cols = [c for c in self.df.columns if c.startswith(embedding_col + "_")]
            if not self.emb_cols:
                raise ValueError(f"No flattened columns found with prefix '{embedding_col}_'")
        else:
            if embedding_col not in self.df.columns:
                raise ValueError(f"Column '{embedding_col}' not found. Available: {self.df.columns.tolist()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.flattened:
            emb = row[self.emb_cols].to_numpy(dtype=np.float32)
        else:
            emb = np.array(json.loads(row[self.embedding_col]), dtype=np.float32)
        emb = torch.from_numpy(emb)

        if self.label_col is not None:
            label = torch.tensor(row[self.label_col], dtype=torch.float32)
            return emb, label
        return emb, -1  # dummy label for inference-only use


# ---------------------------------------------------------------------------
# Train / predict loops (mirrors the loss/activation logic used by
# main_predict.py and engine_finetune.py)
# ---------------------------------------------------------------------------

def get_criterion(task: str, binary_class_weights: Optional[List[float]] = None):
    if task == "binary":
        if binary_class_weights:
            w = torch.tensor(binary_class_weights)
            return nn.BCEWithLogitsLoss(pos_weight=w)
        return nn.BCEWithLogitsLoss()
    elif task == "regression":
        return nn.MSELoss()
    elif task == "multiclass":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown task: {task}")


def train(head: nn.Module, loader: DataLoader, task: str, device: torch.device,
          epochs: int = 50, lr: float = 1e-3, binary_class_weights: Optional[List[float]] = None):
    head.to(device)
    criterion = get_criterion(task, binary_class_weights)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    for epoch in range(epochs):
        head.train()
        running_loss = 0.0
        for emb, label in loader:
            emb, label = emb.to(device), label.to(device)
            optimizer.zero_grad()
            output = head(emb)

            if task == "binary":
                loss = criterion(output.squeeze(-1), label.float())
            elif task == "regression":
                loss = criterion(output.squeeze(-1), label.float())
            else:  # multiclass
                loss = criterion(output, label.long())

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * emb.size(0)

        print(f"Epoch {epoch + 1}/{epochs} - loss: {running_loss / len(loader.dataset):.4f}")

    return head


@torch.no_grad()
def predict(head: nn.Module, loader: DataLoader, task: str, device: torch.device):
    head.eval()
    head.to(device)
    all_preds = []
    for emb, _ in loader:
        emb = emb.to(device)
        output = head(emb)
        if task == "binary":
            pred = torch.sigmoid(output)
        elif task == "multiclass":
            pred = torch.softmax(output, dim=1)
        else:  # regression: raw output, no activation
            pred = output
        all_preds.extend(pred.cpu().numpy().tolist())
    return all_preds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args_parser():
    p = argparse.ArgumentParser("TANGERINE head on pre-extracted embeddings")
    p.add_argument("--mode", choices=["train", "predict"], required=True)
    p.add_argument("--input_csv", required=True, type=str)
    p.add_argument("--output_csv", default=None, type=str, help="Required for --mode predict")
    p.add_argument("--embedding_col", default="Embedding", type=str)
    p.add_argument("--flattened", action="store_true",
                    help="Set if extract_encoder_embeddings.py was run with --flatten_columns")
    p.add_argument("--label_col", default=None, type=str, help="Required for --mode train")

    p.add_argument("--head_type", choices=["linear", "mlp"], default="linear")
    p.add_argument("--hidden_layers", type=int, nargs="+", default=[64],
                    help="Only used when --head_type mlp")
    p.add_argument("--embed_dim", type=int, default=1024, help="Per-token embed dim (2048 total after concat)")
    p.add_argument("--nb_classes", type=int, default=1)
    p.add_argument("--task", choices=["binary", "multiclass", "regression"], default="binary")

    p.add_argument("--apply_fc_norm", action="store_true",
                    help="Set this if your embeddings are 'concat' (raw, un-normalized). "
                         "Leave unset if your embeddings are 'pre_logits' (already fc_norm'd, the default).")

    p.add_argument("--checkpoint", default=None, type=str,
                    help="Optional: load fc_norm/head weights from a fine-tuned TANGERINE checkpoint")
    p.add_argument("--save_checkpoint", default=None, type=str,
                    help="Optional: where to save the trained head's state_dict")

    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--epochs", default=50, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--binary_class_weights", type=float, nargs="+", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main():
    args = get_args_parser().parse_args()
    device = torch.device(args.device)

    head = build_head(
        head_type=args.head_type,
        embed_dim=args.embed_dim,
        num_classes=args.nb_classes,
        apply_fc_norm=args.apply_fc_norm,
        hidden_layers=args.hidden_layers,
    )

    if args.checkpoint:
        load_head_weights_from_checkpoint(head, args.checkpoint)

    dataset = EmbeddingDataset(
        csv_path=args.input_csv,
        embedding_col=args.embedding_col,
        label_col=args.label_col if args.mode == "train" else None,
        flattened=args.flattened,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(args.mode == "train"))

    if args.mode == "train":
        if args.label_col is None:
            raise ValueError("--label_col is required for --mode train")
        head = train(head, loader, task=args.task, device=device, epochs=args.epochs,
                     lr=args.lr, binary_class_weights=args.binary_class_weights)
        if args.save_checkpoint:
            Path(args.save_checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save(head.state_dict(), args.save_checkpoint)
            print(f"Saved head checkpoint to {args.save_checkpoint}")

    if args.mode == "predict" or args.mode == "train":
        preds = predict(head, loader, task=args.task, device=device)
        if args.output_csv:
            df = pd.read_csv(args.input_csv)
            df["Predictions"] = pd.Series(preds)
            df.to_csv(args.output_csv, index=False)
            print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()


# EXAMPLES 
# Linear head for binary classication:
# python prediction_head.py --mode train \
#   --input_csv embeddings_train.csv --label_col Label \
#   --head_type linear --nb_classes 1 --task binary \
#   --save_checkpoint head_ckpt.pt

# MLP Variant:
# python prediction_head.py --mode train \
#    --input_csv embeddings_train.csv --label_col Label \
#    --head_type mlp --hidden_layers 64 --nb_classes 1 --task binary

# Previously finetuned checkpoint's head weight
#python tangerine_head.py --mode predict \
#    --input_csv embeddings_test.csv --output_csv preds.csv \
#    --head_type linear --nb_classes 1 --task binary \
#   --checkpoint /path/to/finetuned_tangerine_checkpoint.pth