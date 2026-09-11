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

If you have "pre_logits" embeddings -> leave --apply_fc_norm unset.
If you have "concat" embeddings     -> pass --apply_fc_norm (this script will LayerNorm them,
    with fresh learnable weights trained jointly with the head, exactly as TANGERINE does when
    fine-tuning end-to-end -- unless you load an existing fc_norm via --checkpoint).

--------------------------------------------------------------------------
--mode: train / evaluate / predict
--------------------------------------------------------------------------
    train    : fits the head on --input_csv (must have --label_col). If --val_csv is given,
               validation loss is tracked each epoch and the BEST checkpoint (lowest val loss)
               is kept -- mirroring TANGERINE's own early-stopping/model-selection strategy.
               If --test_csv is also given, final metrics (AUROC/AUPRC, accuracy, or MSE/R2,
               depending on --task) are reported on that held-out set at the end.
    evaluate : loads a previously trained/saved head (--checkpoint) and reports metrics on
               --input_csv (must have --label_col). Use this for a clean held-out test-set run.
    predict  : loads a previously trained/saved head (--checkpoint) and just writes predictions
               for --input_csv (no labels required).

Do NOT fit the head and report metrics on the same cohort -- that is data leakage, not a
performance evaluation. Split into train/val/test (the TANGERINE paper used ~50:10:40, or
60:10:30 for cohorts with very few positive cases) and only report metrics from --mode evaluate
on a held-out split the head never saw during training.
"""

import argparse
import copy
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


class FrozenANNHead(nn.Module):
    """Matches the paper's TANGERINE-Frozen classifier ("Frozen embeddings demonstrate
    strong representational power" / Methods): "The ANN architecture consisted of two
    fully connected layers with hidden dimensions [128, 32], followed by a sigmoid
    output layer for binary classification."

    Two things the paper does NOT specify, called out here rather than silently guessed:
      - The activation between the two hidden layers. Defaults to ReLU (configurable
        via activation_cls) -- the fine-tuning-time MLPHead above explicitly uses
        LeakyReLU, but the paper's frozen-ANN description doesn't state one.
      - Whether fc_norm/LayerNorm is applied to the input embedding for this specific
        classifier (unlike the fine-tuning heads, where it's explicitly described).
        Defaults to apply_fc_norm=False here, i.e. a literal reading of "two fully
        connected layers" with no normalisation step mentioned; set True to try it
        with LayerNorm anyway.

    Outputs raw logits (no sigmoid baked into forward()) for consistency with this
    codebase's train/predict functions, which apply sigmoid/softmax themselves --
    NOT because the paper's classifier omits the sigmoid (it explicitly has one).
    """

    def __init__(self, embed_dim: int = 1024, num_classes: int = 1,
                 hidden_dims: Optional[List[int]] = None, apply_fc_norm: bool = False,
                 activation_cls=nn.ReLU):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 32]
        self.apply_fc_norm = apply_fc_norm
        self.fc_norm = nn.LayerNorm(embed_dim * 2)

        layers = []
        input_dim = embed_dim * 2
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation_cls())
            input_dim = h
        layers.append(nn.Linear(input_dim, num_classes))
        self.head = nn.Sequential(*layers)

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
    elif head_type == "frozen_ann":
        return FrozenANNHead(embed_dim=embed_dim, num_classes=num_classes,
                              hidden_dims=hidden_layers, apply_fc_norm=apply_fc_norm)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")


def load_head_weights_from_checkpoint(head: nn.Module, checkpoint_path: str) -> None:
    """Load fc_norm.* and head.* keys from a checkpoint into this standalone head module.
    Works both with checkpoints saved by this script (head.state_dict() directly) and with
    full fine-tuned TANGERINE ViT checkpoints (only the relevant keys are picked out).
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

        if label_col is not None:
            n_before = len(self.df)
            self.df = self.df[self.df[label_col].notna()].reset_index(drop=True)
            n_dropped = n_before - len(self.df)
            if n_dropped > 0:
                print(f"EmbeddingDataset({csv_path}): dropped {n_dropped}/{n_before} rows missing "
                      f"'{label_col}' (expected if this CSV holds multiple outcomes, e.g. a "
                      f"category-level split from make_splits.py).")

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
# Train / predict / evaluate (mirrors the loss/activation logic used by
# main_predict.py and engine_finetune.py, plus proper val-based checkpointing)
# ---------------------------------------------------------------------------

class NPZEmbeddingDataset(Dataset):
    """Reads SID + label from a (small, label-only) CSV, and looks up each SID's
    embedding from a single consolidated embeddings store -- a .npz with 'sids'
    (str array) and 'embeddings' (float32 [N, D] array), as produced by
    build_embeddings_csv.py. This is the recommended path: split CSVs stay
    SID+outcome only, embeddings are never duplicated into them.

    The whole store is loaded into memory once per Dataset instantiation (cheap:
    a few thousand subjects x 2048-d float32 is tens of MB, not a concern).
    """

    def __init__(self, csv_path: str, embeddings_store: str, id_col: str = "SID",
                 label_col: Optional[str] = None):
        self.df = pd.read_csv(csv_path)
        self.id_col = id_col
        self.label_col = label_col

        if label_col is not None:
            n_before = len(self.df)
            self.df = self.df[self.df[label_col].notna()].reset_index(drop=True)
            n_dropped = n_before - len(self.df)
            if n_dropped > 0:
                print(f"NPZEmbeddingDataset({csv_path}): dropped {n_dropped}/{n_before} rows missing "
                      f"'{label_col}' (expected if this CSV holds multiple outcomes, e.g. a "
                      f"category-level split from make_splits.py).")

        store = np.load(embeddings_store)
        store_sids = store["sids"].astype(str)
        self.store_embeddings = store["embeddings"].astype(np.float32)
        self.sid_to_row = {sid: i for i, sid in enumerate(store_sids)}

        csv_sids = self.df[id_col].astype(str)
        missing = [sid for sid in csv_sids if sid not in self.sid_to_row]
        if missing:
            raise ValueError(f"{len(missing)} SIDs in {csv_path} not found in embeddings store "
                              f"{embeddings_store}, e.g.: {missing[:5]}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = str(row[self.id_col])
        emb = torch.from_numpy(self.store_embeddings[self.sid_to_row[sid]])

        if self.label_col is not None:
            label = torch.tensor(row[self.label_col], dtype=torch.float32)
            return emb, label
        return emb, -1  # dummy label for inference-only use


class PathEmbeddingDataset(Dataset):
    """Reads SID + EmbeddingPath + label from a CSV (as produced by make_splits.py),
    and loads each subject's embedding directly from their individual .npz file at
    __getitem__ time -- no consolidated store needed. Embeddings are cached in
    memory after first load (per Dataset instance), so a file is only read from
    disk once even though the same Dataset/DataLoader is reused across epochs.
    """

    def __init__(self, csv_path: str, embedding_path_col: str = "EmbeddingPath",
                 npz_key: str = "image", label_col: Optional[str] = None):
        self.df = pd.read_csv(csv_path)
        self.embedding_path_col = embedding_path_col
        self.npz_key = npz_key
        self.label_col = label_col
        self._cache = {}

        if label_col is not None:
            n_before = len(self.df)
            self.df = self.df[self.df[label_col].notna()].reset_index(drop=True)
            n_dropped = n_before - len(self.df)
            if n_dropped > 0:
                print(f"PathEmbeddingDataset({csv_path}): dropped {n_dropped}/{n_before} rows missing "
                      f"'{label_col}' (expected if this CSV holds multiple outcomes, e.g. a "
                      f"category-level split from make_splits.py).")

    def __len__(self):
        return len(self.df)

    def _load(self, path: str) -> np.ndarray:
        if path not in self._cache:
            data = np.load(path, allow_pickle=True)
            if self.npz_key not in data.files:
                raise KeyError(f"'{self.npz_key}' not found in {path}. Available keys: {data.files}")
            emb = np.asarray(data[self.npz_key]).squeeze()
            if emb.ndim != 1:
                raise ValueError(f"Embedding at {path} has shape {emb.shape}, expected 1D.")
            self._cache[path] = emb.astype(np.float32)
        return self._cache[path]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = torch.from_numpy(self._load(row[self.embedding_path_col]))

        if self.label_col is not None:
            label = torch.tensor(row[self.label_col], dtype=torch.float32)
            return emb, label
        return emb, -1  # dummy label for inference-only use


def make_embedding_dataset(csv_path: str, label_col: Optional[str], embeddings_store: Optional[str] = None,
                            id_col: str = "SID", embedding_col: str = "Embedding",
                            flattened: bool = False, embedding_path_col: str = "EmbeddingPath",
                            npz_key: str = "image") -> Dataset:
    """Picks the right Dataset for how this CSV stores embedding info, in priority order:
      1. embeddings_store given -> NPZEmbeddingDataset (legacy: one consolidated store, SID lookup).
      2. CSV has an embedding_path_col -> PathEmbeddingDataset (recommended: load per-subject .npz
         directly from the path make_splits.py already resolved -- no separate store needed).
      3. else -> EmbeddingDataset (legacy: embedding baked into the CSV as a JSON column).
    """
    if embeddings_store:
        return NPZEmbeddingDataset(csv_path, embeddings_store, id_col=id_col, label_col=label_col)

    header = pd.read_csv(csv_path, nrows=0).columns
    if embedding_path_col and embedding_path_col in header:
        return PathEmbeddingDataset(csv_path, embedding_path_col=embedding_path_col,
                                     npz_key=npz_key, label_col=label_col)

    return EmbeddingDataset(csv_path, embedding_col=embedding_col, label_col=label_col, flattened=flattened)


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


def _compute_loss(criterion, output, label, task):
    if task in ("binary", "regression"):
        return criterion(output.squeeze(-1), label.float())
    else:  # multiclass
        return criterion(output, label.long())


@torch.no_grad()
def evaluate_loss(head: nn.Module, loader: DataLoader, criterion, task: str, device: torch.device) -> float:
    head.eval()
    running_loss = 0.0
    for emb, label in loader:
        emb, label = emb.to(device), label.to(device)
        output = head(emb)
        loss = _compute_loss(criterion, output, label, task)
        running_loss += loss.item() * emb.size(0)
    return running_loss / len(loader.dataset)


def train(head: nn.Module, train_loader: DataLoader, task: str, device: torch.device,
          val_loader: Optional[DataLoader] = None, epochs: int = 50, lr: float = 1e-3,
          binary_class_weights: Optional[List[float]] = None):
    head.to(device)
    criterion = get_criterion(task, binary_class_weights)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0

    for epoch in range(epochs):
        head.train()
        running_loss = 0.0
        for emb, label in train_loader:
            emb, label = emb.to(device), label.to(device)
            optimizer.zero_grad()
            output = head(emb)
            loss = _compute_loss(criterion, output, label, task)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * emb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        if val_loader is not None:
            val_loss = evaluate_loss(head, val_loader, criterion, task, device)
            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(head.state_dict())
                best_epoch = epoch + 1
                marker = "  <- best so far"
            print(f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}{marker}")
        else:
            print(f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f}  (no val_csv given: "
                  f"NOT doing early stopping / best-checkpoint selection)")

    if val_loader is not None:
        print(f"Restoring best checkpoint from epoch {best_epoch} (val_loss={best_val_loss:.4f})")
        head.load_state_dict(best_state)

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


def compute_metrics(preds: List, labels: List, task: str) -> dict:
    from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                                  mean_squared_error, r2_score)
    preds = np.array(preds)
    labels = np.array(labels)
    metrics = {}

    if task == "binary":
        preds_flat = preds.reshape(-1)
        uniques = sorted(np.unique(labels[~np.isnan(labels)]).tolist()) if np.issubdtype(labels.dtype, np.floating) \
            else sorted(np.unique(labels).tolist())
        if len(uniques) > 2:
            raise ValueError(
                f"task='binary' but labels have {len(uniques)} distinct values: {uniques}. "
                f"This column needs recoding down to exactly 2 values before training -- see the "
                f"\"recode\" option on this outcome in make_splits.py's OUTCOMES list.")
        metrics["AUROC"] = roc_auc_score(labels, preds_flat)
        metrics["AUPRC"] = average_precision_score(labels, preds_flat)
    elif task == "multiclass":
        pred_classes = preds.argmax(axis=1)
        metrics["Accuracy"] = accuracy_score(labels, pred_classes)
        try:
            metrics["AUROC_macro_ovr"] = roc_auc_score(labels, preds, multi_class="ovr", average="macro")
        except ValueError:
            pass  # e.g. a class missing from this split
    else:  # regression
        preds_flat = preds.reshape(-1)
        mse = mean_squared_error(labels, preds_flat)
        metrics["MSE"] = mse
        metrics["RMSE"] = mse ** 0.5
        metrics["MAE"] = np.mean(np.abs(labels - preds_flat))
        metrics["R2"] = r2_score(labels, preds_flat)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args_parser():
    p = argparse.ArgumentParser("TANGERINE head on pre-extracted embeddings")
    p.add_argument("--mode", choices=["train", "evaluate", "predict"], required=True)

    p.add_argument("--input_csv", required=True, type=str,
                    help="train: training split. evaluate: held-out split with labels. predict: any cohort.")
    p.add_argument("--val_csv", default=None, type=str,
                    help="mode=train only: held-out validation split, used for early stopping / "
                         "best-checkpoint selection (recommended -- mirrors TANGERINE's own protocol).")
    p.add_argument("--test_csv", default=None, type=str,
                    help="mode=train only: held-out test split; final metrics reported on this set "
                         "after training finishes (never used to select the checkpoint).")
    p.add_argument("--output_csv", default=None, type=str, help="Where to write predictions (predict/train/evaluate)")

    p.add_argument("--embedding_col", default="Embedding", type=str)
    p.add_argument("--flattened", action="store_true",
                    help="Set if extract_encoder_embeddings.py was run with --flatten_columns")
    p.add_argument("--embeddings_store", default=None, type=str,
                    help="Optional: path to a consolidated .npz embeddings store (sids + embeddings arrays, "
                         "from build_embeddings_csv.py). If given, --input_csv/--val_csv etc. only need SID+label "
                         "columns, and embeddings are looked up by SID instead of read from --embedding_col.")
    p.add_argument("--id_col", default="SID", type=str, help="SID column name, only used with --embeddings_store")
    p.add_argument("--label_col", default=None, type=str,
                    help="Required for mode=train and mode=evaluate. Also used for val_csv/test_csv if given.")

    p.add_argument("--head_type", choices=["linear", "mlp", "frozen_ann"], default="linear")
    p.add_argument("--hidden_layers", type=int, nargs="+", default=None,
                    help="Only used for --head_type mlp/frozen_ann. Leave unset to use each head's own "
                         "paper-matched default: [64] for mlp, [128, 32] for frozen_ann.")
    p.add_argument("--embed_dim", type=int, default=1024, help="Per-token embed dim (2048 total after concat)")
    p.add_argument("--nb_classes", type=int, default=1)
    p.add_argument("--task", choices=["binary", "multiclass", "regression"], default="binary")

    p.add_argument("--apply_fc_norm", action="store_true",
                    help="Set this if your embeddings are 'concat' (raw, un-normalized). "
                         "Leave unset if your embeddings are 'pre_logits' (already fc_norm'd, the default).")

    p.add_argument("--checkpoint", default=None, type=str,
                    help="mode=evaluate/predict: required, load a trained head's weights. "
                         "mode=train: optional, warm-start fc_norm/head from an existing checkpoint.")
    p.add_argument("--save_checkpoint", default=None, type=str,
                    help="mode=train: where to save the best head state_dict")

    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--epochs", default=50, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--binary_class_weights", type=float, nargs="+", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def _make_loader(csv_path, args, label_col, shuffle):
    ds = make_embedding_dataset(csv_path, label_col=label_col, embeddings_store=args.embeddings_store,
                                 id_col=args.id_col, embedding_col=args.embedding_col, flattened=args.flattened)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)


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

    # ---------------- train ----------------
    if args.mode == "train":
        if args.label_col is None:
            raise ValueError("--label_col is required for --mode train")

        train_loader = _make_loader(args.input_csv, args, args.label_col, shuffle=True)
        val_loader = _make_loader(args.val_csv, args, args.label_col, shuffle=False) if args.val_csv else None
        if val_loader is None:
            print("WARNING: no --val_csv given. Training will run for the full --epochs with no "
                  "early stopping / best-checkpoint selection -- risk of overfitting the head. "
                  "Strongly recommended to pass --val_csv.")

        head = train(head, train_loader, task=args.task, device=device, val_loader=val_loader,
                     epochs=args.epochs, lr=args.lr, binary_class_weights=args.binary_class_weights)

        if args.save_checkpoint:
            Path(args.save_checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save(head.state_dict(), args.save_checkpoint)
            print(f"Saved head checkpoint to {args.save_checkpoint}")

        if args.test_csv:
            test_loader = _make_loader(args.test_csv, args, args.label_col, shuffle=False)
            preds = predict(head, test_loader, task=args.task, device=device)
            labels = pd.read_csv(args.test_csv)[args.label_col].tolist()
            metrics = compute_metrics(preds, labels, args.task)
            print(f"Held-out test metrics ({args.test_csv}): {metrics}")
            if args.output_csv:
                df = pd.read_csv(args.test_csv)
                df["Predictions"] = pd.Series(preds)
                df.to_csv(args.output_csv, index=False)
                print(f"Test predictions saved to {args.output_csv}")

    # ---------------- evaluate ----------------
    elif args.mode == "evaluate":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --mode evaluate")
        if args.label_col is None:
            raise ValueError("--label_col is required for --mode evaluate")

        loader = _make_loader(args.input_csv, args, args.label_col, shuffle=False)
        preds = predict(head, loader, task=args.task, device=device)
        labels = pd.read_csv(args.input_csv)[args.label_col].tolist()
        metrics = compute_metrics(preds, labels, args.task)
        print(f"Metrics on {args.input_csv}: {metrics}")

        if args.output_csv:
            df = pd.read_csv(args.input_csv)
            df["Predictions"] = pd.Series(preds)
            df.to_csv(args.output_csv, index=False)
            print(f"Predictions saved to {args.output_csv}")

    # ---------------- predict ----------------
    elif args.mode == "predict":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --mode predict")

        loader = _make_loader(args.input_csv, args, None, shuffle=False)
        preds = predict(head, loader, task=args.task, device=device)
        if args.output_csv:
            df = pd.read_csv(args.input_csv)
            df["Predictions"] = pd.Series(preds)
            df.to_csv(args.output_csv, index=False)
            print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()