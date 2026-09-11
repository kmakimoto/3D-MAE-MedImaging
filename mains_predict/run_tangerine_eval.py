"""
run_tangerine_eval.py

Trains and evaluates the TANGERINE prediction head (LinearHead / MLPHead, from
tangerine_head.py -- keep it in the same directory) on pre-extracted, frozen
embeddings, using TANGERINE's own statistical protocol from the paper's Methods:

    "Each model is trained five times with different random seeds ... The mean
    and standard deviation of performance metrics across the five runs are
    calculated, and the standard error is derived with confidence intervals
    (CIs) at 95% are computed using 1.96x standard error."

So for --n_seeds repeats (default 5):
    - a fresh head is initialized and trained on --train_csv
    - the checkpoint with the lowest --val_csv loss is kept (early stopping,
      matching main_finetune_class_epoch.py's best_model_loss.pth behaviour)
    - it is evaluated once on the held-out --test_csv

Across the n_seeds repeats, mean, SD, SE (=SD/sqrt(n)), and 95% CI (=mean +/- 1.96*SE)
are reported per metric. This is the "across-seed" layer of uncertainty: how much
performance varies due to training randomness.

Additionally (unless --n_bootstrap 0), each seed's own test-set predictions are run
through evaluate_predictions.py's percentile-bootstrap evaluation (imported, not
subprocessed -- evaluate_predictions.py itself is unmodified and still runs standalone
via its own CLI). This is a second, complementary "within-seed" layer of uncertainty:
how much a single trained head's test-set estimate would wobble given the finite size
of the test set, plus a richer metric set (accuracy, F1, sensitivity, specificity,
Brier score) that the paper doesn't report but is useful diagnostic detail. Results
are saved per seed under <output_dir>/bootstrap/seed_{seed}_bootstrap_metrics.json.

These two CIs answer different questions and are not substitutes for each other --
see the two docstring layers above.

--------------------------------------------------------------------------
Metrics
--------------------------------------------------------------------------
Classification (binary/multiclass): AUROC and AUPRC -- this is exactly what the
TANGERINE paper reports. Multiclass AUROC uses macro one-vs-rest averaging.

Regression: the TANGERINE paper does not include any regression tasks or metrics
-- there is no paper-specific protocol to copy here. This script reports standard
regression metrics (MAE, RMSE, R2) instead, run through the same 5-seed / mean /
SD / 95% CI protocol described above, for consistency with the classification side.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
python mains_predict/run_tangerine_eval.py \
    --train_csv example_inputs/splits/longitudinal_train.csv \
    --val_csv example_inputs/splits/longitudinal_val.csv \
    --test_csv example_inputs/splits/longitudinal_test.csv \
    --label_col Pi10_Thirona_P2 --task regression \
    --head_type frozen_ann --nb_classes 1 \
    --output_dir results/Pi10_Thirona_P2
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prediction_head import (  # noqa: E402
    build_head, EmbeddingDataset, DataLoader, get_criterion, _compute_loss,
    evaluate_loss, predict, compute_metrics, make_embedding_dataset,
)
from evaluate_predictions import (  # noqa: E402
    evaluate_predictions as bootstrap_evaluate, make_plots, compute_point_metrics,
)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_seed(args, seed, device):
    set_seed(seed)

    head = build_head(
        head_type=args.head_type, embed_dim=args.embed_dim, num_classes=args.nb_classes,
        apply_fc_norm=args.apply_fc_norm, hidden_layers=args.hidden_layers,
    ).to(device)

    train_ds = make_embedding_dataset(args.train_csv, args.label_col, args.embeddings_store,
                                       args.id_col, args.embedding_col, args.flattened,
                                       args.embedding_path_col, args.npz_key)
    val_ds = make_embedding_dataset(args.val_csv, args.label_col, args.embeddings_store,
                                     args.id_col, args.embedding_col, args.flattened,
                                     args.embedding_path_col, args.npz_key)
    test_ds = make_embedding_dataset(args.test_csv, args.label_col, args.embeddings_store,
                                      args.id_col, args.embedding_col, args.flattened,
                                      args.embedding_path_col, args.npz_key)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    criterion = get_criterion(args.task, args.binary_class_weights)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0

    for epoch in range(args.epochs):
        head.train()
        for emb, label in train_loader:
            emb, label = emb.to(device), label.to(device)
            optimizer.zero_grad()
            output = head(emb)
            loss = _compute_loss(criterion, output, label, args.task)
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(head, val_loader, criterion, args.task, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(head.state_dict())
            best_epoch = epoch + 1

    head.load_state_dict(best_state)
    print(f"  seed {seed}: best val_loss={best_val_loss:.4f} at epoch {best_epoch}")

    if args.save_checkpoints:
        ckpt_dir = Path(args.output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), ckpt_dir / f"seed_{seed}.pt")

    preds = predict(head, test_loader, task=args.task, device=device)
    labels = test_ds.df[args.label_col].tolist()  # test_ds already dropped rows missing this label_col
    metrics = compute_metrics(preds, labels, args.task)
    metrics["seed"] = seed
    metrics["best_val_loss"] = best_val_loss
    metrics["best_epoch"] = best_epoch

    probs_arr = np.array(preds, dtype=np.float32)
    labels_arr = np.array(labels)

    # Richer metric set (accuracy, F1, recall, sensitivity, specificity, Brier) from
    # evaluate_predictions.py's compute_point_metrics, merged into the SAME per-seed dict
    # that feeds the 5-seed mean/SD/95% CI aggregation below -- not just the within-seed
    # bootstrap layer. AUROC/AUPRC/MSE already come from compute_metrics above, so those
    # keys are skipped here to avoid confusing near-duplicate columns.
    if args.task in ("binary", "multiclass"):
        point_metrics, _, _ = compute_point_metrics(probs_arr, labels_arr, args.task, args.threshold)
        for k in ("accuracy", "f1", "recall", "sensitivity", "specificity", "brier"):
            if k in point_metrics:
                metrics[k] = point_metrics[k]

    if args.n_bootstrap > 0:
        bootstrap_metrics, b_preds, b_probs = bootstrap_evaluate(
            probs_arr, labels_arr, task=args.task, threshold=args.threshold,
            n_bootstrap=args.n_bootstrap, ci=args.ci, seed=seed)

        bootstrap_dir = Path(args.output_dir) / "bootstrap"
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        with open(bootstrap_dir / f"seed_{seed}_bootstrap_metrics.json", "w") as f:
            json.dump(bootstrap_metrics, f, indent=2)

        if args.bootstrap_plots and args.task != "regression":
            make_plots(args.task, labels_arr, b_preds, b_probs,
                       str(bootstrap_dir / f"seed_{seed}_plots"))

    return metrics


def aggregate(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in per_seed_df.columns if c not in ("seed", "best_epoch")]
    rows = []
    n = len(per_seed_df)
    for col in metric_cols:
        vals = per_seed_df[col].astype(float)
        mean = vals.mean()
        sd = vals.std(ddof=1) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n > 1 else 0.0
        ci95 = 1.96 * se
        rows.append({"metric": col, "mean": mean, "sd": sd, "se": se,
                     "ci95_lower": mean - ci95, "ci95_upper": mean + ci95, "n_seeds": n})
    return pd.DataFrame(rows)


def get_args_parser():
    p = argparse.ArgumentParser("Train/evaluate TANGERINE head with paper's 5-seed protocol")
    p.add_argument("--train_csv", required=True, type=str)
    p.add_argument("--val_csv", required=True, type=str)
    p.add_argument("--test_csv", required=True, type=str)
    p.add_argument("--embedding_col", default="Embedding", type=str)
    p.add_argument("--flattened", action="store_true")
    p.add_argument("--label_col", required=True, type=str)

    p.add_argument("--embeddings_store", default=None, type=str,
                    help="Legacy: path to a consolidated .npz embeddings store (sids + embeddings arrays). "
                         "If given, takes priority over --embedding_path_col.")
    p.add_argument("--id_col", default="SID", type=str)
    p.add_argument("--embedding_path_col", default="EmbeddingPath", type=str,
                    help="Recommended: column in the split CSVs holding each subject's own .npz path "
                         "(as produced by make_splits.py). Embeddings are loaded directly from these "
                         "paths, cached in memory per run -- no consolidated store needed.")
    p.add_argument("--npz_key", default="image", type=str,
                    help="Key inside each .npz file holding the embedding array, only used with --embedding_path_col.")

    p.add_argument("--head_type", choices=["linear", "mlp", "frozen_ann"], default="linear")
    p.add_argument("--hidden_layers", type=int, nargs="+", default=None,
                    help="Only used for --head_type mlp/frozen_ann. Leave unset to use each head's own "
                         "paper-matched default: [64] for mlp, [128, 32] for frozen_ann.")
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--nb_classes", type=int, default=1)
    p.add_argument("--task", choices=["binary", "multiclass", "regression"], default="binary")
    p.add_argument("--apply_fc_norm", action="store_true",
                    help="Set if embeddings are raw 'concat' (not yet fc_norm'd).")

    p.add_argument("--n_seeds", default=5, type=int,
                    help="Number of independently trained+evaluated repeats (TANGERINE paper uses 5).")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Explicit list of seeds; overrides --n_seeds if given.")

    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--epochs", default=50, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--binary_class_weights", type=float, nargs="+", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--save_checkpoints", action="store_true")
    p.add_argument("--output_dir", required=True, type=str)

    p.add_argument("--n_bootstrap", default=1000, type=int,
                    help="Bootstrap resamples for the within-seed CI layer (evaluate_predictions.py). "
                         "Set to 0 to skip this layer entirely (only the across-seed CI is computed).")
    p.add_argument("--ci", default=0.95, type=float, help="Confidence level for the bootstrap CIs.")
    p.add_argument("--threshold", default=0.5, type=float,
                    help="Decision threshold used for accuracy/F1/sensitivity/specificity in the bootstrap layer.")
    p.add_argument("--bootstrap_plots", action="store_true",
                    help="Also save an ROC curve + confusion matrix per seed (evaluate_predictions.py's make_plots).")
    return p


def validate_labels(args):
    """Catches label/--nb_classes mismatches with a clear Python error before any
    training starts, instead of letting them surface as an opaque CUDA device-side
    assert deep inside a backward() call (which also poisons the CUDA context for
    the rest of the process)."""
    if args.task != "multiclass":
        return
    for name, path in (("train_csv", args.train_csv), ("val_csv", args.val_csv), ("test_csv", args.test_csv)):
        df = pd.read_csv(path)
        if args.label_col not in df.columns:
            continue
        vals = df[args.label_col].dropna()
        if len(vals) == 0:
            continue
        non_integer = vals[vals != vals.astype(int)]
        out_of_range = vals[(vals < 0) | (vals >= args.nb_classes)]
        if len(non_integer) > 0 or len(out_of_range) > 0:
            uniques = sorted(vals.unique().tolist())
            raise ValueError(
                f"--task multiclass with --nb_classes {args.nb_classes} requires '{args.label_col}' to contain "
                f"only integers in [0, {args.nb_classes}) -- but {name} ({path}) has values outside that range "
                f"and/or non-integer values. All distinct non-missing values found there: {uniques}. Set "
                f"--nb_classes to the actual number of distinct classes, and make sure they are contiguous "
                f"integers starting at 0 (recode any sentinel/out-of-sequence codes in make_splits.py first).")


def main():
    args = get_args_parser().parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_labels(args)

    seeds = args.seeds if args.seeds else list(range(args.n_seeds))

    all_metrics = []
    for seed in seeds:
        print(f"=== seed {seed} ===")
        metrics = train_one_seed(args, seed, device)
        all_metrics.append(metrics)

    per_seed_df = pd.DataFrame(all_metrics)
    per_seed_df.to_csv(out_dir / "metrics_per_seed.csv", index=False)

    summary_df = aggregate(per_seed_df)
    summary_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    print(f"\n=== {args.label_col} ({args.task}) -- {len(seeds)}-seed summary ===")
    for _, row in summary_df.iterrows():
        print(f"  {row['metric']}: {row['mean']:.4f}  (95% CI: {row['ci95_lower']:.4f}-{row['ci95_upper']:.4f}, "
              f"SD={row['sd']:.4f}, n={int(row['n_seeds'])})")

    print(f"\nSaved metrics_per_seed.csv and metrics_summary.csv to {out_dir}")


if __name__ == "__main__":
    main()