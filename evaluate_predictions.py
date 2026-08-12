"""
evaluate_predictions.py

Standalone evaluation script for the 3D-MAE-MedImaging repo's prediction output.

main_predict.py writes a CSV with an added 'Predictions' column, where each row
holds the raw probability output of the model (a Python list, because the
original script calls `.tolist()` on a per-sample tensor). This script:

  1. Loads that CSV (optionally joining with a separate CSV containing ground
     truth labels, on an ID column).
  2. Parses the stringified list in the predictions column back into floats.
  3. Computes accuracy / F1 / AUC / PR-AUC / recall (sensitivity) /
     specificity / Brier score (classification) or MSE (regression), using
     the same sklearn functions as `evaluate()` in engine_finetune.py plus a
     few additions.
  4. Computes 95% (configurable) percentile-bootstrap confidence intervals
     for every metric, using 1000 (configurable) resamples by default.
  5. Saves the metrics as both metrics.json and metrics.csv, and
     (optionally) an ROC curve + confusion matrix plot, since the original
     repo has no built-in visualization.

Usage examples
---------------
Binary classification, single CSV with both predictions and true labels:
    python3 evaluate_predictions.py \
        --pred_csv /results/example_data_out.csv \
        --label_col label \
        --task binary

Multiclass, labels in a separate CSV joined on 'patient_id':
    python3 evaluate_predictions.py \
        --pred_csv /results/example_data_out.csv \
        --label_csv /data/test_labels.csv \
        --id_col patient_id \
        --label_col label \
        --task multiclass

Regression:
    python3 evaluate_predictions.py \
        --pred_csv /results/example_data_out.csv \
        --label_col target_value \
        --task regression
"""

import argparse
import ast
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Evaluate predictions produced by main_predict.py', add_help=True)

    parser.add_argument('--pred_csv', required=True, type=str,
                         help='CSV produced by main_predict.py (must contain --pred_col).')
    parser.add_argument('--pred_col', default='Predictions', type=str,
                         help="Name of the column holding predicted probabilities.")
    parser.add_argument('--label_csv', default=None, type=str,
                         help='Optional separate CSV containing ground-truth labels. '
                              'If omitted, labels are expected in --pred_csv itself.')
    parser.add_argument('--id_col', default=None, type=str,
                         help='Column to join --pred_csv and --label_csv on. '
                              'Required if --label_csv is given.')
    parser.add_argument('--label_col', required=True, type=str,
                         help='Name of the ground-truth label column.')
    parser.add_argument('--task', required=True, type=str,
                         choices=['binary', 'multiclass', 'multilabel', 'regression'],
                         help='Type of task, mirrors the criterion branches in evaluate().')
    parser.add_argument('--threshold', default=0.5, type=float,
                         help='Decision threshold for binary/multilabel classification.')
    parser.add_argument('--positive_class_index', default=1, type=int,
                         help='For multiclass softmax output, which column is the '
                              '"positive" class probability used for a binary-style AUC. '
                              'Ignored for true multiclass AUC (uses all columns).')
    parser.add_argument('--output_dir', default='.', type=str,
                         help='Where to save metrics.json and plots.')
    parser.add_argument('--no_plots', action='store_true',
                         help='Skip generating ROC curve / confusion matrix plots.')
    parser.add_argument('--n_bootstrap', default=1000, type=int,
                         help='Number of bootstrap resamples for confidence intervals. '
                              'Set to 0 to skip CI computation.')
    parser.add_argument('--ci', default=0.95, type=float,
                         help='Confidence level for bootstrap intervals (e.g. 0.95 for 95%%).')
    parser.add_argument('--seed', default=0, type=int,
                         help='Random seed for the bootstrap resampling.')
    return parser


def parse_predictions(series):
    """Convert a column of stringified lists (or already-lists/floats) into a 2D numpy array."""
    def _parse_one(val):
        if isinstance(val, str):
            try:
                val = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                # Handles cases like "[np.float64(0.87), np.float64(0.13)]",
                # which can show up if numpy scalars were stored without
                # calling .tolist() first (newer numpy reprs wrap floats in
                # "np.float64(...)"). Strip the wrapper, then parse normally.
                cleaned = re.sub(r'np\.float\d+\(([^)]*)\)', r'\1', val)
                val = ast.literal_eval(cleaned)
        if isinstance(val, (int, float)):
            val = [val]
        return np.asarray(val, dtype=np.float32)

    parsed = series.apply(_parse_one)
    return np.stack(parsed.to_numpy())


def load_data(args):
    pred_df = pd.read_csv(args.pred_csv)
    if args.pred_col not in pred_df.columns:
        raise ValueError(f"'{args.pred_col}' not found in {args.pred_csv}. "
                          f"Available columns: {list(pred_df.columns)}")

    if args.label_csv:
        if not args.id_col:
            raise ValueError('--id_col is required when using --label_csv to join.')
        label_df = pd.read_csv(args.label_csv)
        merged = pred_df.merge(label_df, on=args.id_col, how='inner', suffixes=('', '_label'))
        if len(merged) != len(pred_df):
            print(f"Warning: {len(pred_df) - len(merged)} rows from {args.pred_csv} "
                  f"did not find a matching label in {args.label_csv} and were dropped.")
        df = merged
    else:
        df = pred_df

    if args.label_col not in df.columns:
        raise ValueError(f"'{args.label_col}' not found after loading/merging. "
                          f"Available columns: {list(df.columns)}")

    probs = parse_predictions(df[args.pred_col])
    if args.task == 'multilabel':
        # Multilabel targets are themselves list-valued (e.g. "[0, 1, 0]"),
        # so they round-trip through CSV as strings just like predictions do.
        targets = parse_predictions(df[args.label_col])
    else:
        targets = df[args.label_col].to_numpy()
    return probs, targets, df


def compute_point_metrics(probs, targets, task, threshold=0.5, positive_class_index=1):
    """Computes metric values for one (probs, targets) pair. No printing, no
    randomness — used both for the headline numbers and inside each bootstrap
    resample. Raises on degenerate resamples (e.g. a single class present),
    which the bootstrap loop catches and skips.

    Sensitivity and recall are numerically identical (both = TP / (TP + FN));
    both are reported since clinical/ML audiences use different names for it."""
    metrics = {}

    if task == 'regression':
        preds = probs.squeeze(-1) if probs.shape[-1] == 1 else probs
        metrics['mse'] = float(mean_squared_error(targets, preds))
        return metrics, None, None

    if task == 'binary':
        p = probs[:, 0] if probs.shape[-1] == 1 else probs[:, positive_class_index]
        preds = (p >= threshold).astype(np.float32)
        metrics['accuracy'] = float(accuracy_score(targets, preds))
        metrics['f1'] = float(f1_score(targets, preds, average='binary'))
        metrics['auc'] = float(roc_auc_score(targets, p))
        metrics['pr_auc'] = float(average_precision_score(targets, p))
        recall = float(recall_score(targets, preds, average='binary'))
        metrics['recall'] = recall
        metrics['sensitivity'] = recall
        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else float('nan')
        metrics['brier'] = float(brier_score_loss(targets, p))
        return metrics, preds, p

    if task == 'multiclass':
        preds = np.argmax(probs, axis=1)
        n_classes = probs.shape[1]
        metrics['accuracy'] = float(accuracy_score(targets, preds))
        metrics['f1'] = float(f1_score(targets, preds, average='weighted'))
        metrics['auc'] = float(roc_auc_score(targets, probs, multi_class='ovr'))
        # One-hot targets are reused for both PR-AUC and the Brier score below.
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(targets)), targets.astype(int)] = 1.0
        metrics['pr_auc'] = float(average_precision_score(one_hot, probs, average='macro'))
        recall = float(recall_score(targets, preds, average='macro'))
        metrics['recall'] = recall
        metrics['sensitivity'] = recall
        metrics['specificity'] = _macro_specificity_multiclass(targets, preds, n_classes)
        # Multi-class Brier score: mean over samples of sum over classes of
        # (predicted prob - one-hot indicator)^2.
        metrics['brier'] = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
        return metrics, preds, probs

    if task == 'multilabel':
        preds = (probs >= threshold).astype(np.float32)
        metrics['accuracy'] = float((preds == targets).mean())
        metrics['f1'] = float(f1_score(targets, preds, average='samples'))
        metrics['auc'] = float(roc_auc_score(targets, probs, average='macro'))
        metrics['pr_auc'] = float(average_precision_score(targets, probs, average='macro'))
        recall = float(recall_score(targets, preds, average='macro'))
        metrics['recall'] = recall
        metrics['sensitivity'] = recall
        metrics['specificity'] = _macro_specificity_multilabel(targets, preds)
        # Per-label Brier score, macro-averaged across labels.
        n_labels = targets.shape[1]
        brier_per_label = [
            brier_score_loss(targets[:, j], probs[:, j]) for j in range(n_labels)
        ]
        metrics['brier'] = float(np.mean(brier_per_label))
        return metrics, preds, probs

    raise ValueError(f'Unsupported task: {task}')


def _macro_specificity_multiclass(targets, preds, n_classes):
    """Specificity (TN / (TN + FP)) computed one-vs-rest per class, then
    macro-averaged. Undefined (nan) if a class has no true negatives to speak
    of, which is excluded from the average like sklearn does for recall/F1."""
    cm = confusion_matrix(targets, preds, labels=np.arange(n_classes))
    specificities = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        if (tn + fp) > 0:
            specificities.append(tn / (tn + fp))
    if not specificities:
        return float('nan')
    return float(np.mean(specificities))


def _macro_specificity_multilabel(targets, preds):
    """Specificity per label (TN / (TN + FP)), macro-averaged across labels."""
    n_labels = targets.shape[1]
    specificities = []
    for j in range(n_labels):
        t, p = targets[:, j], preds[:, j]
        tn = np.sum((t == 0) & (p == 0))
        fp = np.sum((t == 0) & (p == 1))
        if (tn + fp) > 0:
            specificities.append(tn / (tn + fp))
    if not specificities:
        return float('nan')
    return float(np.mean(specificities))


def bootstrap_confidence_intervals(probs, targets, task, threshold, positive_class_index,
                                    n_bootstrap, ci, seed):
    """Percentile-bootstrap 95% (or --ci) CIs for every metric in
    compute_point_metrics. Resamples (probs, targets) pairs together, with
    replacement, n_bootstrap times. Resamples that make a metric undefined
    (e.g. only one class present, so AUC/F1 aren't computable) are skipped
    for that metric only, and the number skipped is reported so silent bias
    from excessive skipping is visible."""
    rng = np.random.default_rng(seed)
    n = len(targets)
    per_metric_samples = {}
    n_resamples_errored = 0

    with warnings.catch_warnings():
        # sklearn emits UndefinedMetricWarning (and returns nan) rather than
        # raising when a resample happens to contain only one class. We
        # silence the spam here and instead drop nan/inf values per-metric
        # below, reporting how many were dropped.
        warnings.simplefilter('ignore')
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            try:
                sample_metrics, _, _ = compute_point_metrics(
                    probs[idx], targets[idx], task, threshold, positive_class_index)
            except Exception:
                n_resamples_errored += 1
                continue
            for k, v in sample_metrics.items():
                per_metric_samples.setdefault(k, []).append(v)

    alpha = (1 - ci) / 2
    ci_results = {}
    for k, samples in per_metric_samples.items():
        samples = np.asarray(samples, dtype=np.float64)
        n_total = len(samples)
        finite = samples[np.isfinite(samples)]
        n_undefined = n_total - len(finite)
        if len(finite) == 0:
            ci_results[k] = {'lower': None, 'upper': None,
                              'n_valid_resamples': 0, 'n_undefined_resamples': n_undefined}
            continue
        lower = float(np.percentile(finite, 100 * alpha))
        upper = float(np.percentile(finite, 100 * (1 - alpha)))
        ci_results[k] = {
            'lower': lower,
            'upper': upper,
            'n_valid_resamples': int(len(finite)),
            'n_undefined_resamples': int(n_undefined),
        }
        if n_undefined > 0:
            print(f"Note: metric '{k}' was undefined (e.g. only one class present) "
                  f"in {n_undefined}/{n_total} bootstrap resamples; those were "
                  f"excluded from its CI.")

    if n_resamples_errored > 0:
        print(f"Note: {n_resamples_errored}/{n_bootstrap} bootstrap resamples "
              f"raised an error and were skipped entirely.")

    return ci_results


def evaluate_predictions(probs, targets, task, threshold=0.5, positive_class_index=1,
                          n_bootstrap=1000, ci=0.95, seed=0):
    """Computes point-estimate metrics (mirroring engine_finetune.py's evaluate())
    plus, if n_bootstrap > 0, percentile-bootstrap confidence intervals for each."""
    metrics, preds, probs_for_plot = compute_point_metrics(
        probs, targets, task, threshold, positive_class_index)

    ci_results = {}
    if n_bootstrap > 0:
        ci_results = bootstrap_confidence_intervals(
            probs, targets, task, threshold, positive_class_index, n_bootstrap, ci, seed)

    ci_pct = int(round(ci * 100))
    for name, value in metrics.items():
        entry = ci_results.get(name)
        if entry and entry['lower'] is not None:
            lo, hi = entry['lower'], entry['upper']
            print(f'* {name}: {value:.4f}  [{ci_pct}% CI: {lo:.4f} - {hi:.4f}]')
        elif entry:
            print(f'* {name}: {value:.4f}  [{ci_pct}% CI: undefined for all resamples]')
        else:
            print(f'* {name}: {value:.4f}')

    metrics_with_ci = {
        name: {'value': value, **ci_results.get(name, {})}
        for name, value in metrics.items()
    }
    return metrics_with_ci, preds, probs_for_plot


def make_plots(task, targets, preds, probs, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Confusion matrix (binary / multiclass only)
    if task in ('binary', 'multiclass') and preds is not None:
        cm = confusion_matrix(targets, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close(fig)

    # ROC curve (binary only)
    if task == 'binary' and probs is not None:
        fpr, tpr, _ = roc_curve(targets, probs)
        auc_val = roc_auc_score(targets, probs)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f'ROC (AUC = {auc_val:.3f})')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
        plt.close(fig)


def main(args):
    probs, targets, _ = load_data(args)
    metrics, preds, probs_for_plot = evaluate_predictions(
        probs, targets, args.task, args.threshold, args.positive_class_index,
        n_bootstrap=args.n_bootstrap, ci=args.ci, seed=args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    metrics_csv_path = os.path.join(args.output_dir, 'metrics.csv')
    metrics_rows = []
    for name, entry in metrics.items():
        metrics_rows.append({
            'metric': name,
            'value': entry.get('value'),
            'ci_lower': entry.get('lower'),
            'ci_upper': entry.get('upper'),
            'n_valid_resamples': entry.get('n_valid_resamples'),
            'n_undefined_resamples': entry.get('n_undefined_resamples'),
        })
    pd.DataFrame(metrics_rows).to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics to {metrics_csv_path}")

    if not args.no_plots and args.task != 'regression':
        make_plots(args.task, targets, preds, probs_for_plot, args.output_dir)
        print(f"Saved plots to {args.output_dir}")


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)