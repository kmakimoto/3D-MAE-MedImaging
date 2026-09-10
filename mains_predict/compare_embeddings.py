#!/usr/bin/env python3
"""Compare your TANGERINE embeddings CSV against the TANGERINE .npz embeddings.

Hardcoded for this one comparison -- edit the CONFIG block below if any path
or column name changes. No command-line arguments.

Steps:
  1. Convert the relevant .npz embeddings (INSP scans, matching SIDs in CSV A)
     into a CSV.
  2. Merge in COPD_P1 / pctEmph_Thirona_P1 labels from the labels file.
  3. Run linear-probe downstream evaluation (classification + regression)
     with bootstrap confidence intervals, comparing the two embedding sets.
  4. Per-subject cosine similarity.
  5. Linear CKA + orthogonal Procrustes (global geometric equivalence).
  6. Linear reconstruction Y ~ XW in both directions, cross-validated R^2
     (functional/informational equivalence: is B linearly recoverable from
     A, and vice versa).
  7. Pairwise similarity correlation: within-A subject-subject cosine
     similarities vs. within-B subject-subject cosine similarities, plotted
     against each other (representational similarity analysis).
  8. Top-10 Jaccard neighbor overlap (retrieval equivalence).
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# CONFIG -- edit these if a path or column name changes
# --------------------------------------------------------------------------

CSV_A = "example_inputs/test_embeddings_20260909.csv"
EMBEDDING_COL_A = "Embedding"
SID_COL_A = "sid"                 # column in CSV_A holding the subject ID
PATH_COL_A = "Path"               # fallback if SID_COL_A isn't present
SID_REGEX_A = r"([A-Za-z0-9]+)_(?:INSP|EXP)"  # used with PATH_COL_A as fallback

NPZ_DIR = "/home/ahc44/Datos/COPDGene/COPDGeneEmbeddingFM/"
NPZ_GLOB = "*tangerine*embedding*.npz"
NPZ_KEY = "image"

SCAN_TYPE = "INSP"                # "INSP", "EXP", or None to disable filtering

LABELS_CSV = "example_inputs/COPDGene_P1P2P3.csv"
LABELS_SID_COL = "sid"
CLASSIFICATION_LABEL_COL = "COPD_P1"
CLASSIFICATION_RECODE = {3.0: 0.0}   # value 3 -> 0; set to {} to disable
REGRESSION_LABEL_COL = "pctEmph_Thirona_P1"

N_SPLITS = 5
N_BOOTSTRAP = 1000
SEED = 0
TOP_K = 10

OUTPUT_DIR = "embedding_comparison_results"

# --------------------------------------------------------------------------
# Step 1: load CSV A, determine SIDs, optionally filter to one scan type
# --------------------------------------------------------------------------

print(f"Loading CSV A: {CSV_A}")
df_a = pd.read_csv(CSV_A)

if SID_COL_A in df_a.columns:
    df_a["SID"] = df_a[SID_COL_A].astype(str)
else:
    df_a["SID"] = df_a[PATH_COL_A].apply(lambda p: re.search(SID_REGEX_A, str(p)).group(1))

if SCAN_TYPE and PATH_COL_A in df_a.columns:
    other = "EXP" if SCAN_TYPE == "INSP" else "INSP"
    n_before = len(df_a)
    df_a = df_a[df_a[PATH_COL_A].apply(lambda p: SCAN_TYPE in str(p) and other not in str(p))]
    print(f"scan_type={SCAN_TYPE}: kept {len(df_a)}/{n_before} rows in CSV A")

target_sids = set(df_a["SID"])
print(f"CSV A: {len(df_a)} rows, {len(target_sids)} unique SIDs")

# --------------------------------------------------------------------------
# Step 2: convert the relevant .npz embeddings into CSV B
# --------------------------------------------------------------------------

print(f"\nScanning {NPZ_DIR} for '{NPZ_GLOB}'...")
npz_paths = sorted(Path(NPZ_DIR).rglob(NPZ_GLOB))
if SCAN_TYPE:
    other = "EXP" if SCAN_TYPE == "INSP" else "INSP"
    npz_paths = [p for p in npz_paths if SCAN_TYPE in p.name and other not in p.name]
print(f"{len(npz_paths)} .npz files match glob + scan_type")

rows_b = []
n_skipped_not_target = n_skipped_wrong_key = n_skipped_error = 0
for p in npz_paths:
    sid = p.stem.split("_")[0]
    if sid not in target_sids:
        n_skipped_not_target += 1
        continue
    try:
        data = np.load(p, allow_pickle=True)
        if NPZ_KEY not in data.files:
            n_skipped_wrong_key += 1
            continue
        embedding = np.asarray(data[NPZ_KEY]).squeeze()
        if embedding.ndim != 1:
            n_skipped_wrong_key += 1
            continue
    except Exception as exc:
        n_skipped_error += 1
        print(f"  Failed to load {p.name}: {exc}")
        continue
    rows_b.append({"SID": sid, EMBEDDING_COL_A: json.dumps(embedding.tolist())})

df_b = pd.DataFrame(rows_b)
print(
    f"Converted {len(df_b)} .npz embeddings "
    f"({n_skipped_not_target} skipped: not in CSV A's SIDs | "
    f"{n_skipped_wrong_key} skipped: wrong key/shape | "
    f"{n_skipped_error} skipped: load error)"
)

missing_sids = target_sids - set(df_b["SID"])
if missing_sids:
    print(f"  WARNING: {len(missing_sids)} SIDs from CSV A had no matching .npz, e.g.: {list(missing_sids)[:5]}")

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)
df_b.to_csv(output_dir / "csv_b_from_npz.csv", index=False)

# --------------------------------------------------------------------------
# Step 3: merge the two embedding sets by SID
# --------------------------------------------------------------------------

merged = df_a.merge(df_b, on="SID", how="inner", suffixes=("_a", "_b"))
print(f"\nMatched on SID: {len(merged)} rows (CSV A: {len(df_a)}, CSV B: {len(df_b)})")

emb_col_a = EMBEDDING_COL_A if EMBEDDING_COL_A in merged.columns else f"{EMBEDDING_COL_A}_a"
emb_col_b = EMBEDDING_COL_A if EMBEDDING_COL_A in merged.columns else f"{EMBEDDING_COL_A}_b"
emb_a = np.stack([np.array(json.loads(v)) for v in merged[emb_col_a]])
emb_b = np.stack([np.array(json.loads(v)) for v in merged[emb_col_b]])
assert emb_a.shape == emb_b.shape, f"Shape mismatch: {emb_a.shape} vs {emb_b.shape}"

# --------------------------------------------------------------------------
# Step 4: merge in labels
# --------------------------------------------------------------------------

print(f"\nLoading labels: {LABELS_CSV}")
labels_read_fn = pd.read_excel if Path(LABELS_CSV).suffix.lower() in (".xlsx", ".xls") else pd.read_csv
df_labels = labels_read_fn(LABELS_CSV)
df_labels["SID"] = df_labels[LABELS_SID_COL].astype(str)
merged = merged.merge(df_labels, on="SID", how="left", suffixes=("", "_label"))
print(f"Labels matched: {merged['SID'].isin(df_labels['SID']).sum()}/{len(merged)}")

results = {"n_subjects_matched": len(merged), "embedding_dim": emb_a.shape[1]}

# --------------------------------------------------------------------------
# Step 5: downstream linear probes, with bootstrap CIs
# --------------------------------------------------------------------------


def bootstrap_ci(metric_fn, y_true, y_score, n_boot=N_BOOTSTRAP, seed=SEED, alpha=0.05):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    stats, n_skipped = [], 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            stats.append(metric_fn(y_true[idx], y_score[idx]))
        except ValueError:
            n_skipped += 1
    stats = np.array(stats)
    return {
        "point_estimate": float(metric_fn(y_true, y_score)),
        "ci_lower": float(np.percentile(stats, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(stats, 100 * (1 - alpha / 2))),
        "n_boot": len(stats),
        "n_boot_skipped": n_skipped,
    }


def classification_probe(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_proba = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X[train_idx]), y[train_idx])
        oof_proba[test_idx] = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
    oof_pred = (oof_proba >= 0.5).astype(int)
    return {
        "auc": bootstrap_ci(roc_auc_score, y, oof_proba),
        "accuracy": bootstrap_ci(accuracy_score, y, oof_pred),
    }


def regression_probe(X, y, alpha=10.0):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(y))
    for train_idx, test_idx in kf.split(X):
        scaler = StandardScaler().fit(X[train_idx])
        reg = Ridge(alpha=alpha).fit(scaler.transform(X[train_idx]), y[train_idx])
        oof_pred[test_idx] = reg.predict(scaler.transform(X[test_idx]))
    return {
        "r2": bootstrap_ci(r2_score, y, oof_pred),
        "mae": bootstrap_ci(mean_absolute_error, y, oof_pred),
        "rmse": bootstrap_ci(lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))), y, oof_pred),
    }


print("\n=== Downstream linear-probe evaluation ===")

# --- classification ---
y_raw = pd.to_numeric(merged[CLASSIFICATION_LABEL_COL], errors="coerce")
if CLASSIFICATION_RECODE:
    print(f"  Recoding {int(y_raw.isin(list(CLASSIFICATION_RECODE.keys())).sum())} rows: {CLASSIFICATION_RECODE}")
    y_raw = y_raw.replace(CLASSIFICATION_RECODE)
notna_mask = y_raw.notna()
binary_mask = y_raw.isin([0, 1, 0.0, 1.0])
non_binary = notna_mask & ~binary_mask
if non_binary.any():
    print(f"  WARNING: dropping {int(non_binary.sum())} rows with non-binary values: {sorted(y_raw[non_binary].unique())}")
valid_cls = (notna_mask & binary_mask).values
y_cls = y_raw[valid_cls].astype(int).values
print(f"  '{CLASSIFICATION_LABEL_COL}': {valid_cls.sum()} subjects, class balance {dict(zip(*np.unique(y_cls, return_counts=True)))}")

results["classification"] = {
    "label_col": CLASSIFICATION_LABEL_COL,
    "n_subjects": int(valid_cls.sum()),
    "embedding_a": classification_probe(emb_a[valid_cls], y_cls),
    "embedding_b": classification_probe(emb_b[valid_cls], y_cls),
}

# --- regression ---
y_reg_raw = pd.to_numeric(merged[REGRESSION_LABEL_COL], errors="coerce")
valid_reg = y_reg_raw.notna().values
y_reg = y_reg_raw[valid_reg].values
print(f"  '{REGRESSION_LABEL_COL}': {valid_reg.sum()} subjects")

results["regression"] = {
    "label_col": REGRESSION_LABEL_COL,
    "n_subjects": int(valid_reg.sum()),
    "embedding_a": regression_probe(emb_a[valid_reg], y_reg),
    "embedding_b": regression_probe(emb_b[valid_reg], y_reg),
}

print(json.dumps(results["classification"], indent=2))
print(json.dumps(results["regression"], indent=2))

# --------------------------------------------------------------------------
# Step 6: cosine similarity
# --------------------------------------------------------------------------

a_norm = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
b_norm = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)
cos_sims = np.sum(a_norm * b_norm, axis=1)
results["cosine_similarity"] = {
    "mean": float(cos_sims.mean()),
    "std": float(cos_sims.std()),
    "min": float(cos_sims.min()),
    "max": float(cos_sims.max()),
    "median": float(np.median(cos_sims)),
}
print("\n=== Cosine similarity ===")
print(json.dumps(results["cosine_similarity"], indent=2))
merged["cosine_similarity"] = cos_sims

# --------------------------------------------------------------------------
# Step 7: linear CKA + orthogonal Procrustes
# --------------------------------------------------------------------------

Xc = emb_a - emb_a.mean(axis=0, keepdims=True)
Yc = emb_b - emb_b.mean(axis=0, keepdims=True)
hsic = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
cka = float(hsic / (np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")))

Xn = Xc / np.linalg.norm(Xc, ord="fro")
Yn = Yc / np.linalg.norm(Yc, ord="fro")
R, scale = orthogonal_procrustes(Xn, Yn)
disparity = float(np.linalg.norm(Xn @ R - Yn, ord="fro"))

results["linear_cka"] = cka
results["orthogonal_procrustes"] = {"disparity": disparity, "procrustes_scale": float(scale)}
print("\n=== Linear CKA + orthogonal Procrustes ===")
print(f"Linear CKA: {cka:.4f}  (1.0 = identical representational geometry)")
print(json.dumps(results["orthogonal_procrustes"], indent=2))

# --------------------------------------------------------------------------
# Step 8: linear reconstruction Y ~ XW (functional / informational equivalence)
#
# Fits a multi-output ridge regression predicting one full embedding matrix
# from the other, cross-validated, and reports R^2 across all 2048 output
# dimensions. R^2 ~= 1.0 means every linear feature of Y is recoverable from
# X -- i.e. the two embedding spaces carry the same information, not just
# similar overall geometry. Tested in both directions since reconstructing
# B from A is not necessarily symmetric with reconstructing A from B.
#
# RidgeCV auto-selects the regularization strength per fold via internal
# leave-one-out CV, since a fixed alpha tuned for a single-target regression
# (as used for the clinical labels above) is not a safe default when
# predicting 2048 correlated output dimensions from n=168 samples -- this
# task is deep in small-n/high-dim territory, so proper regularization
# matters more than usual for a trustworthy R^2 estimate.
# --------------------------------------------------------------------------

RECONSTRUCTION_ALPHAS = np.logspace(-2, 4, 13)


def linear_reconstruction_probe(X, Y, n_splits=N_SPLITS, seed=SEED):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros_like(Y, dtype=float)
    for train_idx, test_idx in kf.split(X):
        scaler_x = StandardScaler().fit(X[train_idx])
        reg = RidgeCV(alphas=RECONSTRUCTION_ALPHAS).fit(scaler_x.transform(X[train_idx]), Y[train_idx])
        oof_pred[test_idx] = reg.predict(scaler_x.transform(X[test_idx]))

    r2_per_dim = r2_score(Y, oof_pred, multioutput="raw_values")
    r2_uniform_avg = float(r2_score(Y, oof_pred, multioutput="uniform_average"))

    rng = np.random.RandomState(seed)
    n = Y.shape[0]
    boot_stats = [
        r2_score(Y[idx], oof_pred[idx], multioutput="uniform_average")
        for idx in (rng.randint(0, n, n) for _ in range(N_BOOTSTRAP))
    ]
    boot_stats = np.array(boot_stats)

    return {
        "r2_mean_across_dims": r2_uniform_avg,
        "r2_median_across_dims": float(np.median(r2_per_dim)),
        "r2_min_across_dims": float(np.min(r2_per_dim)),
        "r2_max_across_dims": float(np.max(r2_per_dim)),
        "ci_lower": float(np.percentile(boot_stats, 2.5)),
        "ci_upper": float(np.percentile(boot_stats, 97.5)),
        "n_boot": len(boot_stats),
    }


print("\n=== Linear reconstruction Y ~ XW (functional equivalence) ===")
results["linear_reconstruction"] = {
    "b_from_a": linear_reconstruction_probe(emb_a, emb_b),  # predict B given A
    "a_from_b": linear_reconstruction_probe(emb_b, emb_a),  # predict A given B
}
print(json.dumps(results["linear_reconstruction"], indent=2))

# --------------------------------------------------------------------------
# Step 8: pairwise similarity correlation (representational similarity
# analysis)
#
# Different question from Step 6's per-subject cosine similarity above.
# There, we asked: "for subject i, how similar is embedding_a[i] to
# embedding_b[i]?" Here we ask a structural question instead: "if subject i
# and subject j are similar to each other under embedding A, are they also
# similar to each other under embedding B?"
#
# For every pair of subjects (i, j), compute their cosine similarity within
# A and within B separately, then correlate the two resulting lists of
# pairwise similarities across all C(n,2) pairs. A strong correlation means
# the *relative structure* between subjects (who is similar to whom) is
# preserved across the two embedding spaces, independent of any overall
# scale/offset difference between them.
# --------------------------------------------------------------------------

sim_matrix_a = a_norm @ a_norm.T
sim_matrix_b = b_norm @ b_norm.T
iu = np.triu_indices(emb_a.shape[0], k=1)  # upper triangle, excluding diagonal (self-similarity)
pairwise_a = sim_matrix_a[iu]
pairwise_b = sim_matrix_b[iu]

pearson_r, pearson_p = pearsonr(pairwise_a, pairwise_b)
spearman_r, spearman_p = spearmanr(pairwise_a, pairwise_b)

results["pairwise_similarity_correlation"] = {
    "n_pairs": int(len(pairwise_a)),
    "pearson_r": float(pearson_r),
    "pearson_p": float(pearson_p),
    "spearman_r": float(spearman_r),
    "spearman_p": float(spearman_p),
}
print("\n=== Pairwise similarity correlation (subject-subject structure) ===")
print(json.dumps(results["pairwise_similarity_correlation"], indent=2))

plt.figure(figsize=(6, 6))
plt.scatter(pairwise_a, pairwise_b, s=4, alpha=0.25, edgecolors="none")
lims = [min(pairwise_a.min(), pairwise_b.min()), max(pairwise_a.max(), pairwise_b.max())]
plt.plot(lims, lims, "r--", alpha=0.6, linewidth=1, label="y = x")
plt.xlabel("Pairwise cosine similarity, Embedding A")
plt.ylabel("Pairwise cosine similarity, Embedding B")
plt.title(f"Pairwise similarity correlation\nPearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f} (n={len(pairwise_a)} pairs)")
plt.legend()
plt.tight_layout()
plot_path = output_dir / "pairwise_similarity_correlation.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Scatter plot saved to {plot_path}")

# --------------------------------------------------------------------------
# Step 9: top-k Jaccard neighbor overlap
# --------------------------------------------------------------------------


def topk_neighbors(emb, k):
    norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sims = norm @ norm.T
    np.fill_diagonal(sims, -np.inf)
    return [set(row) for row in np.argsort(-sims, axis=1)[:, :k]]


neighbors_a = topk_neighbors(emb_a, TOP_K)
neighbors_b = topk_neighbors(emb_b, TOP_K)
jaccards = np.array([
    len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
    for sa, sb in zip(neighbors_a, neighbors_b)
])
results["top_k_jaccard"] = {
    "k": TOP_K,
    "jaccard_mean": float(jaccards.mean()),
    "jaccard_std": float(jaccards.std()),
    "jaccard_min": float(jaccards.min()),
    "jaccard_max": float(jaccards.max()),
}
print(f"\n=== Top-{TOP_K} Jaccard neighbor overlap ===")
print(json.dumps(results["top_k_jaccard"], indent=2))

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------

with open(output_dir / "comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)
merged[["SID", "cosine_similarity"]].to_csv(output_dir / "per_subject_cosine_similarity.csv", index=False)

print(f"\nFull results saved to {output_dir / 'comparison_results.json'}")
print(f"Per-subject cosine similarity saved to {output_dir / 'per_subject_cosine_similarity.csv'}")