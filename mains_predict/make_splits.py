#!/usr/bin/env python3
"""make_splits.py

Hardcoded for this one cohort -- edit the CONFIG block below if any path,
column name, or outcome changes. No command-line arguments.

Scans a folder of TANGERINE .npz embeddings, merges in outcomes from a labels
CSV, and writes ONE train/val/test split PER CATEGORY (e.g. "cross_sectional"
and "longitudinal") -- every outcome within a category shares the same
subject-level train/val/test membership (a subject who lands in train for one
cross-sectional outcome is in train for every cross-sectional outcome).

Each output CSV has columns: SID, EmbeddingPath, <outcome columns for that
category>. run_tangerine_eval.py loads each subject's embedding directly from
EmbeddingPath -- no intermediate CSV or consolidated embeddings store needed.

Mark at most one outcome per category with "stratify": True in OUTCOMES below
to nominate it as that category's stratification anchor (falls back to a
plain random split if no outcome in a category is marked). Low-prevalence
binary anchors (minority class fraction < LOW_PREVALENCE_THRESHOLD)
automatically switch to the 60:10:30 ratios instead of 50:10:40, matching
TANGERINE's own treatment of its low-prevalence cancer tasks.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# CONFIG -- edit these if a path, column name, or outcome changes
# --------------------------------------------------------------------------

NPZ_DIR = "/home/ahc44/Datos/COPDGene/COPDGeneEmbeddingFM/"
NPZ_GLOB = "*tangerine*embedding*.npz"
NPZ_KEY = "image"
SCAN_TYPE = "INSP"             

LABELS_CSV = "example_inputs/COPDGene_P1P2P3.csv"
LABELS_SID_COL = "sid"

ID_COL = "SID"
EMBEDDING_PATH_COL = "EmbeddingPath"

# List every outcome here. "category" groups outcomes into one shared split
# (e.g. all cross-sectional outcomes get one train/val/test partition, all
# longitudinal outcomes get another). "stratify": True marks AT MOST ONE
# outcome per category as that category's stratification anchor -- every
# other outcome in the category just rides along with that split.
OUTCOMES = [
    {"name": "COPD_P1",             "task": "binary",     "category": "cross_sectional", "stratify": True, "recode": {3.0: 0.0}},   
    {"name": "finalGold_P1",             "task": "multiclass",     "category": "cross_sectional", "stratify": True, "recode": {-1.0: 5.0, -2.0:6.0}},
    {"name": "pctEmph_Thirona_P1",  "task": "regression",  "category": "cross_sectional"},
    {"name": "PRM_pct_airtrapping_Thirona_P1",  "task": "regression",  "category": "cross_sectional"},
    {"name": "FEV1_post_P1",     "task": "regression",     "category": "cross_sectional"},
    {"name": "FEV1_FVC_post_P1",  "task": "regression", "category": "cross_sectional"},
    {"name": "Pi10_Thirona_P1",  "task": "regression", "category": "cross_sectional"},
    {"name": "WallAreaPct_seg_Thirona_P1",  "task": "regression", "category": "cross_sectional"},
    {"name": "Change_P1_P2_FEV1_ml_yr",  "task": "regression", "category": "longitudinal"},
    {"name": "FEV1_post_P2",  "task": "regression", "category": "longitudinal"},
    {"name": "Change_P1_P2_Gold_class",  "task": "multiclass", "category": "longitudinal", "recode": {-1.0: 2.0}},
    {"name": "Pi10_Thirona_P2",  "task": "regression", "category": "longitudinal"},
]

TRAIN_FRAC = 0.5
VAL_FRAC = 0.1
TEST_FRAC = 0.4

LOW_PREVALENCE_THRESHOLD = 0.05     # binary anchor: switch ratios below this minority-class fraction
LOW_PREVALENCE_TRAIN_FRAC = 0.6
LOW_PREVALENCE_VAL_FRAC = 0.1
LOW_PREVALENCE_TEST_FRAC = 0.3

REGRESSION_BINS = 5                 # quantile bins used to stratify a regression anchor
STRATIFY_REGRESSION = True          # set False to disable quantile-bin stratification for regression anchors

SEED = 42
OUTPUT_DIR = "/home/km2347/3D-MAE-MedImaging/example_inputs/splits"

# --------------------------------------------------------------------------
# Step 1: scan .npz embeddings
# --------------------------------------------------------------------------
 
print(f"Scanning {NPZ_DIR} for '{NPZ_GLOB}'...")
npz_paths = sorted(Path(NPZ_DIR).rglob(NPZ_GLOB))
 
if SCAN_TYPE:
    other = "EXP" if SCAN_TYPE == "INSP" else "INSP"
    n_before = len(npz_paths)
    npz_paths = [p for p in npz_paths if SCAN_TYPE in p.name and other not in p.name]
    print(f"scan_type={SCAN_TYPE}: kept {len(npz_paths)}/{n_before} files")
else:
    print(f"{len(npz_paths)} files found (no scan_type filtering)")
 
sids, paths = [], []
n_skipped_wrong_key = n_skipped_error = n_skipped_dupe = 0
seen_sids = set()
for p in npz_paths:
    sid = p.stem.split("_")[0]
    if sid in seen_sids:
        n_skipped_dupe += 1
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
    sids.append(sid)
    paths.append(str(p.resolve()))
    seen_sids.add(sid)
 
print(f"Loaded {len(sids)} valid embeddings "
      f"({n_skipped_wrong_key} skipped: wrong key/shape | {n_skipped_error} skipped: load error | "
      f"{n_skipped_dupe} skipped: duplicate SID)")
 
df_emb = pd.DataFrame({ID_COL: sids, EMBEDDING_PATH_COL: paths})
 
# --------------------------------------------------------------------------
# Step 2: load labels
# --------------------------------------------------------------------------
 
print(f"\nLoading labels: {LABELS_CSV}")
outcome_names = [o["name"] for o in OUTCOMES]
read_fn = pd.read_excel if Path(LABELS_CSV).suffix.lower() in (".xlsx", ".xls") else pd.read_csv
df_labels = read_fn(LABELS_CSV)
df_labels[ID_COL] = df_labels[LABELS_SID_COL].astype(str)
 
missing_cols = [c for c in outcome_names if c not in df_labels.columns]
if missing_cols:
    raise ValueError(f"Outcome columns not found in {LABELS_CSV}: {missing_cols}. "
                      f"Available columns: {list(df_labels.columns)}")
 
df_labels = df_labels[[ID_COL] + outcome_names]
 
for oc in OUTCOMES:
    col = oc["name"]
    coerced = pd.to_numeric(df_labels[col], errors="coerce")
    n_non_numeric = int((coerced.isna() & df_labels[col].notna()).sum())
    if n_non_numeric > 0:
        examples = df_labels.loc[coerced.isna() & df_labels[col].notna(), col].unique()[:5]
        print(f"  WARNING: '{col}' had {n_non_numeric} non-numeric value(s) coerced to NaN, e.g.: {list(examples)}")
    df_labels[col] = coerced
 
    recode = oc.get("recode")
    if recode:
        n_recoded = int(df_labels[col].isin(recode.keys()).sum())
        df_labels[col] = df_labels[col].replace(recode)
        print(f"  '{col}': recoded {n_recoded} values via {recode}")
 
    if oc["task"] == "binary":
        uniques = sorted(df_labels[col].dropna().unique().tolist())
        if len(uniques) > 2:
            print(f"  WARNING: '{col}' is marked task=binary but has {len(uniques)} distinct non-missing "
                  f"values after coercion/recoding: {uniques}. roc_auc_score will fail on this later unless "
                  f"you add/adjust a \"recode\" dict for '{col}' in OUTCOMES to fold it down to exactly 2 values.")
 
if df_labels[ID_COL].duplicated().any():
    n_dupes = int(df_labels[ID_COL].duplicated().sum())
    print(f"  WARNING: {n_dupes} duplicate SIDs in {LABELS_CSV} -- keeping the first occurrence.")
    df_labels = df_labels.drop_duplicates(subset=ID_COL, keep="first")
 
# --------------------------------------------------------------------------
# Step 3: merge (embeddings side is the left -- keep every subject with a
# usable embedding even if some/all outcomes are missing for them)
# --------------------------------------------------------------------------
 
master = df_emb.merge(df_labels, on=ID_COL, how="left")
 
sids_without_any_label = master[master[outcome_names].isna().all(axis=1)][ID_COL].tolist()
if sids_without_any_label:
    print(f"\nWARNING: {len(sids_without_any_label)} subjects have an embedding but no matching row "
          f"in {LABELS_CSV} at all (all outcome columns NaN), e.g.: {sids_without_any_label[:5]}")
 
sids_no_embedding = set(df_labels[ID_COL]) - set(df_emb[ID_COL])
if sids_no_embedding:
    print(f"Note: {len(sids_no_embedding)} SIDs in {LABELS_CSV} had no matching .npz embedding "
          f"and are excluded from every split (e.g.: {list(sids_no_embedding)[:5]})")
 
 
# --------------------------------------------------------------------------
# Step 4: split each category, using its stratify=True outcome (if any) as
# the stratification anchor
# --------------------------------------------------------------------------
 
def three_way_split(df, strat_key, train_frac, val_frac, test_frac, seed):
    """Stratified (or, if strat_key is None / stratification fails, plain
    random) three-way split. Too few rows to meaningfully split three ways
    (e.g. n=1 or 2, which happens for small missing-anchor subgroups) all go
    to train rather than raising."""
    empty = df.iloc[0:0]
    if len(df) == 0:
        return empty, empty, empty
    if len(df) < 3:
        return df, empty, empty
 
    try:
        train_val_df, test_df = train_test_split(df, test_size=test_frac, stratify=strat_key, random_state=seed)
    except ValueError:
        train_val_df, test_df = train_test_split(df, test_size=test_frac, stratify=None, random_state=seed)
 
    if len(train_val_df) < 2:
        return train_val_df, empty, test_df
 
    strat_remaining = strat_key.loc[train_val_df.index] if strat_key is not None else None
    val_size_of_remaining = val_frac / (train_frac + val_frac)
    try:
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_of_remaining, stratify=strat_remaining, random_state=seed)
    except ValueError:
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_of_remaining, stratify=None, random_state=seed)
    return train_df, val_df, test_df
 
 
def summarize(df, outcome, task):
    avail = df[df[outcome].notna()]
    if task == "binary":
        return {"n": len(avail), "prevalence": round(avail[outcome].mean(), 4) if len(avail) else None}
    elif task == "multiclass":
        return {"n": len(avail), "class_counts": avail[outcome].value_counts().to_dict()}
    else:
        return {"n": len(avail),
                "mean": round(avail[outcome].mean(), 4) if len(avail) else None,
                "std": round(avail[outcome].std(), 4) if len(avail) else None}
 
 
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
 
categories = sorted(set(o["category"] for o in OUTCOMES))
summary_rows = []
 
print()
for category in categories:
    cat_outcomes = [o for o in OUTCOMES if o["category"] == category]
    cat_outcome_names = [o["name"] for o in cat_outcomes]
 
    sub = master[master[cat_outcome_names].notna().any(axis=1)].copy()
    if len(sub) == 0:
        print(f"[{category}] SKIPPED: no subject has a non-missing value for any outcome in this category.")
        continue
 
    anchors = [o for o in cat_outcomes if o.get("stratify")]
    if len(anchors) > 1:
        print(f"  WARNING: multiple outcomes marked stratify=True in '{category}' "
              f"({[o['name'] for o in anchors]}); using the first one.")
    anchor = anchors[0] if anchors else None
 
    if anchor is None:
        train_df, val_df, test_df = three_way_split(sub, None, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, SEED)
        regime, anchor_name = "standard (no stratification anchor)", None
    else:
        name, task = anchor["name"], anchor["task"]
        has_anchor = sub[sub[name].notna()]
        missing_anchor = sub[sub[name].isna()]
 
        train_frac, val_frac, test_frac, regime = TRAIN_FRAC, VAL_FRAC, TEST_FRAC, "standard"
        strat_key = None
        if task == "binary":
            prevalence = has_anchor[name].value_counts(normalize=True).min()
            if prevalence < LOW_PREVALENCE_THRESHOLD:
                train_frac, val_frac, test_frac, regime = (
                    LOW_PREVALENCE_TRAIN_FRAC, LOW_PREVALENCE_VAL_FRAC, LOW_PREVALENCE_TEST_FRAC, "low_prevalence")
            strat_key = has_anchor[name]
        elif task == "multiclass":
            strat_key = has_anchor[name]
        elif task == "regression" and STRATIFY_REGRESSION:
            try:
                strat_key = pd.qcut(has_anchor[name], q=REGRESSION_BINS, duplicates="drop")
            except ValueError:
                strat_key = None
 
        train_a, val_a, test_a = three_way_split(has_anchor, strat_key, train_frac, val_frac, test_frac, SEED)
        train_m, val_m, test_m = three_way_split(missing_anchor, None, train_frac, val_frac, test_frac, SEED)
        train_df = pd.concat([train_a, train_m])
        val_df = pd.concat([val_a, val_m])
        test_df = pd.concat([test_a, test_m])
        anchor_name = name
 
    keep_cols = [ID_COL, EMBEDDING_PATH_COL] + cat_outcome_names
    train_df[keep_cols].to_csv(out_dir / f"{category}_train.csv", index=False)
    val_df[keep_cols].to_csv(out_dir / f"{category}_val.csv", index=False)
    test_df[keep_cols].to_csv(out_dir / f"{category}_test.csv", index=False)
 
    anchor_note = f", stratified on '{anchor_name}'" if anchor_name else ""
    print(f"[{category}] ({regime} split{anchor_note}) "
          f"n_total={len(sub)}  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
 
    for oc in cat_outcomes:
        name, task = oc["name"], oc["task"]
        row = {"category": category, "outcome": name, "task": task, "split_regime": regime,
               "stratification_anchor": anchor_name, "n_category_total": len(sub),
               "n_category_train": len(train_df), "n_category_val": len(val_df), "n_category_test": len(test_df)}
        row.update({f"train_{k}": v for k, v in summarize(train_df, name, task).items()})
        row.update({f"val_{k}": v for k, v in summarize(val_df, name, task).items()})
        row.update({f"test_{k}": v for k, v in summarize(test_df, name, task).items()})
        summary_rows.append(row)
        print(f"    '{name}' ({task}): train_n={row['train_n']}  val_n={row['val_n']}  test_n={row['test_n']}")
 
pd.DataFrame(summary_rows).to_csv(out_dir / "split_summary.csv", index=False)
print(f"\nWrote per-category train/val/test CSVs (SID + {EMBEDDING_PATH_COL} + outcomes) "
      f"and split_summary.csv to {out_dir}")