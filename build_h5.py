# Code to create .h5 file for COPDGene data to be used in the TANGERINE model
# Kalysta Makimoto

"""
build_h5.py — Convert COPDGene .nrrd CT volumes (Phase 1 / STD kernel / INSP)
into a single HDF5 file of fixed-size (256, 256, 256), HU-clipped, int16 cubes,
based on the TANGERINE paper.

Resume-safe: if OUTPUT_H5 already exists, already-written sids are skipped,
so a rerun after a crash/interruption picks up where it left off.

REWRITE NOTE: this version replaces concurrent.futures.ProcessPoolExecutor
with multiprocessing.Pool, processed in small chunks with a fresh pool per
chunk. Rationale, from a real stuck-run investigation:
  - ProcessPoolExecutor has no reliable way to force-kill a pool that has
    stopped making progress (its shutdown() waits for workers to exit on
    their own). multiprocessing.Pool.terminate() forcibly kills worker
    processes regardless of their state.
  - A single long-lived pool running for ~10 hours has more opportunity to
    degrade silently. Recreating the pool every CHUNK_SIZE subjects bounds
    the blast radius of any one pool going bad to a single chunk, and
    naturally gives every worker a clean restart periodically.
  - A heartbeat is logged every HEARTBEAT_SEC regardless of whether any
    subject has completed, so the log file itself is trustworthy evidence
    of whether the process is alive and progressing — no more ambiguity
    between "the job is stuck" and "my terminal view stopped updating."
  - Per-subject wall-clock timeout is enforced from the main process (via
    polling AsyncResult.ready()), not via SIGALRM inside the worker — this
    also works even if the underlying hang were ever a truly uninterruptible
    (D-state) I/O wait, since it doesn't depend on the stuck worker
    responding to anything; the chunk's pool is simply terminated and a
    fresh one started for the next chunk.

Usage:
    # Small test run first
    python build_h5.py --limit 10

    # Full run cohort
    python build_h5.py
"""

import os
import glob
import argparse
import signal
import sys
import time
import multiprocessing
import h5py
import numpy as np
import pandas as pd
import SimpleITK as sitk

# ---- Config -------------------------------------------------------------
RAW_ROOT = "/datagpu/datasets/km2347/COPDGene_master"        # mounted master to local
OUTPUT_H5 = "/home/km2347/ct_volumes.h5"       # local
LABELS_CSV = "/home/km2347/COPDGene_Data/COPDGene_P1P2P3_Flat_SM_NS_Sep24.txt"

TARGET_SIZE = (256, 256, 256)   # (D, H, W) — matches TANGERINE pretraining resolution
HU_MIN, HU_MAX = -1200, 800

N_WORKERS = 8
CHUNK_SIZE = 200                    # fresh pool every this many subjects
PER_SUBJECT_TIMEOUT_SEC = 240       # if a subject hasn't completed after this long, the
                                     # whole chunk's pool is terminated and rebuilt
HEARTBEAT_SEC = 30                  # log a liveness line at least this often, always
FLUSH_EVERY_N = 20                  # periodic HDF5 flush so a crash loses at most this many writes

ID_COLUMN = "sid"

DEMO_COLUMNS = ["gender", "Age_P1", "race", "ATS_PackYears_P1",
                "smoking_status_P1", "finalGold_P1",
                "finalGold_P2", "binary_FEV1_decline_60",
                "multiclass_FEV1_decline"]

MISSING_LOG = "/home/km2347/3D-MAE-MedImaging/missing_or_ambiguous_sids.txt"


def log(msg):
    """Timestamped, immediately-flushed print — so `tail -f` and the log file
    itself are always trustworthy about what's actually happening and when."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---- Path: Phase 1 (COPD), STD kernel, INSP only --------------------
def sid_to_nrrd_path(sid):
    """
    Naming convention: {sid}_INSP_STD_{site}_COPD/{sid}_INSP_STD_{site}_COPD.nrrd
    """
    pattern = os.path.join(RAW_ROOT, sid, f"{sid}_INSP_STD_*_COPD", f"{sid}_INSP_STD_*_COPD.nrrd")
    matches = glob.glob(pattern)
    if len(matches) == 0:
        return None, "no_match"
    if len(matches) > 1:
        return None, f"ambiguous:{matches}"
    return matches[0], None


# ---- Resample images ----------------------------------------------
def resample_to_fixed_size(sitk_image, target_size_dhw):
    original_size = sitk_image.GetSize()          # SimpleITK order: (W, H, D)
    original_spacing = sitk_image.GetSpacing()
    target_size_sitk = (target_size_dhw[2], target_size_dhw[1], target_size_dhw[0])
    new_spacing = [
        osz * osp / nsz
        for osz, osp, nsz in zip(original_size, original_spacing, target_size_sitk)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(target_size_sitk)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk.sitkBSpline)
    return resample.Execute(sitk_image)


def process_one(sid):
    """Runs in a worker process. Returns (sid, array_or_None, cid_or_None, error_or_None)."""
    nrrd_path, path_err = sid_to_nrrd_path(sid)
    if nrrd_path is None:
        return sid, None, None, path_err

    try:
        img = sitk.ReadImage(nrrd_path)
        img = resample_to_fixed_size(img, TARGET_SIZE)
        arr = sitk.GetArrayFromImage(img)                # (D, H, W)
        arr = np.clip(arr, HU_MIN, HU_MAX).astype(np.int16)
        cid = os.path.basename(nrrd_path).replace(".nrrd", "")
        return sid, arr, cid, None
    except Exception as e:
        return sid, None, None, str(e)


def write_result(h5f, df_by_sid, sid, arr, cid, written):
    dset = h5f.create_dataset(sid, data=arr, compression="gzip", compression_opts=1)
    dset.attrs["sid"] = sid
    dset.attrs["cid"] = cid
    if sid in df_by_sid.index:
        row = df_by_sid.loc[sid]
        for col in DEMO_COLUMNS:
            val = row[col]
            if pd.isna(val):
                dset.attrs[col] = "NA"
            elif isinstance(val, (int, float, np.integer, np.floating)):
                dset.attrs[col] = float(val)
            else:
                dset.attrs[col] = str(val)
    written.append(sid)


def process_chunk(sids_chunk, h5f, df_by_sid, written, skipped):
    """Runs one chunk against a fresh multiprocessing.Pool. If any subject
    exceeds PER_SUBJECT_TIMEOUT_SEC without completing, the whole pool is
    forcibly terminated (multiprocessing.Pool.terminate() kills workers
    regardless of their state) and the still-pending subjects in this chunk
    are recorded as timeouts, so the run can move on to the next chunk."""
    pool = multiprocessing.Pool(processes=N_WORKERS)
    start_times = {sid: time.time() for sid in sids_chunk}
    async_results = {sid: pool.apply_async(process_one, (sid,)) for sid in sids_chunk}
    pending = dict(async_results)
    last_heartbeat = time.time()
    last_progress = time.time()

    try:
        while pending:
            now = time.time()

            for sid, ar in list(pending.items()):
                if ar.ready():
                    try:
                        _, arr, cid, err = ar.get()
                    except Exception as e:
                        arr, cid, err = None, None, str(e)
                    if arr is None:
                        skipped.append((sid, err))
                        log(f"SKIP  {sid}: {err}")
                    else:
                        write_result(h5f, df_by_sid, sid, arr, cid, written)
                        if len(written) % FLUSH_EVERY_N == 0:
                            h5f.flush()
                    del pending[sid]
                    last_progress = now

            if now - last_heartbeat >= HEARTBEAT_SEC:
                oldest_pending_age = max((now - start_times[s] for s in pending), default=0)
                log(f"heartbeat: {len(written)} written, {len(skipped)} skipped, "
                    f"{len(pending)} pending in this chunk, "
                    f"oldest pending task age={oldest_pending_age:.0f}s, "
                    f"idle since last completion={now - last_progress:.0f}s")
                last_heartbeat = now

            timed_out = [s for s in pending if now - start_times[s] > PER_SUBJECT_TIMEOUT_SEC]
            if timed_out:
                log(f"TIMEOUT: {len(timed_out)} subject(s) exceeded {PER_SUBJECT_TIMEOUT_SEC}s: "
                    f"{timed_out} — terminating this chunk's pool and moving on")
                for sid in timed_out:
                    skipped.append((sid, f"TIMEOUT after {PER_SUBJECT_TIMEOUT_SEC}s (pool terminated)"))
                pool.terminate()
                pool.join()
                # anything else still pending in this chunk (not itself timed out
                # yet, but its worker just got killed) also can't complete now
                for sid in pending:
                    if sid not in timed_out:
                        skipped.append((sid, "pool terminated due to a sibling subject's timeout"))
                return  # abandon rest of this chunk, caller moves to the next one

            time.sleep(1)
    finally:
        try:
            pool.close()
            pool.join()
        except Exception:
            pass


# ---- Main -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="tests a subset of subjects")
    args = parser.parse_args()

    df = pd.read_csv(LABELS_CSV, sep="\t", low_memory=False)
    assert ID_COLUMN in df.columns, f"CSV must have column {ID_COLUMN!r}, found {df.columns.tolist()}"
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)

    missing_demo_cols = [c for c in DEMO_COLUMNS if c not in df.columns]
    if missing_demo_cols:
        raise ValueError(f"DEMO_COLUMNS not found in CSV: {missing_demo_cols}. "
                          f"Available columns: {df.columns.tolist()}")

    df_by_sid = df.set_index(ID_COLUMN)
    sids = df[ID_COLUMN].tolist()

    if args.limit:
        sids = sids[:args.limit]
        log(f"TEST RUN: limited to first {len(sids)} subjects")

    mode = "a" if os.path.exists(OUTPUT_H5) else "w"
    if mode == "a":
        log(f"{OUTPUT_H5} already exists — resuming, will skip already-written sids")

    written = []
    skipped = []

    os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)
    h5f = h5py.File(OUTPUT_H5, mode)

    def handle_shutdown(signum, frame):
        log(f"Received signal {signum} — flushing and closing {OUTPUT_H5} cleanly before exit...")
        try:
            h5f.flush()
            h5f.close()
            log("File closed safely. Forcing immediate exit.")
        except Exception as e:
            log(f"Warning: error during clean shutdown: {e}")
        os._exit(1)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGHUP, handle_shutdown)

    already_done = set(h5f.keys())
    sids_to_process = [s for s in sids if s not in already_done]
    log(f"{len(already_done)} already in {OUTPUT_H5}, processing {len(sids_to_process)} remaining")
    log(f"N_WORKERS={N_WORKERS}, CHUNK_SIZE={CHUNK_SIZE}, "
        f"per-subject timeout={PER_SUBJECT_TIMEOUT_SEC}s, heartbeat every {HEARTBEAT_SEC}s")

    try:
        chunks = [sids_to_process[i:i + CHUNK_SIZE] for i in range(0, len(sids_to_process), CHUNK_SIZE)]
        for chunk_idx, chunk in enumerate(chunks):
            log(f"--- chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} subjects) ---")
            process_chunk(chunk, h5f, df_by_sid, written, skipped)
            log(f"chunk {chunk_idx + 1}/{len(chunks)} done. "
                f"Running totals: {len(written)} written, {len(skipped)} skipped")
    finally:
        h5f.flush()
        h5f.close()

    log(f"Wrote {len(written)} volumes to {OUTPUT_H5}")
    log(f"Skipped {len(skipped)} subjects (missing, ambiguous, or timed out)")

    if skipped:
        with open(MISSING_LOG, "w") as f:
            for sid, reason in skipped:
                f.write(f"{sid}\t{reason}\n")
        log(f"Details written to {MISSING_LOG} — review before the full run" if args.limit
            else f"Details written to {MISSING_LOG}")


if __name__ == "__main__":
    main()