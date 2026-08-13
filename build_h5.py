# Code to create .h5 file for COPDGene data to be used in the TANGERINE model
# Kalysta Makimoto

"""
build_h5.py — Convert COPDGene .nrrd CT volumes (Phase 1 / STD kernel / INSP)
into a single HDF5 file of fixed-size (256, 256, 256), HU-clipped, int16 cubes,
based on the TANGERINE paper.

Resume-safe: if OUTPUT_H5 already exists, already-written sids are skipped,
so a rerun after a crash/interruption picks up where it left off.

DESIGN NOTES (accumulated from real debugging over this project):
  - Uses multiprocessing.Pool (not concurrent.futures.ProcessPoolExecutor)
    because Pool.terminate() can forcibly kill worker processes regardless
    of their state — ProcessPoolExecutor has no equivalent, which caused
    earlier stuck runs to hang indefinitely even on Ctrl+C.
  - One continuous pool handles the whole batch (no fixed chunking) — the
    pool is only ever torn down and rebuilt reactively, when a real
    per-subject timeout is detected (our proxy for "the SSHFS mount likely
    died"), not on a fixed schedule. Subjects get MAX_RETRIES attempts
    across pool rebuilds before being permanently recorded as skipped.
  - Worker processes reset their signal handlers to OS default on startup
    (worker_init). Without this, forked workers inherit the main process's
    SIGTERM handler, so pool.terminate() causes every worker to try to run
    handle_shutdown() (touching the same forked HDF5 file object
    concurrently from multiple processes) instead of just dying cleanly.
  - The main process's shutdown handler is guarded against re-entrancy: a
    second SIGINT/SIGTERM arriving while a flush/close is already in
    progress is ignored, rather than being allowed to interrupt
    h5f.flush()/h5f.close() mid-operation — an interrupted flush/close is
    exactly what corrupted the HDF5 file's internal index once already
    (a double Ctrl-C is an easy, realistic way to trigger this).
  - A progress line is logged (flushed immediately) on every single
    subject's completion, so `tail -f` on the log file is always a
    trustworthy, real-time view of what's happening — no live terminal
    session required to check on a long run.

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
MAX_TASKS_PER_CHILD = 200           # per-worker recycling within the pool (memory-creep safety net)
PER_SUBJECT_TIMEOUT_SEC = 240       # if a subject hasn't completed after this long, the
                                     # whole pool is terminated and rebuilt for the remainder
MAX_RETRIES = 500                     # attempts a timed-out subject gets before being permanently skipped
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


def worker_init():
    """Runs once when each worker process starts. Resets signal handlers to
    OS default — without this, forked workers inherit the main process's
    SIGTERM handler, so pool.terminate() causes every worker to run
    handle_shutdown() (touching the same forked HDF5 file object
    concurrently from multiple processes) instead of just dying cleanly."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)


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


def run_pool_pass(sids_to_run, h5f, df_by_sid, written, skipped, total_to_process):
    """Runs a single multiprocessing.Pool over all of sids_to_run. Returns a
    list of sids that need to be retried (timed out but under MAX_RETRIES),
    having already logged/recorded permanent skips for anything else.

    The pool is only ever terminated reactively — when a subject exceeds
    PER_SUBJECT_TIMEOUT_SEC — not on any fixed schedule. That timeout is
    treated as evidence the SSHFS mount likely died broadly (observed
    behavior: many subjects time out together, not one at a time), so the
    whole pool is torn down and everything still pending is handed back to
    the caller to retry against a fresh pool.
    """
    pool = multiprocessing.Pool(processes=N_WORKERS, maxtasksperchild=MAX_TASKS_PER_CHILD,
                                 initializer=worker_init)
    start_times = {sid: time.time() for sid in sids_to_run}
    async_results = {sid: pool.apply_async(process_one, (sid,)) for sid in sids_to_run}
    pending = dict(async_results)

    def log_progress():
        done = len(written) + len(skipped)
        pct = 100 * done / total_to_process if total_to_process else 0
        log(f"progress: {done}/{total_to_process} ({pct:.1f}%) — "
            f"{len(written)} written, {len(skipped)} skipped")

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
                    log_progress()

            timed_out = [s for s in pending if now - start_times[s] > PER_SUBJECT_TIMEOUT_SEC]
            if timed_out:
                still_pending = list(pending.keys())
                log(f"TIMEOUT: {len(timed_out)} subject(s) exceeded {PER_SUBJECT_TIMEOUT_SEC}s "
                    f"(likely mount issue) — terminating pool. "
                    f"{len(still_pending)} subjects total will be retried.")
                pool.terminate()
                pool.join()
                return still_pending  # caller decides retry vs permanent skip

            time.sleep(1)
    finally:
        try:
            pool.close()
            pool.join()
        except Exception:
            pass

    return []  # everything in this pass completed normally


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

    shutting_down = {"in_progress": False}

    def handle_shutdown(signum, frame):
        if shutting_down["in_progress"]:
            # A second signal arrived while we were already flushing/closing —
            # ignore it rather than let it interrupt h5f.flush()/h5f.close()
            # mid-operation, which can corrupt the file's internal index
            # (this is what caused an earlier corruption incident, likely
            # from a rapid double Ctrl-C).
            log(f"Signal {signum} received again during shutdown — already closing, ignoring.")
            return
        shutting_down["in_progress"] = True
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
    log(f"N_WORKERS={N_WORKERS}, per-subject timeout={PER_SUBJECT_TIMEOUT_SEC}s, max retries={MAX_RETRIES}")

    total_to_process = len(sids_to_process)
    retry_counts = {}

    try:
        remaining = sids_to_process
        while remaining:
            still_pending = run_pool_pass(remaining, h5f, df_by_sid, written, skipped, total_to_process)

            if not still_pending:
                break  # pass completed normally, nothing left to retry

            retry_batch = []
            for sid in still_pending:
                retry_counts[sid] = retry_counts.get(sid, 0) + 1
                if retry_counts[sid] > MAX_RETRIES:
                    skipped.append((sid, f"TIMEOUT after {MAX_RETRIES} retries — permanently skipped"))
                    log(f"SKIP  {sid}: exceeded max retries ({MAX_RETRIES}), giving up on this subject")
                else:
                    retry_batch.append(sid)

            if retry_batch:
                log(f"Rebuilding pool and retrying {len(retry_batch)} subject(s) "
                    f"(attempt {max(retry_counts[s] for s in retry_batch)}/{MAX_RETRIES})...")
            remaining = retry_batch
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
    # Use 'spawn' instead of the Linux default 'fork'. fork() duplicates the
    # entire parent process, including its open file descriptor table, at
    # the OS level — regardless of whether worker code ever references the
    # corresponding Python object. Every worker was silently inheriting a
    # duplicate open handle to ct_volumes.h5 purely as a fork side effect,
    # even though process_one() never touches h5f. Across dozens of pool
    # rebuild cycles in a long run, this is a plausible cause of the HDF5
    # index corruption seen even on a clean, uninterrupted completion.
    # spawn starts each worker as a genuinely fresh process with no
    # inherited file descriptors, eliminating this risk at the root.
    multiprocessing.set_start_method("spawn", force=True)
    main()