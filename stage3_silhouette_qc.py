#!/usr/bin/env python
"""
Stage 3: CASIA-B silhouette quality-control audit.
Validation only: does not train models, edit images, or delete files.

Recommended first run: sample a few frames per sequence.
Optional --full-scan checks every PNG referenced by the Stage 2 validated manifest.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


def parse_args():
    p = argparse.ArgumentParser(description="Audit CASIA-B silhouette PNG quality.")
    p.add_argument("--stage2-dir", required=True, help="Folder containing Stage 2 validation outputs.")
    p.add_argument("--output-dir", default="Stage 3", help="Where Stage 3 reports will be saved.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-sequence", type=int, default=2,
                   help="Number of frame files sampled per sequence in normal mode.")
    p.add_argument("--max-sequences", type=int, default=300,
                   help="Maximum sequences sampled in normal mode; use 0 for all sequences.")
    p.add_argument("--full-scan", action="store_true",
                   help="Check every PNG in every sequence path (can take a long time).")
    p.add_argument("--max-foreground-occupancy", type=float, default=0.95)
    p.add_argument("--min-foreground-occupancy", type=float, default=0.001)
    return p.parse_args()


def first_existing_column(df, candidates, required=True):
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise ValueError(
            f"Could not find any expected column among {candidates}. "
            f"Manifest columns are: {list(df.columns)}"
        )
    return None


def resolve_path(value, root):
    p = Path(str(value))
    if p.is_absolute():
        return p
    return (root / p).resolve()


def image_stats(path, min_occ, max_occ):
    """Read and inspect one image without changing it."""
    result = {
        "image_path": str(path),
        "read_ok": False,
        "width": None,
        "height": None,
        "mode": None,
        "unique_values_sampled": None,
        "is_grayscale_mode": None,
        "is_binary_like": None,
        "foreground_occupancy": None,
        "empty_or_nearly_empty": None,
        "occupancy_outlier": None,
        "error": "",
    }
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            result["mode"] = im.mode
            gray = im.convert("L")
            arr = np.asarray(gray)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"Unexpected image array shape: {arr.shape}")
        result["read_ok"] = True
        result["width"] = int(arr.shape[1])
        result["height"] = int(arr.shape[0])
        result["is_grayscale_mode"] = result["mode"] in ("1", "L", "I", "I;16")
        # Downsample only for statistics on large frames.
        small = arr[::max(1, arr.shape[0] // 128), ::max(1, arr.shape[1] // 128)]
        vals = np.unique(small)
        result["unique_values_sampled"] = int(len(vals))
        result["is_binary_like"] = bool(len(vals) <= 2)
        # Estimate foreground using non-zero pixels. This is a diagnostic, not a universal
        # assumption that every CASIA-B mask must be encoded in exactly this way.
        occupancy = float(np.count_nonzero(arr) / arr.size)
        result["foreground_occupancy"] = occupancy
        result["empty_or_nearly_empty"] = bool(occupancy < min_occ)
        result["occupancy_outlier"] = bool(
            occupancy < min_occ or occupancy > max_occ
        )
    except (OSError, ValueError, UnidentifiedImageError, SyntaxError) as e:
        result["error"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def main():
    args = parse_args()
    started = time.time()
    stage2_dir = Path(args.stage2_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 84)
    log("STAGE 3 — CASIA-B SILHOUETTE QUALITY CONTROL")
    log("=" * 84)
    log(f"Started at  : {datetime.now().astimezone().isoformat(timespec='seconds')}")
    log(f"Stage 2 dir : {stage2_dir}")
    log(f"Output dir  : {out_dir}")
    log("Training    : DISABLED")
    log("Source edits: NONE")

    manifest_path = stage2_dir / "stage2_manifest_validated.csv"
    if not stage2_dir.is_dir():
        raise NotADirectoryError(f"Stage 2 directory does not exist: {stage2_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing validated manifest: {manifest_path}")

    df = pd.read_csv(manifest_path)
    path_col = first_existing_column(df, [
        "sequence_path", "sequence_dir", "sequence_directory", "directory_path", "path"
    ])
    subject_col = first_existing_column(df, ["subject_id", "subject", "person_id"], required=False)
    condition_col = first_existing_column(df, ["condition", "condition_label"], required=False)
    view_col = first_existing_column(df, ["view", "view_id", "viewpoint"], required=False)
    seq_col = first_existing_column(df, ["sequence", "sequence_label", "sequence_name"], required=False)
    first_col = first_existing_column(df, ["first_frame", "first_frame_path", "first_frame_file"], required=False)
    last_col = first_existing_column(df, ["last_frame", "last_frame_path", "last_frame_file"], required=False)

    log(f"Manifest rows: {len(df)}")
    log(f"Sequence path column: {path_col}")
    log(f"Manifest columns: {list(df.columns)}")

    rng = random.Random(args.seed)
    indices = list(range(len(df)))
    if not args.full_scan and args.max_sequences > 0 and len(indices) > args.max_sequences:
        indices = sorted(rng.sample(indices, args.max_sequences))
    work = df.iloc[indices].copy()
    log(f"Sequences selected: {len(work)} ({'FULL SCAN' if args.full_scan else 'sampled mode'})")

    records = []
    missing_dirs = 0
    missing_frames = 0
    total_candidates = 0
    sequence_summaries = []

    for count, (_, row) in enumerate(work.iterrows(), start=1):
        seq_dir = resolve_path(row[path_col], stage2_dir.parent)
        meta = {
            "manifest_row_index": int(row.name),
            "subject_id": str(row[subject_col]) if subject_col else "",
            "condition": str(row[condition_col]) if condition_col else "",
            "view": str(row[view_col]) if view_col else "",
            "sequence": str(row[seq_col]) if seq_col else "",
            "sequence_path": str(seq_dir),
        }
        if not seq_dir.is_dir():
            missing_dirs += 1
            records.append({**meta, "image_path": "", "read_ok": False,
                            "error": "Sequence directory missing"})
            continue

        if args.full_scan:
            candidates = sorted(p for p in seq_dir.iterdir()
                                if p.is_file() and p.suffix.lower() == ".png")
        else:
            all_pngs = sorted(p for p in seq_dir.iterdir()
                              if p.is_file() and p.suffix.lower() == ".png")
            if not all_pngs:
                candidates = []
            else:
                wanted = min(max(1, args.samples_per_sequence), len(all_pngs))
                if wanted == len(all_pngs):
                    candidates = all_pngs
                else:
                    chosen_idx = sorted(set(
                        [0, len(all_pngs) - 1] if wanted >= 2 else
                        [rng.randrange(len(all_pngs))]
                    ))
                    while len(chosen_idx) < wanted:
                        chosen_idx.append(rng.randrange(len(all_pngs)))
                        chosen_idx = sorted(set(chosen_idx))
                    candidates = [all_pngs[i] for i in chosen_idx[:wanted]]

        if not candidates:
            missing_frames += 1
            records.append({**meta, "image_path": "", "read_ok": False,
                            "error": "No PNG frames found in sequence directory"})
        for path in candidates:
            total_candidates += 1
            stats = image_stats(path, args.min_foreground_occupancy,
                               args.max_foreground_occupancy)
            records.append({**meta, **stats})

        if count % 100 == 0 or count == len(work):
            log(f"Progress: {count}/{len(work)} sequences processed")

    frame_df = pd.DataFrame(records)
    frame_csv = out_dir / "stage3_frame_qc.csv"
    frame_df.to_csv(frame_csv, index=False)

    valid = frame_df[frame_df["read_ok"] == True] if not frame_df.empty else frame_df
    bad_read = int((frame_df["read_ok"] == False).sum()) if not frame_df.empty else 0
    occupancy_outliers = int(valid["occupancy_outlier"].fillna(False).sum()) if not valid.empty else 0
    binary_count = int(valid["is_binary_like"].fillna(False).sum()) if not valid.empty else 0
    dim_counts = valid.groupby(["width", "height"]).size().reset_index(name="image_count") if not valid.empty else pd.DataFrame(columns=["width","height","image_count"])
    dim_csv = out_dir / "stage3_dimension_summary.csv"
    dim_counts.to_csv(dim_csv, index=False)

    report = {
        "stage": 3,
        "title": "CASIA-B silhouette quality control",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage2_manifest": str(manifest_path),
        "manifest_rows": int(len(df)),
        "sequences_selected": int(len(work)),
        "mode": "full_scan" if args.full_scan else "sampled",
        "samples_per_sequence_requested": int(args.samples_per_sequence),
        "max_sequences_setting": int(args.max_sequences),
        "images_checked": int(total_candidates),
        "successful_image_reads": int(len(valid)),
        "failed_or_missing_image_records": bad_read,
        "missing_sequence_directories": int(missing_dirs),
        "sequences_without_png_frames": int(missing_frames),
        "binary_like_images": binary_count,
        "dimension_counts": [
            {"width": int(r.width), "height": int(r.height), "image_count": int(r.image_count)}
            for r in dim_counts.itertuples(index=False)
        ],
        "foreground_occupancy_outliers": occupancy_outliers,
        "occupancy_thresholds": {
            "minimum": float(args.min_foreground_occupancy),
            "maximum": float(args.max_foreground_occupancy)
        },
        "notes": [
            "Sampled mode is a screening audit, not proof that every frame is valid.",
            "Non-zero foreground occupancy is a diagnostic assumption; inspect sample images before setting exclusions.",
            "No images were modified or deleted. No model training was performed."
        ],
        "elapsed_seconds": round(time.time() - started, 2)
    }
    report_path = out_dir / "stage3_qc_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_path = out_dir / "stage3_console_log.txt"
    log("QC output saved: " + str(frame_csv))
    log("Dimension summary saved: " + str(dim_csv))
    log("Report saved: " + str(report_path))
    log(f"Images checked: {total_candidates}")
    log(f"Successful reads: {len(valid)}")
    log(f"Failed/missing records: {bad_read}")
    log(f"Foreground occupancy outliers: {occupancy_outliers}")
    log(f"Elapsed seconds: {report['elapsed_seconds']}")
    log("STAGE 3 SCREENING FINISHED")
    log("Interpret findings before defining any exclusions or preprocessing.")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nSTAGE 3 FAILED")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
