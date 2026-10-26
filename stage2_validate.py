#!/usr/bin/env python3
"""
Stage 2 — CASIA-B metadata validation and experiment setup.

This stage:
  1. Loads Stage 1 manifest, subject summary, split, config, and audit summary.
  2. Validates required columns, labels, views, split sizes, and subject disjointness.
  3. Checks duplicate sequence paths and (subject, view, sequence) keys.
  4. Optionally checks sequence directory and first/last frame paths on disk.
  5. Saves a validation report and split-annotated manifest.

It does NOT train a model, extract embeddings, or modify source images.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_VIEWS = {
    "000", "018", "036", "054", "072", "090",
    "108", "126", "144", "162", "180"
}
EXPECTED_CONDITIONS = {"normal", "bag", "coat"}
EXPECTED_SEQUENCES = {
    "bg-01", "bg-02", "cl-01", "cl-02",
    "nm-01", "nm-02", "nm-03", "nm-04", "nm-05", "nm-06"
}
EXPECTED_SPLITS = {"train": 64, "val": 10, "test": 50}
REQUIRED_FILES = [
    "casia_b_sequence_manifest.csv",
    "casia_b_subject_summary.csv",
    "casia_b_subject_split.csv",
    "stage1_config.json",
    "stage1_audit_summary.json",
]
REQUIRED_MANIFEST_COLUMNS = {
    "subject_id", "view", "condition", "sequence", "sequence_path",
    "frame_count", "extensions", "first_frame", "last_frame"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2 CASIA-B metadata validation; no model training."
    )
    parser.add_argument(
        "--stage1-dir", required=True,
        help="Directory containing Stage 1 CSV and JSON outputs."
    )
    parser.add_argument(
        "--output-dir", default="Stage 2",
        help='Directory for Stage 2 outputs. Default: "./Stage 2".'
    )
    parser.add_argument(
        "--skip-path-check", action="store_true",
        help="Skip checking sequence/first-frame/last-frame paths on disk."
    )
    parser.add_argument(
        "--max-path-checks", type=int, default=0,
        help="Optional maximum manifest rows to path-check (0 means all rows)."
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required Stage 1 file not found: {path}")


def path_exists(value):
    try:
        return Path(str(value)).exists()
    except (TypeError, ValueError, OSError):
        return False


def main():
    args = parse_args()
    stage1_dir = Path(args.stage1_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "stage2_console_log.txt"
    old_stdout, old_stderr = sys.stdout, sys.stderr
    status = "failed"

    with log_path.open("w", encoding="utf-8") as log:
        class Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for stream in self.streams:
                    stream.write(data)
                    stream.flush()
            def flush(self):
                for stream in self.streams:
                    stream.flush()

        sys.stdout = Tee(old_stdout, log)
        sys.stderr = Tee(old_stderr, log)
        started = datetime.now().astimezone().isoformat(timespec="seconds")

        try:
            print("=" * 88)
            print("STAGE 2 — CASIA-B DATA VALIDATION AND EXPERIMENT SETUP")
            print("=" * 88)
            print("Started at  :", started)
            print("Stage 1 dir :", stage1_dir)
            print("Output dir  :", output_dir)
            print("Seed        :", args.seed)
            print("Training    : DISABLED (this stage is validation only)")
            print()

            if not stage1_dir.is_dir():
                raise NotADirectoryError(f"Stage 1 directory does not exist: {stage1_dir}")
            for filename in REQUIRED_FILES:
                require_file(stage1_dir / filename)
            print(f"[1/8] Required Stage 1 files found: {len(REQUIRED_FILES)}/{len(REQUIRED_FILES)}")

            random.seed(args.seed)
            np.random.seed(args.seed)

            manifest = pd.read_csv(stage1_dir / "casia_b_sequence_manifest.csv")
            subject_summary = pd.read_csv(stage1_dir / "casia_b_subject_summary.csv")
            split_df = pd.read_csv(stage1_dir / "casia_b_subject_split.csv")
            with (stage1_dir / "stage1_config.json").open(encoding="utf-8") as f:
                config = json.load(f)
            with (stage1_dir / "stage1_audit_summary.json").open(encoding="utf-8") as f:
                audit = json.load(f)

            print("\n[2/8] Loaded artifacts")
            print("Manifest shape       :", manifest.shape)
            print("Subject summary shape:", subject_summary.shape)
            print("Split table shape    :", split_df.shape)

            missing_cols = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
            if missing_cols:
                raise ValueError(f"Manifest missing columns: {sorted(missing_cols)}")
            if not {"subject_id", "split"}.issubset(split_df.columns):
                raise ValueError("Split file must contain subject_id and split columns.")
            if manifest.empty:
                raise ValueError("Manifest is empty.")
            for df in (manifest, subject_summary, split_df):
                if "subject_id" in df.columns:
                    df["subject_id"] = df["subject_id"].astype(str).str.zfill(3)

            manifest["view"] = manifest["view"].astype(str).str.zfill(3)
            manifest["condition"] = manifest["condition"].astype(str).str.lower()
            manifest["sequence"] = manifest["sequence"].astype(str).str.lower()
            manifest["frame_count"] = pd.to_numeric(manifest["frame_count"], errors="coerce")

            if manifest["subject_id"].isna().any() or manifest["sequence_path"].isna().any():
                raise ValueError("Manifest contains missing subject IDs or sequence paths.")
            if manifest["frame_count"].isna().any() or (manifest["frame_count"] <= 0).any():
                raise ValueError("Manifest has invalid or non-positive frame counts.")
            print("PASS: Required columns and basic metadata.")

            print("\n[3/8] Label and viewpoint coverage")
            observed_views = set(manifest["view"].unique())
            observed_conditions = set(manifest["condition"].unique())
            observed_sequences = set(manifest["sequence"].unique())
            missing_views = sorted(EXPECTED_VIEWS - observed_views)
            missing_conditions = sorted(EXPECTED_CONDITIONS - observed_conditions)
            missing_sequences = sorted(EXPECTED_SEQUENCES - observed_sequences)
            print("Observed views:", ", ".join(sorted(observed_views)))
            print("Missing expected views:", missing_views or "none")
            print("Observed conditions:", sorted(observed_conditions))
            print("Condition counts:\n", manifest["condition"].value_counts().to_string())
            print("Sequence label counts:\n", manifest["sequence"].value_counts().sort_index().to_string())
            if "UNKNOWN" in observed_conditions or missing_views or missing_conditions or missing_sequences:
                raise ValueError(
                    f"Label coverage failed. Unknown condition present="
                    f"{'UNKNOWN' in observed_conditions}; missing views={missing_views}; "
                    f"missing conditions={missing_conditions}; missing sequence labels={missing_sequences}"
                )
            print("PASS: Expected conditions, views, and sequence labels.")

            print("\n[4/8] Subject split validation")
            if split_df["subject_id"].duplicated().any():
                raise ValueError("Duplicate subject rows found in split table.")
            actual_split_names = set(split_df["split"].astype(str).unique())
            if not actual_split_names.issubset(set(EXPECTED_SPLITS)):
                raise ValueError(f"Unexpected split names: {sorted(actual_split_names - set(EXPECTED_SPLITS))}")
            split_counts = split_df["split"].value_counts().to_dict()
            for name, expected in EXPECTED_SPLITS.items():
                actual = int(split_counts.get(name, 0))
                print(f"{name:>5}: {actual} (expected {expected})")
                if actual != expected:
                    raise ValueError(f"{name} split has {actual} subjects; expected {expected}.")
            manifest_subjects = set(manifest["subject_id"].unique())
            split_subjects = set(split_df["subject_id"].unique())
            if manifest_subjects != split_subjects:
                raise ValueError(
                    f"Manifest/split subject mismatch: manifest-only={sorted(manifest_subjects - split_subjects)}, "
                    f"split-only={sorted(split_subjects - manifest_subjects)}"
                )
            split_sets = {name: set(split_df.loc[split_df["split"] == name, "subject_id"])
                          for name in EXPECTED_SPLITS}
            if not (split_sets["train"].isdisjoint(split_sets["val"])
                    and split_sets["train"].isdisjoint(split_sets["test"])
                    and split_sets["val"].isdisjoint(split_sets["test"])):
                raise ValueError("Subject leakage detected between splits.")
            print("PASS: Split counts and subject-disjointness.")

            print("\n[5/8] Duplicate checks")
            duplicate_paths = int(manifest["sequence_path"].duplicated().sum())
            key_cols = ["subject_id", "view", "sequence"]
            duplicate_keys = int(manifest.duplicated(subset=key_cols).sum())
            print("Duplicate sequence paths:", duplicate_paths)
            print("Duplicate (subject, view, sequence) keys:", duplicate_keys)
            if duplicate_paths or duplicate_keys:
                raise ValueError("Duplicate sequence paths or sequence keys detected.")

            manifest_with_split = manifest.merge(
                split_df[["subject_id", "split"]], on="subject_id",
                how="left", validate="many_to_one"
            )
            if manifest_with_split["split"].isna().any():
                raise ValueError("Some manifest rows did not receive a split label.")

            print("\n[6/8] On-disk path validation")
            path_check_rows = 0
            missing_sequence_dirs = missing_first = missing_last = 0
            if args.skip_path_check:
                print("Skipped by command-line option.")
            else:
                check_df = manifest
                if args.max_path_checks > 0:
                    check_df = manifest.head(args.max_path_checks)
                path_check_rows = len(check_df)
                dir_ok = check_df["sequence_path"].map(path_exists)
                first_ok = check_df["first_frame"].map(path_exists)
                last_ok = check_df["last_frame"].map(path_exists)
                missing_sequence_dirs = int((~dir_ok).sum())
                missing_first = int((~first_ok).sum())
                missing_last = int((~last_ok).sum())
                print("Rows checked:", path_check_rows)
                print("Missing sequence directories:", missing_sequence_dirs)
                print("Missing first-frame paths:", missing_first)
                print("Missing last-frame paths:", missing_last)
                if missing_sequence_dirs or missing_first or missing_last:
                    examples = check_df.loc[~(dir_ok & first_ok & last_ok),
                        ["subject_id", "sequence", "view", "sequence_path", "first_frame", "last_frame"]]
                    examples.to_csv(output_dir / "stage2_missing_paths.csv", index=False, encoding="utf-8-sig")
                    raise FileNotFoundError(
                        "Some paths are missing in this runtime. If the dataset is on another machine/server, "
                        "set the correct --stage1-dir and ensure manifest paths refer to this environment. "
                        "Examples saved to stage2_missing_paths.csv."
                    )
                print("PASS: Checked paths exist.")

            print("\n[7/8] Summaries and outputs")
            coverage = (manifest_with_split.groupby(["split", "condition"])
                        .agg(subjects=("subject_id", "nunique"),
                             sequences=("sequence_path", "count"),
                             frames=("frame_count", "sum")).reset_index())
            coverage.to_csv(output_dir / "stage2_condition_split_coverage.csv",
                            index=False, encoding="utf-8-sig")
            manifest_with_split.to_csv(output_dir / "stage2_manifest_validated.csv",
                                        index=False, encoding="utf-8-sig")

            report = {
                "stage": "Stage 2 — CASIA-B data validation and experiment setup",
                "started_at": started,
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "stage1_dir": str(stage1_dir),
                "output_dir": str(output_dir),
                "seed": args.seed,
                "manifest_rows": int(len(manifest)),
                "unique_subjects": int(manifest["subject_id"].nunique()),
                "total_recorded_frames": int(manifest["frame_count"].sum()),
                "conditions": sorted(manifest["condition"].unique().tolist()),
                "views": sorted(manifest["view"].unique().tolist()),
                "sequence_labels": sorted(manifest["sequence"].unique().tolist()),
                "split_subject_counts": {str(k): int(v) for k, v in split_counts.items()},
                "duplicate_sequence_paths": duplicate_paths,
                "duplicate_sequence_keys": duplicate_keys,
                "path_check_skipped": bool(args.skip_path_check),
                "path_check_rows": path_check_rows,
                "missing_sequence_dirs": missing_sequence_dirs,
                "missing_first_frames": missing_first,
                "missing_last_frames": missing_last,
                "status": "PASS"
            }
            with (output_dir / "stage2_data_validation_report.json").open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print("Validated manifest saved:", output_dir / "stage2_manifest_validated.csv")
            print("Coverage table saved    :", output_dir / "stage2_condition_split_coverage.csv")
            print("Validation report saved :", output_dir / "stage2_data_validation_report.json")
            print("\n[8/8] STAGE 2 FINISHED")
            print("Status: PASS")
            print("No model training was performed. Source images were not modified.")

        except Exception as exc:
            print("\nSTAGE 2 FAILED")
            print(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    print("Console log saved:", log_path)


if __name__ == "__main__":
    main()
