#!/usr/bin/env python
"""Stage 6: CASIA-B sequence-length and sampling-policy analysis.

Metadata-only. Does not train, read/modify image pixels, change splits, or
remove sequences from the source manifest.

Run from project root (PowerShell):
python .\\IMPLEMENTATION\\stage6_sequence_sampling_analysis.py --stage5-dir ".\\Stage 5" --output-dir ".\\Stage 6"
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"subject_id", "view", "condition", "sequence", "split", "frame_count"}


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage6")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(out_dir / "stage6_console_log.txt", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze CASIA-B sequence lengths and candidate sampling budgets.")
    parser.add_argument("--stage5-dir", required=True, help="Folder containing stage5_sequence_manifest.csv")
    parser.add_argument("--output-dir", required=True, help="Output folder for Stage 6 reports")
    parser.add_argument("--short-threshold", type=int, default=20, help="Flag sequences shorter than this many frames (default: 20)")
    parser.add_argument("--policy-lengths", default="16,32,64", help="Candidate frame budgets, comma-separated (default: 16,32,64)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    logger = setup_logger(out_dir)
    stage5_dir = Path(args.stage5_dir).resolve()
    manifest_path = stage5_dir / "stage5_sequence_manifest.csv"
    logger.info("=" * 78)
    logger.info("STAGE 6 — CASIA-B SEQUENCE-LENGTH & SAMPLING-POLICY ANALYSIS")
    logger.info("=" * 78)
    logger.info(f"Stage 5 dir: {stage5_dir}")
    logger.info(f"Output dir: {out_dir}")
    logger.info("Training: DISABLED | Source images modified: NO | Split modified: NO")

    try:
        if not stage5_dir.is_dir():
            raise NotADirectoryError(f"Stage 5 directory not found: {stage5_dir}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Stage 5 manifest not found: {manifest_path}")
        if args.short_threshold < 1:
            raise ValueError("--short-threshold must be at least 1")
        budgets = sorted(set(int(x.strip()) for x in args.policy_lengths.split(",") if x.strip()))
        if not budgets or any(x < 1 for x in budgets):
            raise ValueError("--policy-lengths must contain positive integers, e.g. 16,32,64")

        logger.info("\n[1/4] Metadata checks")
        df = pd.read_csv(manifest_path)
        missing_cols = sorted(REQUIRED_COLUMNS - set(df.columns))
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        if df.empty:
            raise ValueError("Stage 5 manifest is empty")
        df["frame_count"] = pd.to_numeric(df["frame_count"], errors="coerce")
        if df["frame_count"].isna().any() or (df["frame_count"] <= 0).any():
            raise ValueError("frame_count contains missing, non-numeric, or non-positive values")
        df["split"] = df["split"].astype(str).str.lower()
        df["condition"] = df["condition"].astype(str).str.lower()
        if not set(df["split"]).issubset({"train", "val", "test"}):
            raise ValueError(f"Unexpected split labels: {sorted(set(df['split']))}")
        duplicates = int(df.duplicated(["subject_id", "view", "sequence"]).sum())
        leakage = int((df.groupby("subject_id")["split"].nunique() > 1).sum())
        if duplicates:
            raise ValueError(f"Duplicate sequence keys detected: {duplicates}")
        if leakage:
            raise ValueError(f"Subject split leakage detected: {leakage} subjects")
        df["frame_count"] = df["frame_count"].astype(int)
        logger.info(f"PASS: manifest rows = {len(df):,}")
        logger.info("PASS: required columns, positive frame counts, unique sequence keys, subject-disjoint split")

        logger.info("\n[2/4] Sequence-length distribution")
        grouped = df.groupby(["split", "condition"])["frame_count"]
        summary = grouped.agg(
            sequence_count="count", min_frames="min", q1_frames=lambda s: s.quantile(0.25),
            median_frames="median", q3_frames=lambda s: s.quantile(0.75),
            mean_frames="mean", max_frames="max", std_frames="std"
        ).reset_index()
        short_counts = (df.assign(is_short=df["frame_count"] < args.short_threshold)
                        .groupby(["split", "condition"])["is_short"].sum().reset_index(name="below_short_threshold"))
        summary = summary.merge(short_counts, on=["split", "condition"], how="left")
        summary["short_threshold_frames"] = args.short_threshold
        summary_path = out_dir / "stage6_sequence_length_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(summary.to_string(index=False))
        logger.info(f"Saved: {summary_path}")

        logger.info("\n[3/4] Short-sequence audit and candidate sampling budgets")
        short_df = df.loc[df["frame_count"] < args.short_threshold].copy()
        short_df = short_df.sort_values(["split", "condition", "frame_count", "subject_id", "sequence", "view"])
        short_path = out_dir / "stage6_short_sequence_audit.csv"
        short_df.to_csv(short_path, index=False)
        logger.info(f"Sequences shorter than {args.short_threshold} frames: {len(short_df):,} / {len(df):,}")
        logger.info(f"Saved: {short_path}")

        policy_rows = []
        for split in ["train", "val", "test"]:
            part = df[df["split"] == split]
            counts = part["frame_count"]
            for budget in budgets:
                enough = int((counts >= budget).sum())
                n = len(part)
                policy_rows.append({
                    "split": split,
                    "candidate_frame_budget": budget,
                    "sequence_count": int(n),
                    "sequences_with_at_least_budget_frames": enough,
                    "sequences_shorter_than_budget": int((counts < budget).sum()),
                    "fraction_with_at_least_budget_frames": float(enough / n) if n else None,
                    "fraction_shorter_than_budget": float((counts < budget).mean()) if n else None,
                    "note": "Metadata capacity only; no frames selected and no optimal policy inferred."
                })
        policy_df = pd.DataFrame(policy_rows)
        policy_path = out_dir / "stage6_candidate_sampling_summary.csv"
        policy_df.to_csv(policy_path, index=False)
        logger.info(policy_df.to_string(index=False))
        logger.info(f"Saved: {policy_path}")

        logger.info("\n[4/4] Final status")
        report = {
            "stage": 6,
            "status": "PASS",
            "input_manifest": str(manifest_path),
            "sequence_count": int(len(df)),
            "duplicate_sequence_keys": duplicates,
            "subjects_with_split_leakage": leakage,
            "subjects_by_split": {str(k): int(v) for k, v in df.groupby("split")["subject_id"].nunique().items()},
            "frame_count_min": int(df["frame_count"].min()),
            "frame_count_median": float(df["frame_count"].median()),
            "frame_count_mean": float(df["frame_count"].mean()),
            "frame_count_max": int(df["frame_count"].max()),
            "short_sequence_threshold": int(args.short_threshold),
            "short_sequence_count": int(len(short_df)),
            "short_sequence_fraction": float(len(short_df) / len(df)),
            "candidate_frame_budgets": budgets,
            "interpretation": "Candidate budgets are compared only for frame-count capacity. This does not select frames or establish an optimal sampling policy.",
            "training_performed": False,
            "source_images_modified": False,
            "split_modified": False
        }
        report_path = out_dir / "stage6_sampling_policy_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"Report saved: {report_path}")
        logger.info("Status: PASS")
        logger.info("No training; source images and subject split unchanged.")
        return 0
    except Exception as exc:
        logger.info("\nSTAGE 6 FAILED")
        logger.info(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
