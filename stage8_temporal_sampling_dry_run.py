#!/usr/bin/env python
"""
Stage 8 — Deterministic temporal sampling specification and dry-run audit.

This stage tests candidate deterministic temporal sampling rules using sequence
metadata only. It does NOT read frame pixels, create image tensors, train models,
change the subject split, or remove sequences from the source manifest.

The chosen default rule for this dry-run is uniform temporal sampling with
rounded, evenly spaced indices over the complete sequence. For sequences
shorter than the requested budget, the dry run reports that they need a
separately documented fallback (repeat/loop, use all frames, or exclude); it
does not silently choose one.

Run from project root (adjust paths if your folders are elsewhere):
  python .\IMPLEMENTATION\stage8_temporal_sampling_dry_run.py --stage5-dir ".\IMPLEMENTATION\Stage 5" --stage7-dir ".\IMPLEMENTATION\Stage 7" --output-dir ".\IMPLEMENTATION\Stage 8"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"subject_id", "view", "condition", "sequence", "split", "frame_count"}


def make_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage8")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(out_dir / "stage8_console_log.txt", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def uniform_indices(frame_count: int, budget: int) -> list[int]:
    """Zero-based evenly spaced indices, inclusive of endpoints when budget > 1."""
    if frame_count < 1 or budget < 1:
        raise ValueError("frame_count and budget must be positive")
    if budget == 1:
        return [0]
    # This function is only used for sequences with enough frames.
    if frame_count < budget:
        raise ValueError("Cannot sample unique uniform indices when frame_count < budget")
    values = np.linspace(0, frame_count - 1, num=budget)
    indices = np.rint(values).astype(int).tolist()
    if len(set(indices)) != budget:
        raise RuntimeError("Rounded uniform indices are not unique")
    return indices


def main() -> int:
    ap = argparse.ArgumentParser(description="Dry-run deterministic temporal sampling policies.")
    ap.add_argument("--stage5-dir", required=True)
    ap.add_argument("--stage7-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--audit-sequences", type=int, default=1000,
                    help="Maximum sequences to write to the index-level audit (default 1000). Use 0 for all.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    log = make_logger(out_dir)
    stage5_dir = Path(args.stage5_dir).resolve()
    stage7_dir = Path(args.stage7_dir).resolve()
    log.info("=" * 78)
    log.info("STAGE 8 — TEMPORAL SAMPLING SPECIFICATION & DRY RUN")
    log.info("=" * 78)
    log.info(f"Stage 5 dir: {stage5_dir}")
    log.info(f"Stage 7 dir: {stage7_dir}")
    log.info(f"Output dir: {out_dir}")
    log.info(f"Candidate budget: {args.budget}")
    log.info("Training: DISABLED | Source images modified: NO | Split modified: NO")

    try:
        if args.budget < 2:
            raise ValueError("--budget must be >= 2 for temporal coverage audit")
        if args.audit_sequences < 0:
            raise ValueError("--audit-sequences must be >= 0")
        manifest_path = stage5_dir / "stage5_sequence_manifest.csv"
        coverage_path = stage7_dir / "stage7_condition_coverage_by_budget.csv"
        for p in (manifest_path, coverage_path):
            if not p.is_file():
                raise FileNotFoundError(f"Required input not found: {p}")

        log.info("\n[1/4] Validate source metadata and Stage 7 evidence")
        df = pd.read_csv(manifest_path)
        missing = sorted(REQUIRED - set(df.columns))
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")
        df["frame_count"] = pd.to_numeric(df["frame_count"], errors="raise").astype(int)
        df["split"] = df["split"].astype(str).str.lower()
        df["condition"] = df["condition"].astype(str).str.lower()
        if df.empty or (df["frame_count"] <= 0).any():
            raise ValueError("Manifest is empty or contains non-positive frame counts")
        if df.duplicated(["subject_id", "view", "sequence"]).any():
            raise ValueError("Duplicate sequence keys detected")
        if (df.groupby("subject_id")["split"].nunique() > 1).any():
            raise ValueError("Subject split leakage detected")
        stage7 = pd.read_csv(coverage_path)
        expected = stage7[
            (stage7["candidate_frame_budget"] == args.budget)
        ]
        if expected.empty:
            raise ValueError(
                f"Stage 7 has no coverage rows for budget {args.budget}; "
                "rerun Stage 7 with this budget or choose 16, 32, or 64."
            )
        log.info(f"PASS: manifest rows = {len(df):,}")
        log.info("PASS: unique keys and subject-disjoint split")
        log.info(f"PASS: Stage 7 contains evidence for budget {args.budget}")

        log.info("\n[2/4] Compute deterministic uniform index dry-run")
        # Index-level audit only: no frame filenames are read and no images are opened.
        eligible = df[df["frame_count"] >= args.budget].copy()
        short = df[df["frame_count"] < args.budget].copy()
        if args.audit_sequences == 0:
            audit_source = df.copy()
        else:
            audit_source = df.head(min(args.audit_sequences, len(df))).copy()

        audit_rows = []
        for row in audit_source.itertuples(index=False):
            n = int(row.frame_count)
            rec = {
                "subject_id": row.subject_id,
                "view": row.view,
                "condition": row.condition,
                "sequence": row.sequence,
                "split": row.split,
                "frame_count": n,
                "budget": args.budget,
            }
            if n >= args.budget:
                idx = uniform_indices(n, args.budget)
                rec.update({
                    "sampling_status": "eligible_uniform_unique_indices",
                    "sampled_indices_zero_based": ";".join(map(str, idx)),
                    "unique_sampled_indices": len(set(idx)),
                    "first_index": idx[0],
                    "last_index": idx[-1],
                })
            else:
                rec.update({
                    "sampling_status": "short_sequence_requires_explicit_fallback",
                    "sampled_indices_zero_based": "",
                    "unique_sampled_indices": 0,
                    "first_index": "",
                    "last_index": "",
                })
            audit_rows.append(rec)
        audit = pd.DataFrame(audit_rows)
        audit_path = out_dir / "stage8_uniform_sampling_index_audit.csv"
        audit.to_csv(audit_path, index=False)
        log.info(f"Sequences audited for index generation: {len(audit):,}")
        log.info(f"Eligible for {args.budget} unique sampled indices: {len(eligible):,}")
        log.info(f"Shorter than budget and needing a declared fallback: {len(short):,}")
        log.info(f"Saved: {audit_path}")

        log.info("\n[3/4] Summarize sampling eligibility by split and condition")
        summary_rows = []
        for split in ("train", "val", "test"):
            for condition in ("normal", "bag", "coat"):
                part = df[(df["split"] == split) & (df["condition"] == condition)]
                enough = int((part["frame_count"] >= args.budget).sum())
                total = len(part)
                summary_rows.append({
                    "split": split,
                    "condition": condition,
                    "budget": args.budget,
                    "sequence_count": total,
                    "eligible_sequences": enough,
                    "short_sequences": total - enough,
                    "eligible_fraction": enough / total if total else None,
                    "short_sequence_policy": "not selected in this stage",
                })
        summary = pd.DataFrame(summary_rows)
        summary_out = out_dir / "stage8_sampling_eligibility_summary.csv"
        summary.to_csv(summary_out, index=False)
        log.info(summary.to_string(index=False))
        log.info(f"Saved: {summary_out}")

        log.info("\n[4/4] Write policy specification and status")
        report = {
            "stage": 8,
            "status": "PASS",
            "candidate_budget": args.budget,
            "candidate_sampling_rule": (
                "For sequences with at least budget frames, choose budget evenly spaced "
                "zero-based indices from 0 through frame_count-1 using numpy.linspace and "
                "round to nearest integer. This is a deterministic dry-run specification, "
                "not yet a validated scientific optimum."
            ),
            "short_sequence_policy": (
                "Unresolved by design. Do not silently repeat frames, pad, or exclude. "
                "Select and justify a fallback before actual model input construction."
            ),
            "sequence_count": int(len(df)),
            "eligible_sequences": int(len(eligible)),
            "short_sequences": int(len(short)),
            "audit_rows_written": int(len(audit)),
            "source_images_opened": False,
            "source_images_modified": False,
            "training_performed": False,
            "subject_split_modified": False,
            "sequences_removed": False,
        }
        report_path = out_dir / "stage8_temporal_sampling_specification.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info(f"Report saved: {report_path}")
        log.info("Status: PASS")
        log.info("No training; source images and subject split unchanged.")
        return 0
    except Exception as exc:
        log.info("\nSTAGE 8 FAILED")
        log.info(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
