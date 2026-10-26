#!/usr/bin/env python
"""Stage 7 — Metadata-only sampling policy decision analysis for CASIA-B.

Does not read or modify source images, change splits, remove sequences, or train.
Run from project root:
  python .\\IMPLEMENTATION\\stage7_sampling_policy_decision.py --stage5-dir ".\\Stage 5" --stage6-dir ".\\Stage 6" --output-dir ".\\Stage 7"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REQUIRED_MANIFEST = {
    "subject_id", "view", "condition", "sequence", "split", "frame_count"
}
EXPECTED_SPLITS = {"train": 64, "val": 10, "test": 50}


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage7")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(output_dir / "stage7_console_log.txt", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare candidate frame budgets and produce a transparent decision aid."
    )
    parser.add_argument("--stage5-dir", required=True)
    parser.add_argument("--stage6-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", default="16,32,64")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    logger = setup_logger(output_dir)
    stage5_dir = Path(args.stage5_dir).resolve()
    stage6_dir = Path(args.stage6_dir).resolve()

    logger.info("=" * 78)
    logger.info("STAGE 7 — SAMPLING-POLICY DECISION ANALYSIS")
    logger.info("=" * 78)
    logger.info(f"Stage 5 dir: {stage5_dir}")
    logger.info(f"Stage 6 dir: {stage6_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("Training: DISABLED | Images modified: NO | Split modified: NO")

    try:
        manifest_path = stage5_dir / "stage5_sequence_manifest.csv"
        summary_path = stage6_dir / "stage6_sequence_length_summary.csv"
        candidate_path = stage6_dir / "stage6_candidate_sampling_summary.csv"
        for path in (manifest_path, summary_path, candidate_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required input not found: {path}")

        budgets = sorted(set(int(x.strip()) for x in args.budgets.split(",") if x.strip()))
        if not budgets or any(b <= 0 for b in budgets):
            raise ValueError("--budgets must contain positive integers")

        logger.info("\\n[1/4] Validate inputs and fixed subject split")
        manifest = pd.read_csv(manifest_path)
        missing = sorted(REQUIRED_MANIFEST - set(manifest.columns))
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")
        if manifest.empty:
            raise ValueError("Manifest is empty")
        manifest["split"] = manifest["split"].astype(str).str.lower()
        manifest["condition"] = manifest["condition"].astype(str).str.lower()
        manifest["frame_count"] = pd.to_numeric(manifest["frame_count"], errors="raise").astype(int)
        if (manifest["frame_count"] <= 0).any():
            raise ValueError("Non-positive frame count detected")
        if manifest.duplicated(["subject_id", "view", "sequence"]).any():
            raise ValueError("Duplicate sequence keys detected")
        per_subject = manifest.groupby("subject_id")["split"].nunique()
        if (per_subject > 1).any():
            raise ValueError("Subject leakage across splits detected")
        actual = {s: int(manifest.loc[manifest.split == s, "subject_id"].nunique())
                  for s in ("train", "val", "test")}
        if actual != EXPECTED_SPLITS:
            raise ValueError(f"Unexpected split subject counts: {actual}")
        logger.info(f"PASS: manifest rows = {len(manifest):,}")
        logger.info(f"PASS: subjects by split = {actual}")

        logger.info("\\n[2/4] Recompute candidate coverage directly from manifest")
        rows = []
        for split in ("train", "val", "test"):
            part = manifest[manifest["split"] == split]
            for condition in ("normal", "bag", "coat"):
                cpart = part[part["condition"] == condition]
                if cpart.empty:
                    logger.info(f"WARNING: no sequences for split={split}, condition={condition}")
                for budget in budgets:
                    enough = int((cpart["frame_count"] >= budget).sum())
                    total = int(len(cpart))
                    rows.append({
                        "split": split,
                        "condition": condition,
                        "candidate_frame_budget": budget,
                        "sequence_count": total,
                        "sequences_with_enough_frames": enough,
                        "sequences_shorter_than_budget": total - enough,
                        "coverage_fraction": (enough / total) if total else None,
                        "decision_status": "descriptive only; not an empirically validated optimum"
                    })
        coverage = pd.DataFrame(rows)
        coverage_out = output_dir / "stage7_condition_coverage_by_budget.csv"
        coverage.to_csv(coverage_out, index=False)
        logger.info(coverage.to_string(index=False))
        logger.info(f"Saved: {coverage_out}")

        logger.info("\\n[3/4] Compare practical trade-offs")
        overall_rows = []
        for budget in budgets:
            enough = int((manifest["frame_count"] >= budget).sum())
            total = int(len(manifest))
            overall_rows.append({
                "candidate_frame_budget": budget,
                "total_sequences": total,
                "sequences_with_enough_frames": enough,
                "sequences_shorter_than_budget": total - enough,
                "coverage_fraction": enough / total,
                "interpretation": (
                    "Lower temporal budget; broad availability" if budget <= 16 else
                    "Intermediate temporal budget; inspect temporal coverage and short-sequence handling" if budget <= 32 else
                    "Higher temporal budget; substantially more short sequences require a documented policy"
                )
            })
        overall = pd.DataFrame(overall_rows)
        overall_out = output_dir / "stage7_overall_budget_comparison.csv"
        overall.to_csv(overall_out, index=False)
        logger.info(overall.to_string(index=False))
        logger.info(f"Saved: {overall_out}")

        short = manifest.loc[manifest["frame_count"] < min(budgets)].copy()
        short_out = output_dir / "stage7_sequences_shorter_than_smallest_budget.csv"
        short.sort_values(["split", "condition", "frame_count", "subject_id"]).to_csv(short_out, index=False)
        logger.info(f"Sequences shorter than smallest budget ({min(budgets)}): {len(short):,}")
        logger.info(f"Saved: {short_out}")

        # Decision aid intentionally does not automatically choose a winning policy.
        report = {
            "stage": 7,
            "status": "PASS",
            "sequence_count": int(len(manifest)),
            "subjects_by_split": actual,
            "candidate_budgets": budgets,
            "overall_coverage": [
                {
                    "budget": int(r.candidate_frame_budget),
                    "sequences_with_enough_frames": int(r.sequences_with_enough_frames),
                    "sequences_shorter_than_budget": int(r.sequences_shorter_than_budget),
                    "coverage_fraction": float(r.coverage_fraction),
                }
                for r in overall.itertuples(index=False)
            ],
            "research_cautions": [
                "Coverage counts do not establish which budget preserves gait information best.",
                "A fixed number of frames is not itself a sampling algorithm; temporal spacing and sequence ordering matter.",
                "Do not discard short sequences solely because they fall below a candidate budget.",
                "Select and document one policy before model comparison; use the same policy across conditions and views.",
                "Any eventual model-based comparison should be designed without using test subjects to tune the policy."
            ],
            "recommended_next_decision": (
                "Review per-condition coverage, then specify a deterministic temporal sampling rule and a transparent "
                "fallback for short sequences. Do not claim an optimal budget from metadata-only counts."
            ),
            "training_performed": False,
            "source_images_modified": False,
            "subject_split_modified": False,
            "sequences_removed": False
        }
        report_out = output_dir / "stage7_sampling_policy_decision_report.json"
        report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("\\n[4/4] Final status")
        logger.info(f"Report saved: {report_out}")
        logger.info("Status: PASS")
        logger.info("No training; source images and subject split unchanged.")
        return 0
    except Exception as exc:
        logger.info("\\nSTAGE 7 FAILED")
        logger.info(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
