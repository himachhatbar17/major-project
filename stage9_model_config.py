#!/usr/bin/env python
"""
Stage 9 — Unsupervised gait representation model/config scaffold (NO TRAINING).

This stage moves from repeated metadata checks to implementation planning:
- loads the existing Stage 5 sequence manifest
- writes an explicit, editable Stage 9 experiment configuration
- records the current candidate policy: 32-frame uniform temporal sampling
- flags short sequences for a future explicit policy decision
- documents the model/learning objective choices that must be finalized before training

It does NOT load image pixels, create embeddings, train, or modify the dataset.

Run from project root:
  python .\IMPLEMENTATION\stage9_model_config.py --stage5-dir ".\IMPLEMENTATION\Stage 5" --output-dir ".\IMPLEMENTATION\Stage 9"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REQUIRED = {"subject_id", "view", "condition", "sequence", "split", "frame_count"}


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage9")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(output_dir / "stage9_console_log.txt", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main() -> int:
    ap = argparse.ArgumentParser(description="Create the Stage 9 model/config scaffold without training.")
    ap.add_argument("--stage5-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frame-budget", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    logger = setup_logger(output_dir)
    stage5_dir = Path(args.stage5_dir).resolve()
    manifest_path = stage5_dir / "stage5_sequence_manifest.csv"

    logger.info("=" * 78)
    logger.info("STAGE 9 — UNSUPERVISED GAIT REPRESENTATION MODEL CONFIG")
    logger.info("=" * 78)
    logger.info(f"Stage 5 dir: {stage5_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("Training: DISABLED | Image pixels loaded: NO | Dataset modified: NO")

    try:
        if args.frame_budget < 2:
            raise ValueError("--frame-budget must be >= 2")
        if args.image_size < 1 or args.embedding_dim < 1:
            raise ValueError("--image-size and --embedding-dim must be positive")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Stage 5 manifest not found: {manifest_path}")

        df = pd.read_csv(manifest_path)
        missing = sorted(REQUIRED - set(df.columns))
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")
        df["frame_count"] = pd.to_numeric(df["frame_count"], errors="raise").astype(int)
        df["split"] = df["split"].astype(str).str.lower()
        df["condition"] = df["condition"].astype(str).str.lower()
        if df.empty or (df["frame_count"] <= 0).any():
            raise ValueError("Manifest empty or invalid frame_count values found")
        if df.duplicated(["subject_id", "view", "sequence"]).any():
            raise ValueError("Duplicate sequence keys detected")
        if (df.groupby("subject_id")["split"].nunique() > 1).any():
            raise ValueError("Subject leakage across splits detected")

        split_subjects = {
            split: int(df.loc[df["split"] == split, "subject_id"].nunique())
            for split in ("train", "val", "test")
        }
        expected = {"train": 64, "val": 10, "test": 50}
        if split_subjects != expected:
            raise ValueError(f"Expected fixed subject split {expected}; found {split_subjects}")

        eligible = int((df["frame_count"] >= args.frame_budget).sum())
        short = int((df["frame_count"] < args.frame_budget).sum())
        condition_counts = {
            str(k): int(v) for k, v in df["condition"].value_counts().items()
        }

        config = {
            "stage": 9,
            "project": "Covariate-Aware, Label-Free Model Selection for Self-Supervised Gait Representations",
            "status": "CONFIG_SCAFFOLD_ONLY",
            "dataset": {
                "name": "CASIA-B",
                "manifest_path": str(manifest_path),
                "sequence_count": int(len(df)),
                "subjects_by_split": split_subjects,
                "condition_sequence_counts": condition_counts,
                "subject_disjoint_split_required": True,
                "split_is_frozen": True
            },
            "input_pipeline_candidate": {
                "image_size": [args.image_size, args.image_size],
                "input_representation": "single-channel grayscale silhouette",
                "candidate_frames_per_sequence": args.frame_budget,
                "temporal_sampling": "deterministic uniform sampling across sequence extent",
                "short_sequence_count": short,
                "eligible_sequence_count": eligible,
                "short_sequence_policy": "UNRESOLVED — decide before any training",
                "normalization": "UNRESOLVED — define from actual image representation before training",
                "augmentation": "UNRESOLVED — specify and justify before training"
            },
            "representation_model": {
                "family": "3D CNN or temporal CNN encoder",
                "encoder_architecture": "UNRESOLVED — choose after confirming research protocol",
                "embedding_dimension_candidate": args.embedding_dim,
                "projection_head": "UNRESOLVED",
                "temporal_aggregation": "UNRESOLVED"
            },
            "self_supervised_objective": {
                "primary_objective": "UNRESOLVED — must be selected before implementation",
                "positive_pair_definition": "UNRESOLVED",
                "negative_pairs_or_collapse_prevention": "UNRESOLVED",
                "identity_labels_used_for_training": False,
                "identity_labels_allowed_for_evaluation_only": True
            },
            "experiment_control": {
                "random_seed": args.seed,
                "train_subjects": 64,
                "validation_subjects": 10,
                "test_subjects": 50,
                "test_set_used_for_model_selection": False,
                "training_started": False
            },
            "next_required_decisions": [
                "Choose and justify one self-supervised pretext objective (not several at once by default).",
                "Specify the treatment of sequences shorter than the frame budget without silent exclusion.",
                "Specify the temporal encoder and embedding/aggregation design.",
                "Define covariate measurements and evaluation protocol before test evaluation."
            ]
        }
        config_path = output_dir / "stage9_experiment_config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        summary = {
            "stage": 9,
            "status": "PASS",
            "manifest_rows": int(len(df)),
            "subjects_by_split": split_subjects,
            "candidate_frame_budget": args.frame_budget,
            "eligible_sequences": eligible,
            "short_sequences_requiring_policy": short,
            "configuration_created": str(config_path),
            "training_started": False,
            "image_pixels_loaded": False,
            "dataset_modified": False
        }
        report_path = output_dir / "stage9_setup_report.json"
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(f"PASS: manifest rows = {len(df):,}")
        logger.info(f"PASS: fixed subject split = {split_subjects}")
        logger.info(f"Candidate {args.frame_budget}-frame eligible sequences = {eligible:,}")
        logger.info(f"Sequences needing short-sequence policy = {short:,}")
        logger.info(f"Config saved: {config_path}")
        logger.info(f"Report saved: {report_path}")
        logger.info("Status: PASS — scaffold only; no training or image loading.")
        return 0
    except Exception as exc:
        logger.info("\nSTAGE 9 FAILED")
        logger.info(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
