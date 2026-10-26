#!/usr/bin/env python
"""
Stage 10 — Shared temporal gait encoder + self-supervised objectives (implementation scaffold).

This file implements:
  - CASIA-B sequence loading from the Stage 5 manifest
  - deterministic 32-frame temporal sampling (short sequences are sampled with
    repeated indices; this is explicit and logged in the sample metadata)
  - a shared 3D-CNN gait encoder
  - temporal contrastive loss for two views of the same sequence
  - masked-frame reconstruction loss and a small decoder
  - architecture/loss shape smoke test only

IMPORTANT:
  - No training loop is present or launched.
  - No optimizer steps or checkpoints are created.
  - Test subjects are never used to fit anything.
  - Confirm the manifest path column names and sequence directory layout match
    your local CASIA-B data before using the Dataset class.
  - This is a starting implementation scaffold, not a validated research result.

Run from project root:
  python .\\IMPLEMENTATION\\stage10_gait_ssl.py --stage5-dir ".\\IMPLEMENTATION\\Stage 5" --dataset-root "D:\\FINAL YEAR PROJECT\\IMPLEMENTATION\\output" --output-dir ".\\IMPLEMENTATION\\Stage 10" --smoke-test

The smoke test checks model/loss tensor shapes only; it does not read dataset images
unless --data-loader-check is also supplied.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader


REQUIRED = {"subject_id", "view", "condition", "sequence", "split", "frame_count"}


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage10")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(output_dir / "stage10_console_log.txt", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def find_column(df: pd.DataFrame, candidates: list[str], purpose: str, required: bool = True) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if required:
        raise ValueError(
            f"Could not identify {purpose} column. Tried {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def natural_key(path: Path):
    import re
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\\d+)", path.name)]


def uniform_sample_indices(frame_count: int, budget: int = 32) -> np.ndarray:
    """Deterministic uniform indices. Repeats are allowed for sequences shorter than budget."""
    if frame_count < 1 or budget < 1:
        raise ValueError("frame_count and budget must be positive")
    return np.rint(np.linspace(0, frame_count - 1, num=budget)).astype(np.int64)


class CASIABSequenceDataset(Dataset):
    """Loads a fixed number of grayscale silhouette frames per sequence.

    The sequence directory is taken from a path column if present; otherwise the
    directory is reconstructed as dataset_root/subject/condition/view/sequence.
    If your dataset's directory layout differs, update resolve_sequence_dir().
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        dataset_root: str | Path,
        split: str,
        frames_per_sequence: int = 32,
        image_size: int = 64,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.dataset_root = Path(dataset_root)
        self.split = split.lower()
        self.frames_per_sequence = int(frames_per_sequence)
        self.image_size = int(image_size)
        self.df = pd.read_csv(self.manifest_csv)

        required = {"subject_id", "view", "condition", "sequence", "split", "frame_count"}
        missing = sorted(required - {str(c).lower() for c in self.df.columns})
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}; found {list(self.df.columns)}")

        # Normalize column names for robust access.
        self.df.columns = [str(c).lower() for c in self.df.columns]
        self.subject_col = "subject_id"
        self.view_col = "view"
        self.condition_col = "condition"
        self.sequence_col = "sequence"
        self.split_col = "split"
        self.count_col = "frame_count"
        self.path_col = find_column(
            self.df,
            ["sequence_path", "sequence_dir", "sequence_directory", "folder_path", "path"],
            "sequence directory",
            required=False,
        )
        self.df = self.df[self.df[self.split_col].astype(str).str.lower() == self.split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No sequences found for split={self.split!r}")

    def __len__(self) -> int:
        return len(self.df)

    def resolve_sequence_dir(self, row: pd.Series) -> Path:
        if self.path_col is not None and pd.notna(row[self.path_col]):
            p = Path(str(row[self.path_col]))
            if not p.is_absolute():
                p = self.dataset_root / p
            if p.is_dir():
                return p

        # Common CASIA-B hierarchy: root / subject / condition / view / sequence.
        subject = str(row[self.subject_col])
        condition = str(row[self.condition_col])
        view = str(row[self.view_col])
        sequence = str(row[self.sequence_col])
        candidates = [
            self.dataset_root / subject / condition / view / sequence,
            self.dataset_root / subject / sequence / view,
            self.dataset_root / subject / condition / sequence / view,
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"Could not resolve sequence directory for subject={subject}, "
            f"condition={condition}, view={view}, sequence={sequence}. "
            f"Checked path column and: {candidates}"
        )

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        sequence_dir = self.resolve_sequence_dir(row)
        frame_paths = sorted(sequence_dir.glob("*.png"), key=natural_key)
        if not frame_paths:
            frame_paths = sorted(sequence_dir.glob("*.PNG"), key=natural_key)
        if not frame_paths:
            raise FileNotFoundError(f"No PNG frames in {sequence_dir}")

        indices = uniform_sample_indices(len(frame_paths), self.frames_per_sequence)
        frames = []
        for idx in indices:
            with Image.open(frame_paths[int(idx)]) as image:
                image = image.convert("L")
                if image.size != (self.image_size, self.image_size):
                    image = image.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
                arr = np.asarray(image, dtype=np.float32) / 255.0
            frames.append(arr)
        video = torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(0)  # T,1,H,W
        # Keep raw [0,1] silhouette intensity; no binarization.
        return {
            "video": video,  # T,1,H,W
            "subject_id": str(row[self.subject_col]),
            "condition": str(row[self.condition_col]),
            "view": str(row[self.view_col]),
            "sequence": str(row[self.sequence_col]),
            "split": self.split,
            "original_frame_count": len(frame_paths),
            "used_repeated_indices": len(set(indices.tolist())) < len(indices),
        }


class Conv3DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=(1, 2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_ch),
        ) if in_ch != out_ch or stride != (1, 1, 1) else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.block(x) + self.skip(x))


class GaitEncoder3D(nn.Module):
    """Shared encoder. Input B,1,T,H,W; output normalized B,D embeddings."""
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = Conv3DBlock(32, 32, stride=(1, 1, 1))
        self.stage2 = Conv3DBlock(32, 64, stride=(1, 2, 2))
        self.stage3 = Conv3DBlock(64, 128, stride=(2, 2, 2))
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        z = self.projection(x)
        return F.normalize(z, dim=1)


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int = 256, projection_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """SimCLR-style NT-Xent for paired views. Batch size must be >= 2."""
    if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("z1 and z2 must be same-shaped 2D tensors")
    batch_size = z1.shape[0]
    if batch_size < 2:
        raise ValueError("NT-Xent needs batch size >= 2")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0)
    logits = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    targets = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])
    return F.cross_entropy(logits, targets)


class MaskedFrameDecoder(nn.Module):
    """Reconstruct per-frame silhouette masks from temporal encoder features."""
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1),
        )

    def forward(self, feature_map, output_size):
        logits = self.decoder(feature_map)
        return F.interpolate(logits, size=output_size, mode="trilinear", align_corners=False)


class MaskedReconstructionModel(nn.Module):
    """Encoder + reconstruction decoder; loss is computed only on masked frames."""
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.encoder = GaitEncoder3D(embedding_dim=embedding_dim)
        self.feature_decoder = MaskedFrameDecoder()
        # Reconstruct directly from a shared intermediate feature map.
        self.feature_net = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
            Conv3DBlock(32, 32, stride=(1, 1, 1)),
            Conv3DBlock(32, 64, stride=(1, 2, 2)),
            Conv3DBlock(64, 128, stride=(2, 2, 2)),
        )

    def forward(self, x):
        features = self.feature_net(x)
        logits = self.feature_decoder(features, output_size=x.shape[2:])
        embedding = self.encoder(x)
        return embedding, logits


def make_temporal_mask(batch: int, frames: int, mask_fraction: float, device) -> torch.Tensor:
    if not 0 < mask_fraction < 1:
        raise ValueError("mask_fraction must be between 0 and 1")
    mask = torch.zeros((batch, 1, frames, 1, 1), dtype=torch.bool, device=device)
    count = max(1, min(frames - 1, int(round(frames * mask_fraction))))
    for b in range(batch):
        idx = torch.randperm(frames, device=device)[:count]
        mask[b, :, idx, :, :] = True
    return mask


def masked_reconstruction_loss(logits: torch.Tensor, target: torch.Tensor, temporal_mask: torch.Tensor):
    """BCE-with-logits evaluated only on masked time steps."""
    if logits.shape != target.shape:
        raise ValueError(f"logits shape {logits.shape} != target shape {target.shape}")
    expanded = temporal_mask.expand_as(target)
    if not expanded.any():
        raise ValueError("Temporal mask selected no reconstruction targets")
    per_element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return per_element[expanded].mean()


def run_smoke_test(logger: logging.Logger, image_size: int, frames: int, embedding_dim: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Smoke-test device: {device}")
    batch = 2
    x = torch.rand(batch, 1, frames, image_size, image_size, device=device)

    encoder = GaitEncoder3D(embedding_dim=embedding_dim).to(device)
    z = encoder(x)
    head = ContrastiveProjectionHead(embedding_dim=embedding_dim).to(device)
    p1 = head(z)
    p2 = head(encoder(x + torch.randn_like(x) * 0.01))
    contrastive = nt_xent_loss(p1, p2)

    recon_model = MaskedReconstructionModel(embedding_dim=embedding_dim).to(device)
    _, logits = recon_model(x)
    mask = make_temporal_mask(batch, frames, mask_fraction=0.25, device=device)
    recon = masked_reconstruction_loss(logits, x, mask)

    logger.info(f"PASS: encoder embedding shape = {tuple(z.shape)}")
    logger.info(f"PASS: contrastive loss is finite = {bool(torch.isfinite(contrastive).item())}")
    logger.info(f"PASS: reconstruction logits shape = {tuple(logits.shape)}")
    logger.info(f"PASS: masked reconstruction loss is finite = {bool(torch.isfinite(recon).item())}")
    return {
        "device": str(device),
        "embedding_shape": list(z.shape),
        "contrastive_loss_finite": bool(torch.isfinite(contrastive).item()),
        "reconstruction_shape": list(logits.shape),
        "reconstruction_loss_finite": bool(torch.isfinite(recon).item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Implement shared gait encoder and SSL objectives; no training.")
    ap.add_argument("--stage5-dir", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--smoke-test", action="store_true", help="Run dummy-tensor shape/loss checks only.")
    ap.add_argument("--data-loader-check", action="store_true",
                    help="Load one real training sequence to verify local paths and image decoding.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    logger = setup_logger(out_dir)
    stage5_dir = Path(args.stage5_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = stage5_dir / "stage5_sequence_manifest.csv"

    logger.info("=" * 78)
    logger.info("STAGE 10 — SHARED GAIT ENCODER & SELF-SUPERVISED OBJECTIVES")
    logger.info("=" * 78)
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Frames candidate: {args.frames} | Image size: {args.image_size} | Embedding dim: {args.embedding_dim}")
    logger.info("TRAINING: DISABLED — no optimizer steps will run.")

    try:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Stage 5 manifest not found: {manifest_path}")
        # The model/loss smoke test uses synthetic tensors and does not need the
        # real dataset directory. Require the dataset root only for a real loader check.
        if args.data_loader_check and not dataset_root.is_dir():
            raise NotADirectoryError(f"Dataset root not found: {dataset_root}")
        if args.frames < 2 or args.image_size < 1 or args.embedding_dim < 1:
            raise ValueError("frames must be >=2; image-size and embedding-dim must be positive")

        df = pd.read_csv(manifest_path)
        if not REQUIRED.issubset({str(c).lower() for c in df.columns}):
            raise ValueError(f"Manifest columns not sufficient. Found: {list(df.columns)}")
        df.columns = [str(c).lower() for c in df.columns]
        split_counts = {
            s: int(df.loc[df["split"].astype(str).str.lower() == s, "subject_id"].nunique())
            for s in ("train", "val", "test")
        }
        if split_counts != {"train": 64, "val": 10, "test": 50}:
            raise ValueError(f"Unexpected subject split: {split_counts}")
        logger.info(f"PASS: manifest rows = {len(df):,}; fixed subject split = {split_counts}")

        results = {
            "stage": 10,
            "status": "IMPLEMENTATION_SCAFFOLD",
            "manifest_rows": int(len(df)),
            "subjects_by_split": split_counts,
            "candidate_frames": args.frames,
            "image_size": args.image_size,
            "embedding_dim": args.embedding_dim,
            "objectives_implemented": [
                "paired-view NT-Xent temporal contrastive objective",
                "masked temporal-frame silhouette reconstruction objective"
            ],
            "training_performed": False,
            "optimizer_steps": 0,
            "dataset_modified": False,
            "warning": (
                "Contrastive positive-pair augmentations and the final masked-input protocol "
                "must be specified and scientifically justified before training. This scaffold "
                "only defines model components and losses."
            )
        }

        if args.smoke_test:
            results["smoke_test"] = run_smoke_test(
                logger, args.image_size, args.frames, args.embedding_dim
            )

        if args.data_loader_check:
            dataset = CASIABSequenceDataset(
                manifest_csv=manifest_path,
                dataset_root=dataset_root,
                split="train",
                frames_per_sequence=args.frames,
                image_size=args.image_size,
            )
            sample = dataset[0]
            results["data_loader_check"] = {
                "status": "PASS",
                "tensor_shape_TCHW": list(sample["video"].shape),
                "original_frame_count": int(sample["original_frame_count"]),
                "used_repeated_indices": bool(sample["used_repeated_indices"]),
                "condition": sample["condition"],
                "view": sample["view"],
            }
            logger.info(f"PASS: one real sequence loaded; tensor shape = {tuple(sample['video'].shape)}")
            logger.info(f"Original frames = {sample['original_frame_count']}; repeated indices = {sample['used_repeated_indices']}")

        report_path = out_dir / "stage10_implementation_report.json"
        report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info(f"Report saved: {report_path}")
        logger.info("Status: PASS — architecture/objective scaffold only.")
        logger.info("No training performed; no optimizer steps; dataset unchanged.")
        return 0
    except Exception as exc:
        logger.info("\nSTAGE 10 FAILED")
        logger.info(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
