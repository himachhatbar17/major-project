#!/usr/bin/env python
"""
Stage 11 — Actual self-supervised gait training pipeline (training is opt-in).

This is the first script with a real training loop. It supports:
  --objective contrastive
  --objective masked_reconstruction

Safety / research controls:
  * It never uses subject IDs as training targets.
  * It trains on train subjects only and uses validation subjects only for
    checkpoint selection.
  * Test subjects are not loaded by the training/validation loaders.
  * Training only starts when --train is explicitly supplied.
  * Without --train, it performs a one-batch real-data loader check.
  * It does not modify source images or manifests.

Important:
  * Confirm --dataset-root points to the actual CASIA-B folder.
  * The Stage 5 manifest should contain a usable sequence directory column
    (e.g. sequence_path/sequence_dir) OR the data layout must match one of the
    directory layouts in resolve_sequence_dir().
  * This is a research baseline implementation. Do not compare objective
    results until the augmentation/masking policy and hyperparameters are fixed.

Example: loader check only
  python .\\IMPLEMENTATION\\stage11_train_ssl.py --stage5-dir ".\\IMPLEMENTATION\\Stage 5" --dataset-root "D:\\PATH\\TO\\CASIA-B" --output-dir ".\\IMPLEMENTATION\\Stage 11" --objective contrastive

Example: explicit training (only when ready)
  python .\\IMPLEMENTATION\\stage11_train_ssl.py --stage5-dir ".\\IMPLEMENTATION\\Stage 5" --dataset-root "D:\\PATH\\TO\\CASIA-B" --output-dir ".\\IMPLEMENTATION\\Stage 11" --objective contrastive --train --epochs 30 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
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


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage11")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(out_dir / "stage11_console_log.txt", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def uniform_indices(n: int, budget: int) -> np.ndarray:
    if n < 1:
        raise ValueError("Cannot sample an empty sequence")
    return np.rint(np.linspace(0, n - 1, num=budget)).astype(np.int64)


class CASIABSequenceDataset(Dataset):
    def __init__(self, manifest_csv: Path, dataset_root: Path, split: str,
                 frames: int = 32, image_size: int = 64, objective: str = "contrastive"):
        self.root = dataset_root
        self.frames = frames
        self.image_size = image_size
        self.objective = objective
        df = pd.read_csv(manifest_csv)
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing = sorted(REQUIRED - set(df.columns))
        if missing:
            raise ValueError(f"Manifest missing columns {missing}; found {list(df.columns)}")
        df["split"] = df["split"].astype(str).str.lower()
        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No manifest rows for split={split}")
        self.path_col = next((c for c in [
            "sequence_path", "sequence_dir", "sequence_directory", "folder_path", "path"
        ] if c in self.df.columns), None)

    def __len__(self):
        return len(self.df)

    def resolve_dir(self, row: pd.Series) -> Path:
        if self.path_col and pd.notna(row[self.path_col]):
            p = Path(str(row[self.path_col]))
            candidates = [p] if p.is_absolute() else [self.root / p, p]
            for c in candidates:
                if c.is_dir():
                    return c

        subject = str(row["subject_id"])
        condition = str(row["condition"])
        view = str(row["view"])
        sequence = str(row["sequence"])
        candidates = [
            self.root / subject / condition / view / sequence,
            self.root / subject / condition / sequence / view,
            self.root / subject / sequence / view,
            self.root / subject / sequence / condition / view,
            self.root / subject / sequence,
        ]
        for c in candidates:
            if c.is_dir() and (list(c.glob("*.png")) or list(c.glob("*.PNG"))):
                return c
        raise FileNotFoundError(
            f"Cannot locate frames for subject={subject}, condition={condition}, "
            f"view={view}, sequence={sequence}. Tried: {candidates}. "
            "Check --dataset-root and the Stage 5 path column."
        )

    def load_frames(self, row: pd.Series) -> torch.Tensor:
        seq_dir = self.resolve_dir(row)
        paths = sorted(seq_dir.glob("*.png"), key=natural_key)
        if not paths:
            paths = sorted(seq_dir.glob("*.PNG"), key=natural_key)
        if not paths:
            raise FileNotFoundError(f"No PNG frames in {seq_dir}")
        idx = uniform_indices(len(paths), self.frames)
        arrs = []
        for i in idx:
            with Image.open(paths[int(i)]) as im:
                im = im.convert("L")
                if im.size != (self.image_size, self.image_size):
                    im = im.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
                arrs.append(np.asarray(im, dtype=np.float32) / 255.0)
        # T,1,H,W
        return torch.from_numpy(np.stack(arrs)).unsqueeze(1)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        video = self.load_frames(row)  # T,1,H,W
        item = {
            "video": video,
            "subject_id": str(row["subject_id"]),
            "condition": str(row["condition"]),
            "view": str(row["view"]),
            "sequence": str(row["sequence"]),
        }
        if self.objective == "contrastive":
            item["view1"] = self._augment(video)
            item["view2"] = self._augment(video)
        return item

    @staticmethod
    def _augment(video: torch.Tensor) -> torch.Tensor:
        # Mild, label-preserving intensity and spatial translation augmentation.
        # Same spatial transform is applied to every frame in a clip.
        out = video.clone()
        if random.random() < 0.8:
            gain = random.uniform(0.9, 1.1)
            out = torch.clamp(out * gain, 0.0, 1.0)
        if random.random() < 0.5:
            t, c, h, w = out.shape
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            shifted = torch.zeros_like(out)
            x_src0, x_src1 = max(0, -dx), min(w, w - dx)
            y_src0, y_src1 = max(0, -dy), min(h, h - dy)
            x_dst0, x_dst1 = max(0, dx), min(w, w + dx)
            y_dst0, y_dst1 = max(0, dy), min(h, h + dy)
            shifted[:, :, y_dst0:y_dst1, x_dst0:x_dst1] = out[:, :, y_src0:y_src1, x_src0:x_src1]
            out = shifted
        return out


class Residual3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1, 1)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.skip = nn.Identity() if in_ch == out_ch and stride == (1, 1, 1) else nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm3d(out_ch)
        )

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.skip(x), inplace=True)


class GaitEncoder3D(nn.Module):
    """Input B,1,T,H,W; exposes both feature map and L2-normalized embedding."""
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True)
        )
        self.block1 = Residual3D(32, 32)
        self.block2 = Residual3D(32, 64, stride=(1, 2, 2))
        self.block3 = Residual3D(64, 128, stride=(2, 2, 2))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, embedding_dim)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x):
        f = self.forward_features(x)
        pooled = self.pool(f).flatten(1)
        return F.normalize(self.fc(pooled), dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, dim=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, proj_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if z1.shape != z2.shape or z1.ndim != 2:
        raise ValueError("Contrastive embeddings must be same-shaped 2D tensors")
    b = z1.shape[0]
    if b < 2:
        raise ValueError("Contrastive objective requires batch size >= 2")
    z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0)
    logits = (z @ z.T) / temperature
    logits.fill_diagonal_(-torch.inf)
    targets = torch.cat([torch.arange(b, 2*b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(logits, targets)


class MaskedReconstructionModel(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.encoder = GaitEncoder3D(embedding_dim)
        self.decoder = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, 1)
        )

    def forward(self, masked_video):
        feat = self.encoder.forward_features(masked_video)
        logits = self.decoder(feat)
        logits = F.interpolate(logits, size=masked_video.shape[2:], mode="trilinear", align_corners=False)
        pooled = self.encoder.pool(feat).flatten(1)
        embedding = F.normalize(self.encoder.fc(pooled), dim=1)
        return embedding, logits


def make_masked_input(video: torch.Tensor, mask_fraction: float = 0.25):
    # video: B,1,T,H,W. Hide complete time steps, reconstruct only those steps.
    b, c, t, h, w = video.shape
    if not 0 < mask_fraction < 1:
        raise ValueError("mask_fraction must be between 0 and 1")
    mask = torch.zeros((b, 1, t, 1, 1), dtype=torch.bool, device=video.device)
    nmask = max(1, min(t - 1, int(round(t * mask_fraction))))
    for i in range(b):
        chosen = torch.randperm(t, device=video.device)[:nmask]
        mask[i, :, chosen, :, :] = True
    masked = video.masked_fill(mask.expand_as(video), 0.0)
    return masked, mask


def masked_bce_loss(logits, target, mask):
    expanded = mask.expand_as(target)
    return F.binary_cross_entropy_with_logits(logits[expanded], target[expanded])


def run_epoch(model, loader, objective, device, optimizer=None, temperature=0.2,
              mask_fraction=0.25, max_batches=0):
    training = optimizer is not None
    model.train(training)
    losses = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            video = batch["video"].to(device, non_blocking=True)  # B,T,1,H,W
            video = video.permute(0, 2, 1, 3, 4).contiguous()  # B,1,T,H,W
            if objective == "contrastive":
                v1 = batch["view1"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4).contiguous()
                v2 = batch["view2"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4).contiguous()
                if training:
                    optimizer.zero_grad(set_to_none=True)
                z1 = model(v1)
                z2 = model(v2)
                loss = nt_xent(z1, z2, temperature)
            else:
                masked, mask = make_masked_input(video, mask_fraction)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                _, logits = model(masked)
                loss = masked_bce_loss(logits, video, mask)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Train a self-supervised CASIA-B gait encoder (opt-in).")
    ap.add_argument("--stage5-dir", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--objective", choices=["contrastive", "masked_reconstruction"], default="contrastive")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--mask-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-batches", type=int, default=0,
                    help="Debug limit; 0 means all batches.")
    ap.add_argument("--max-val-batches", type=int, default=0,
                    help="Debug limit; 0 means all batches.")
    ap.add_argument("--train", action="store_true",
                    help="Explicitly enable optimization/training. Without this, only one-batch data check runs.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    log = setup_logger(out_dir)
    stage5_dir = Path(args.stage5_dir).resolve()
    root = Path(args.dataset_root).resolve()
    manifest = stage5_dir / "stage5_sequence_manifest.csv"
    log.info("=" * 78)
    log.info("STAGE 11 — SELF-SUPERVISED GAIT TRAINING PIPELINE")
    log.info("=" * 78)
    log.info(f"Manifest: {manifest}")
    log.info(f"Dataset root: {root}")
    log.info(f"Objective: {args.objective}")
    log.info(f"Frames: {args.frames}; image size: {args.image_size}; embedding dim: {args.embedding_dim}")
    log.info(f"Training explicitly enabled: {args.train}")

    try:
        if not manifest.is_file():
            raise FileNotFoundError(f"Stage 5 manifest not found: {manifest}")
        if not root.is_dir():
            raise NotADirectoryError(
                f"Dataset root not found: {root}. Supply the actual folder containing CASIA-B subjects."
            )
        if args.frames < 2 or args.batch_size < 2 or args.epochs < 1:
            raise ValueError("frames >= 2, batch-size >= 2, and epochs >= 1 are required")
        seed_everything(args.seed)
        meta = pd.read_csv(manifest)
        meta.columns = [str(c).strip().lower() for c in meta.columns]
        missing = sorted(REQUIRED - set(meta.columns))
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")
        split_subjects = {
            s: int(meta.loc[meta["split"].astype(str).str.lower() == s, "subject_id"].nunique())
            for s in ("train", "val", "test")
        }
        if split_subjects != {"train": 64, "val": 10, "test": 50}:
            raise ValueError(f"Fixed split check failed: {split_subjects}")
        log.info(f"PASS: manifest rows = {len(meta):,}; subject split = {split_subjects}")

        train_ds = CASIABSequenceDataset(manifest, root, "train", args.frames, args.image_size, args.objective)
        val_ds = CASIABSequenceDataset(manifest, root, "val", args.frames, args.image_size, args.objective)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                                  drop_last=(args.objective == "contrastive"))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                                drop_last=(args.objective == "contrastive"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {device}; train sequences: {len(train_ds):,}; val sequences: {len(val_ds):,}")

        if args.objective == "contrastive":
            model = nn.Sequential(GaitEncoder3D(args.embedding_dim), ProjectionHead(args.embedding_dim)).to(device)
        else:
            model = MaskedReconstructionModel(args.embedding_dim).to(device)

        # Without --train, only verify that one real batch can be loaded and forwarded.
        if not args.train:
            batch = next(iter(train_loader))
            x = batch["video"].to(device).permute(0, 2, 1, 3, 4).contiguous()
            with torch.no_grad():
                if args.objective == "contrastive":
                    x1 = batch["view1"].to(device).permute(0, 2, 1, 3, 4).contiguous()
                    x2 = batch["view2"].to(device).permute(0, 2, 1, 3, 4).contiguous()
                    z1, z2 = model(x1), model(x2)
                    if z1.shape[0] < 2:
                        raise ValueError("Need batch size >= 2 for contrastive check")
                    loss_value = float(nt_xent(z1, z2, args.temperature).cpu())
                    output_shape = list(z1.shape)
                else:
                    masked, mask = make_masked_input(x, args.mask_fraction)
                    _, logits = model(masked)
                    loss_value = float(masked_bce_loss(logits, x, mask).cpu())
                    output_shape = list(logits.shape)
            log.info(f"PASS: real data batch loaded; output shape = {output_shape}; finite loss = {np.isfinite(loss_value)}")
            log.info("No optimizer steps taken. To train, rerun with --train after reviewing configuration.")
            return 0

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = float("inf")
        history = []
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(
                model, train_loader, args.objective, device, optimizer=optimizer,
                temperature=args.temperature, mask_fraction=args.mask_fraction,
                max_batches=args.max_train_batches
            )
            val_loss = run_epoch(
                model, val_loader, args.objective, device, optimizer=None,
                temperature=args.temperature, mask_fraction=args.mask_fraction,
                max_batches=args.max_val_batches
            )
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            history.append(row)
            log.info(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
            if np.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "objective": args.objective,
                    "args": vars(args),
                    "best_val_loss": best_val,
                }, out_dir / f"best_{args.objective}_checkpoint.pt")

        pd.DataFrame(history).to_csv(out_dir / f"{args.objective}_loss_history.csv", index=False)
        report = {
            "status": "TRAINING_COMPLETED",
            "objective": args.objective,
            "epochs_requested": args.epochs,
            "best_validation_loss": best_val,
            "subjects_by_split": split_subjects,
            "test_subjects_loaded": False,
            "training_labels_use_subject_identity": False,
            "args": vars(args),
        }
        (out_dir / f"{args.objective}_training_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log.info("Training completed. Test split was not loaded or used for selection.")
        return 0
    except Exception as exc:
        log.info("\nSTAGE 11 FAILED")
        log.info(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
