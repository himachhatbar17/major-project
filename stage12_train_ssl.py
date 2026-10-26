#!/usr/bin/env python
"""
Stage 12 — Controlled self-supervised gait comparison (training is opt-in).

This is the first script with a real training loop. It supports:
  --objective contrastive
  --objective masked_reconstruction
  --objective both (runs both objectives)
  --sampling-policy repeat|variable|both

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
  python .\\IMPLEMENTATION\\stage12_train_ssl.py --stage5-dir ".\\IMPLEMENTATION\\Stage 5" --dataset-root "D:\\PATH\\TO\\CASIA-B" --output-dir ".\\IMPLEMENTATION\\Stage 12" --objective contrastive

Example: explicit training (only when ready)
  python .\\IMPLEMENTATION\\stage12_train_ssl.py --stage5-dir ".\\IMPLEMENTATION\\Stage 5" --dataset-root "D:\\PATH\\TO\\CASIA-B" --output-dir ".\\IMPLEMENTATION\\Stage 12" --objective contrastive --train --epochs 30 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import subprocess
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
    logger = logging.getLogger("stage12")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(out_dir / "stage12_console_log.txt", encoding="utf-8")
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


def pad_collate(batch):
    """Pad variable-length clips in a batch and preserve their true lengths.

    Videos are T,1,H,W. Padding is zero-valued and is excluded from the
    reconstruction mask/loss and from length-aware temporal pooling.
    Contrastive views are padded to the same batch maximum as their source.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    max_t = max(int(item["video"].shape[0]) for item in batch)

    def pad_tensor(key):
        tensors = [item[key] for item in batch]
        template = tensors[0]
        padded = template.new_zeros((len(batch), max_t, *template.shape[1:]))
        for i, tensor in enumerate(tensors):
            if tensor.ndim != template.ndim or tensor.shape[1:] != template.shape[1:]:
                raise ValueError(f"Inconsistent tensor shapes for {key}")
            padded[i, :tensor.shape[0]] = tensor
        return padded

    result = {
        "video": pad_tensor("video"),
        "length": lengths,
    }
    if "view1" in batch[0] and "view2" in batch[0]:
        result["view1"] = pad_tensor("view1")
        result["view2"] = pad_tensor("view2")
    for key in ("subject_id", "condition", "view", "sequence"):
        result[key] = [item[key] for item in batch]
    return result


class CASIABSequenceDataset(Dataset):
    def __init__(self, manifest_csv: Path, dataset_root: Path, split: str,
                 frames: int = 32, image_size: int = 64, objective: str = "contrastive",
                 sampling_policy: str = "repeat"):
        self.root = dataset_root
        self.frames = frames
        self.image_size = image_size
        self.objective = objective
        self.sampling_policy = sampling_policy
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
        if self.sampling_policy == "repeat" or len(paths) >= self.frames:
            idx = uniform_indices(len(paths), self.frames)
        elif self.sampling_policy == "variable":
            # Preserve every available frame for short clips; long clips remain fixed at `frames`.
            idx = np.arange(len(paths), dtype=np.int64)
        else:
            raise ValueError(f"Unknown sampling policy: {self.sampling_policy}")
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
            "length": int(video.shape[0]),
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

    def forward(self, x, lengths=None):
        f = self.forward_features(x)
        if lengths is None:
            pooled = self.pool(f).flatten(1)
        else:
            # block3 downsamples time by stride 2; valid output length is ceil(input_length / 2).
            valid_t = torch.div(lengths.to(f.device) + 1, 2, rounding_mode="floor")
            valid_t = valid_t.clamp(min=1, max=f.shape[2])
            t_index = torch.arange(f.shape[2], device=f.device)[None, :]
            mask = (t_index < valid_t[:, None]).to(f.dtype)[:, None, :, None, None]
            pooled = (f * mask).sum(dim=(2, 3, 4)) / (
                mask.sum(dim=(2, 3, 4)).clamp_min(1.0) * f.shape[3] * f.shape[4]
            )
        return F.normalize(self.fc(pooled), dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, dim=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, proj_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)



class ContrastiveModel(nn.Module):
    """Encoder plus projection head; downstream diagnostics use the encoder embedding."""
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.encoder = GaitEncoder3D(embedding_dim)
        self.projector = ProjectionHead(embedding_dim)

    def forward(self, x, lengths=None):
        return self.projector(self.encoder(x, lengths))

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

    def forward(self, masked_video, lengths=None):
        feat = self.encoder.forward_features(masked_video)
        logits = self.decoder(feat)
        logits = F.interpolate(logits, size=masked_video.shape[2:], mode="trilinear", align_corners=False)
        if lengths is None:
            pooled = self.encoder.pool(feat).flatten(1)
        else:
            valid_t = torch.div(lengths.to(feat.device) + 1, 2, rounding_mode="floor")
            valid_t = valid_t.clamp(min=1, max=feat.shape[2])
            t_index = torch.arange(feat.shape[2], device=feat.device)[None, :]
            mask_t = (t_index < valid_t[:, None]).to(feat.dtype)[:, None, :, None, None]
            pooled = (feat * mask_t).sum(dim=(2, 3, 4)) / (
                mask_t.sum(dim=(2, 3, 4)).clamp_min(1.0) * feat.shape[3] * feat.shape[4]
            )
        embedding = F.normalize(self.encoder.fc(pooled), dim=1)
        return embedding, logits


def make_masked_input(video: torch.Tensor, mask_fraction: float = 0.25, lengths=None):
    # video: B,1,T,H,W. Hide complete time steps, reconstruct only those steps.
    b, c, t, h, w = video.shape
    if not 0 < mask_fraction < 1:
        raise ValueError("mask_fraction must be between 0 and 1")
    mask = torch.zeros((b, 1, t, 1, 1), dtype=torch.bool, device=video.device)
    nmask = max(1, min(t - 1, int(round(t * mask_fraction))))
    for i in range(b):
        valid_t = int(lengths[i].item()) if lengths is not None else t
        valid_t = max(2, min(valid_t, t))
        nmask_i = max(1, min(valid_t - 1, int(round(valid_t * mask_fraction))))
        chosen = torch.randperm(valid_t, device=video.device)[:nmask_i]
        mask[i, :, chosen, :, :] = True
    masked = video.masked_fill(mask.expand_as(video), 0.0)
    return masked, mask


def masked_bce_loss(logits, target, mask, lengths=None):
    expanded = mask.expand_as(target)
    if lengths is not None:
        valid = torch.arange(target.shape[2], device=target.device)[None, :] < lengths.to(target.device)[:, None]
        valid = valid[:, None, :, None, None].expand_as(target)
        expanded = expanded & valid
    if not expanded.any():
        raise ValueError("No valid masked pixels for reconstruction loss")
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
            lengths = batch["length"].to(device)
            video = batch["video"].to(device, non_blocking=True)  # B,T,1,H,W
            video = video.permute(0, 2, 1, 3, 4).contiguous()  # B,1,T,H,W
            if objective == "contrastive":
                v1 = batch["view1"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4).contiguous()
                v2 = batch["view2"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4).contiguous()
                if training:
                    optimizer.zero_grad(set_to_none=True)
                z1 = model(v1, lengths)
                z2 = model(v2, lengths)
                loss = nt_xent(z1, z2, temperature)
            else:
                masked, mask = make_masked_input(video, mask_fraction, lengths)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                _, logits = model(masked, lengths)
                loss = masked_bce_loss(logits, video, mask, lengths)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")



def _effective_rank(embeddings: np.ndarray) -> float:
    """Effective rank of centered embedding matrix; no identity labels used."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return float("nan")
    x = embeddings.astype(np.float64) - embeddings.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    energy = singular ** 2
    total = energy.sum()
    if total <= 1e-12:
        return 0.0
    probs = energy / total
    probs = probs[probs > 1e-12]
    return float(np.exp(-(probs * np.log(probs)).sum()))


def _rankme(embeddings: np.ndarray) -> float:
    """RankMe-style score based on normalized singular values of the embedding matrix."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return float("nan")
    x = embeddings.astype(np.float64) - embeddings.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    total = singular.sum()
    if total <= 1e-12:
        return 0.0
    probs = singular / total
    probs = probs[probs > 1e-12]
    return float(np.exp(-(probs * np.log(probs)).sum()))


def evaluate_test_split(model, objective, manifest, root, args, device, out_dir, log):
    """Post-training evaluation. Test identities are used only here, never for training/selection."""
    test_ds = CASIABSequenceDataset(
        manifest, root, "test", args.frames, args.image_size, objective, args.sampling_policy
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=False, collate_fn=pad_collate
    )
    model.eval()
    records = []
    vectors = []
    with torch.no_grad():
        for batch in test_loader:
            lengths = batch["length"].to(device)
            video = batch["video"].to(device).permute(0, 2, 1, 3, 4).contiguous()
            encoder = model.encoder
            emb = encoder(video, lengths)
            vectors.append(emb.cpu().numpy())
            for i in range(len(batch["subject_id"])):
                records.append({
                    "subject_id": str(batch["subject_id"][i]),
                    "condition": str(batch["condition"][i]),
                    "view": str(batch["view"][i]),
                    "sequence": str(batch["sequence"][i]),
                })
    if not vectors:
        raise RuntimeError("Test split produced no embeddings")
    E = np.concatenate(vectors, axis=0).astype(np.float64)
    # Unit-normalize again to ensure cosine calculations are stable.
    norms = np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-12)
    E = E / norms
    df = pd.DataFrame(records)
    if len(df) != len(E):
        raise RuntimeError("Embedding/metadata row count mismatch")

    # Label-free diagnostics: only the embedding matrix is used here.
    diagnostics = {
        "num_test_sequences": int(len(E)),
        "embedding_dim": int(E.shape[1]),
        "effective_rank": _effective_rank(E),
        "rankme": _rankme(E),
        "mean_pairwise_cosine_similarity": float(((E @ E.T).sum() - len(E)) / max(len(E) * (len(E) - 1), 1)),
    }

    # Identity-based retrieval is a downstream evaluation only. Self-match is excluded.
    sim = E @ E.T
    np.fill_diagonal(sim, -np.inf)
    ids = df["subject_id"].astype(str).to_numpy()
    top1 = np.argmax(sim, axis=1)
    diagnostics["retrieval_rank1"] = float(np.mean(ids[top1] == ids))
    # Mean reciprocal rank over all gallery items with the same identity, excluding self.
    reciprocal_ranks = []
    for i in range(len(E)):
        order = np.argsort(-sim[i])
        ranks = np.flatnonzero(ids[order] == ids[i])
        reciprocal_ranks.append(1.0 / (int(ranks[0]) + 1) if len(ranks) else 0.0)
    diagnostics["retrieval_mrr"] = float(np.mean(reciprocal_ranks))

    # Covariate drift: average cosine distance between same-subject mean embeddings
    # across distinct views and across distinct conditions. This is descriptive, not a selector.
    joined = df.copy()
    joined["_row"] = np.arange(len(df))
    joined["_embedding"] = list(E)
    view_means = joined.groupby(["subject_id", "view"], dropna=False)["_embedding"].apply(
        lambda rows: np.mean(np.stack(rows.to_list()), axis=0)
    )
    cond_means = joined.groupby(["subject_id", "condition"], dropna=False)["_embedding"].apply(
        lambda rows: np.mean(np.stack(rows.to_list()), axis=0)
    )

    def mean_pair_distance(grouped):
        by_subject = {}
        for (subject, covariate), vec in grouped.items():
            vec = np.asarray(vec, dtype=np.float64)
            vec = vec / max(np.linalg.norm(vec), 1e-12)
            by_subject.setdefault(str(subject), []).append(vec)
        distances = []
        for vectors_for_subject in by_subject.values():
            if len(vectors_for_subject) < 2:
                continue
            for a in range(len(vectors_for_subject)):
                for b in range(a + 1, len(vectors_for_subject)):
                    distances.append(1.0 - float(np.dot(vectors_for_subject[a], vectors_for_subject[b])))
        return float(np.mean(distances)) if distances else float("nan")

    diagnostics["same_subject_view_cosine_drift"] = mean_pair_distance(view_means)
    diagnostics["same_subject_condition_cosine_drift"] = mean_pair_distance(cond_means)
    report_path = out_dir / f"{objective}_{args.sampling_policy}_evaluation.json"
    report_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    df.to_csv(out_dir / f"{objective}_{args.sampling_policy}_test_metadata.csv", index=False)
    np.save(out_dir / f"{objective}_{args.sampling_policy}_test_embeddings.npy", E)
    log.info("POST-TRAIN TEST EVALUATION (test identities used only after training):")
    for key, value in diagnostics.items():
        log.info(f"  {key}: {value}")
    log.info(f"Saved evaluation: {report_path}")
    return diagnostics

def main():
    ap = argparse.ArgumentParser(description="Train a self-supervised CASIA-B gait encoder (opt-in).")
    ap.add_argument("--stage5-dir", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--objective", choices=["contrastive", "masked_reconstruction", "both"], default="contrastive")
    ap.add_argument("--sampling-policy", choices=["repeat", "variable", "both"], default="repeat",
                    help="repeat pads short sequences by repeated indices; variable preserves short clips and masks padded frames.")
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
    ap.add_argument("--skip-test-evaluation", action="store_true",
                    help="After training, skip test-set embedding/retrieval evaluation (not recommended for final runs).")
    args = ap.parse_args()

    # Run the requested controlled matrix by invoking this same script for each cell.
    objectives = ["contrastive", "masked_reconstruction"] if args.objective == "both" else [args.objective]
    policies = ["repeat", "variable"] if args.sampling_policy == "both" else [args.sampling_policy]
    if len(objectives) * len(policies) > 1:
        script = str(Path(__file__).resolve())
        base = [a for a in sys.argv[1:] if a != "--train"]
        # Preserve training opt-in only when explicitly supplied.
        if args.train:
            base.append("--train")
        base = [a for a in base if not a.startswith("--objective") and not a.startswith("--sampling-policy")]
        # Remove option values as well as option tokens safely by rebuilding from parsed args.
        common = [
            "--stage5-dir", args.stage5_dir, "--dataset-root", args.dataset_root,
            "--output-dir", args.output_dir, "--frames", str(args.frames),
            "--image-size", str(args.image_size), "--embedding-dim", str(args.embedding_dim),
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--workers", str(args.workers), "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay), "--temperature", str(args.temperature),
            "--mask-fraction", str(args.mask_fraction), "--seed", str(args.seed),
            "--max-train-batches", str(args.max_train_batches), "--max-val-batches", str(args.max_val_batches)
        ]
        failed = False
        for objective in objectives:
            for policy in policies:
                cmd = [sys.executable, script, *common, "--objective", objective,
                       "--sampling-policy", policy]
                if args.train:
                    cmd.append("--train")
                if args.skip_test_evaluation:
                    cmd.append("--skip-test-evaluation")
                print(f"\\n=== CONDITION: objective={objective}, sampling_policy={policy} ===", flush=True)
                result = subprocess.run(cmd, check=False)
                if result.returncode != 0:
                    failed = True
                    print(f"Condition failed with exit code {result.returncode}", flush=True)
        if not failed and args.train:
            # Descriptive comparison across the requested model/sampling matrix.
            root_out = Path(args.output_dir).resolve()
            rows = []
            for objective in objectives:
                for policy in policies:
                    report = root_out / f"{objective}_{policy}" / f"{objective}_{policy}_evaluation.json"
                    if report.is_file():
                        with report.open("r", encoding="utf-8") as f:
                            row = json.load(f)
                        row["objective"] = objective
                        row["sampling_policy"] = policy
                        rows.append(row)
            if rows:
                comparison = pd.DataFrame(rows)
                comparison.to_csv(root_out / "stage12_model_comparison.csv", index=False)
                correlations = {}
                for metric in ("rankme", "effective_rank", "mean_pairwise_cosine_similarity",
                               "same_subject_view_cosine_drift", "same_subject_condition_cosine_drift"):
                    if metric in comparison and comparison[metric].nunique(dropna=True) > 1:
                        correlations[f"spearman_{metric}_vs_retrieval_rank1"] = float(
                            comparison[[metric, "retrieval_rank1"]].corr(method="spearman").iloc[0, 1]
                        )
                best_rankme = comparison.loc[comparison["rankme"].idxmax()]
                best_retrieval = comparison.loc[comparison["retrieval_rank1"].idxmax()]
                selection_summary = {
                    "warning": "Descriptive only: four configurations are too few to validate a general model-selection claim.",
                    "num_configurations": int(len(comparison)),
                    "best_by_rankme": {"objective": best_rankme["objective"], "sampling_policy": best_rankme["sampling_policy"], "retrieval_rank1": float(best_rankme["retrieval_rank1"])},
                    "best_by_test_retrieval_for_analysis_only": {"objective": best_retrieval["objective"], "sampling_policy": best_retrieval["sampling_policy"], "retrieval_rank1": float(best_retrieval["retrieval_rank1"])},
                    "rankme_selection_regret_vs_best_test_retrieval": float(best_retrieval["retrieval_rank1"] - best_rankme["retrieval_rank1"]),
                    "correlations": correlations,
                    "note": "Test results are for final analysis only; do not use them to tune or select checkpoints/hyperparameters.",
                }
                (root_out / "stage12_selection_analysis.json").write_text(json.dumps(selection_summary, indent=2), encoding="utf-8")
                print(f"Saved model comparison: {root_out / 'stage12_model_comparison.csv'}", flush=True)
                print(f"Saved descriptive selection analysis: {root_out / 'stage12_selection_analysis.json'}", flush=True)
        return 1 if failed else 0

    out_dir = Path(args.output_dir).resolve() / f"{args.objective}_{args.sampling_policy}"
    log = setup_logger(out_dir)
    stage5_dir = Path(args.stage5_dir).resolve()
    root = Path(args.dataset_root).resolve()
    manifest = stage5_dir / "stage5_sequence_manifest.csv"
    log.info("=" * 78)
    log.info("STAGE 12 — CONTROLLED SELF-SUPERVISED GAIT COMPARISON")
    log.info("=" * 78)
    log.info(f"Manifest: {manifest}")
    log.info(f"Dataset root: {root}")
    log.info(f"Objective: {args.objective}; sampling policy: {args.sampling_policy}")
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

        train_ds = CASIABSequenceDataset(manifest, root, "train", args.frames, args.image_size,
                                         args.objective, args.sampling_policy)
        val_ds = CASIABSequenceDataset(manifest, root, "val", args.frames, args.image_size,
                                       args.objective, args.sampling_policy)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=False,
                                  drop_last=(args.objective == "contrastive"), collate_fn=pad_collate)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=False,
                                drop_last=(args.objective == "contrastive"), collate_fn=pad_collate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {device}; train sequences: {len(train_ds):,}; val sequences: {len(val_ds):,}")

        if args.objective == "contrastive":
            model = ContrastiveModel(args.embedding_dim).to(device)
        else:
            model = MaskedReconstructionModel(args.embedding_dim).to(device)

        # Without --train, only verify that one real batch can be loaded and forwarded.
        if not args.train:
            batch = next(iter(train_loader))
            lengths = batch["length"].to(device)
            x = batch["video"].to(device).permute(0, 2, 1, 3, 4).contiguous()
            with torch.no_grad():
                if args.objective == "contrastive":
                    x1 = batch["view1"].to(device).permute(0, 2, 1, 3, 4).contiguous()
                    x2 = batch["view2"].to(device).permute(0, 2, 1, 3, 4).contiguous()
                    z1, z2 = model(x1, lengths), model(x2, lengths)
                    if z1.shape[0] < 2:
                        raise ValueError("Need batch size >= 2 for contrastive check")
                    loss_value = float(nt_xent(z1, z2, args.temperature).cpu())
                    output_shape = list(z1.shape)
                else:
                    masked, mask = make_masked_input(x, args.mask_fraction, lengths)
                    _, logits = model(masked, lengths)
                    loss_value = float(masked_bce_loss(logits, x, mask, lengths).cpu())
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
                }, out_dir / f"best_{args.objective}_{args.sampling_policy}_checkpoint.pt")

        pd.DataFrame(history).to_csv(out_dir / f"{args.objective}_{args.sampling_policy}_loss_history.csv", index=False)
        report = {
            "status": "TRAINING_COMPLETED",
            "objective": args.objective,
            "sampling_policy": args.sampling_policy,
            "epochs_requested": args.epochs,
            "best_validation_loss": best_val,
            "subjects_by_split": split_subjects,
            "test_subjects_loaded": False,
            "training_labels_use_subject_identity": False,
            "args": vars(args),
        }
        (out_dir / f"{args.objective}_{args.sampling_policy}_training_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log.info("Training completed. Test split was not used for checkpoint selection.")
        if not args.skip_test_evaluation:
            checkpoint_path = out_dir / f"best_{args.objective}_{args.sampling_policy}_checkpoint.pt"
            if checkpoint_path.is_file():
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                evaluate_test_split(model, args.objective, manifest, root, args, device, out_dir, log)
            else:
                log.warning("No finite validation checkpoint was saved; skipping test evaluation.")
        return 0
    except Exception as exc:
        log.info("\nSTAGE 11 FAILED")
        log.info(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
