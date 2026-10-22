#!/usr/bin/env python3
"""
Stage 1 — CASIA-B dataset audit and subject-disjoint split preparation.

What it does:
  1. Scans a CASIA-B root directory without modifying the dataset.
  2. Builds a per-sequence manifest with subject, view, condition, sequence,
     frame count, extensions, and paths.
  3. Prints a detailed audit report to the terminal.
  4. Saves the report, manifest, subject list, split assignment, and config in
     a folder named "Stage 1".
  5. Creates deterministic subject-disjoint train/validation/test splits
     (64/10/50 by default for 124 subjects), only when feasible.

Expected common layout:
  CASIA-B/
    001/
      bg-01/
        000/
          000001.png
        018/
          ...
      cl-01/
        000/
          ...
      nm-01/
        000/
          ...
    002/
      ...

If your dataset has an extra wrapper folder, set --data-root to the directory
whose immediate children are the subject folders (usually 001 ... 124).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}
DEFAULT_SPLIT = {"train": 64, "val": 10, "test": 50}


class Tee:
    """Write console output to both terminal and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def natural_key(value: str):
    import re
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value)]


def list_subject_dirs(data_root: Path) -> list[Path]:
    """Find immediate numeric subject folders, e.g. 001 ... 124."""
    dirs = [p for p in data_root.iterdir() if p.is_dir()]
    numeric = [p for p in dirs if p.name.isdigit()]
    return sorted(numeric, key=lambda p: natural_key(p.name))


def identify_view_and_sequence(sequence_dir: Path, subject_dir: Path):
    """
    Infer labels from the directory path relative to a subject.

    Supports both common layouts:
      subject/view/condition/frames
      subject/condition/view/frames

    In this dataset, frame files appear directly inside paths such as
    subject/bg-01/000/, so the condition and view must be read from the
    parent path components rather than only from the leaf directory name.
    """
    rel_parts = sequence_dir.relative_to(subject_dir).parts
    view = "UNKNOWN"
    condition = "UNKNOWN"
    sequence = sequence_dir.name

    # The view is the numeric path component matching a standard CASIA-B view.
    expected_views = {"000", "018", "036", "054", "072", "090",
                      "108", "126", "144", "162", "180"}
    numeric_parts = [part for part in rel_parts if part.isdigit()]
    matching_views = [part for part in numeric_parts if part in expected_views]
    if matching_views:
        view = matching_views[-1]
    elif numeric_parts:
        view = numeric_parts[-1]

    # Find condition label anywhere in the relative path, e.g. bg-01, cl-02, nm-03.
    for part in rel_parts:
        label = part.lower()
        if label.startswith("nm"):
            condition = "normal"
            sequence = part
            break
        elif label.startswith("bg"):
            condition = "bag"
            sequence = part
            break
        elif label.startswith("cl"):
            condition = "coat"
            sequence = part
            break

    # If the condition was not found in the path, retain the leaf as sequence
    # so the caller can apply the usual classifier as a fallback.
    return view, condition, sequence


def find_sequence_dirs(subject_dir: Path):
    """
    Locate directories that directly contain image files. In standard CASIA-B,
    these are the sequence folders. This is intentionally based on image files,
    not hard-coded sequence names.
    """
    sequence_dirs = []
    try:
        candidates = [subject_dir] + [p for p in subject_dir.rglob("*") if p.is_dir()]
        for folder in candidates:
            try:
                image_files = [
                    p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ]
            except (PermissionError, OSError):
                continue
            if image_files:
                sequence_dirs.append((folder, image_files))
    except (PermissionError, OSError):
        pass

    # Remove a parent directory if it contains images and also has child
    # directories containing images; standard sequence folders remain.
    seq_set = {folder for folder, _ in sequence_dirs}
    filtered = []
    for folder, images in sequence_dirs:
        child_has_images = any(
            child in seq_set for child in folder.iterdir() if child.is_dir()
        )
        if not child_has_images:
            filtered.append((folder, images))
    return sorted(filtered, key=lambda x: natural_key(str(x[0])))


def classify_condition(sequence_name: str) -> str:
    name = sequence_name.lower()
    if name.startswith("nm"):
        return "normal"
    if name.startswith("bg"):
        return "bag"
    if name.startswith("cl"):
        return "coat"
    return "UNKNOWN"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_subjects(subject_ids: list[str], sizes: dict[str, int], seed: int):
    total_requested = sum(sizes.values())
    if len(subject_ids) != total_requested:
        return None, (
            f"Requested split sizes total {total_requested}, but found "
            f"{len(subject_ids)} subjects. No split was silently resized. "
            "Use --train-n --val-n --test-n to set sizes explicitly."
        )
    shuffled = sorted(subject_ids, key=natural_key)
    random.Random(seed).shuffle(shuffled)
    train_end = sizes["train"]
    val_end = train_end + sizes["val"]
    assignment = {}
    for sid in shuffled[:train_end]:
        assignment[sid] = "train"
    for sid in shuffled[train_end:val_end]:
        assignment[sid] = "val"
    for sid in shuffled[val_end:]:
        assignment[sid] = "test"
    return assignment, None


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 CASIA-B audit, manifest, and subject-disjoint split."
    )
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="Path to the directory whose immediate child folders are CASIA-B subjects."
    )
    parser.add_argument(
        "--output-dir", type=str, default="Stage 1",
        help='Output directory. Default: "./Stage 1".'
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-n", type=int, default=64)
    parser.add_argument("--val-n", type=int, default=10)
    parser.add_argument("--test-n", type=int, default=50)
    parser.add_argument(
        "--expected-subjects", type=int, default=124,
        help="Expected CASIA-B subject count for audit warning; use 0 to disable."
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "stage1_console_log.txt"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = log_path.open("w", encoding="utf-8")
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    status = "completed_with_warnings"
    try:
        print("=" * 88)
        print("STAGE 1 — CASIA-B DATASET AUDIT AND SPLIT PREPARATION")
        print("=" * 88)
        print(f"Started at          : {started}")
        print(f"Dataset root        : {data_root}")
        print(f"Output directory    : {output_dir}")
        print(f"Random seed         : {args.seed}")
        print(f"Requested split     : train={args.train_n}, val={args.val_n}, test={args.test_n}")
        print()

        if not data_root.exists():
            raise FileNotFoundError(
                f"Dataset root does not exist: {data_root}\n"
                "Set --data-root to the folder containing subject directories (e.g. 001, 002, ...)."
            )
        if not data_root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {data_root}")

        subject_dirs = list_subject_dirs(data_root)
        subject_ids = [p.name for p in subject_dirs]
        print(f"[1/7] Subject folders detected: {len(subject_dirs)}")
        print("Subject IDs:", ", ".join(subject_ids) if subject_ids else "(none)")
        if not subject_dirs:
            print("ERROR: No immediate numeric subject folders were found.")
            print("The --data-root path may point one level too high or too low.")
            print("Top-level entries:")
            for p in sorted(data_root.iterdir(), key=lambda x: natural_key(x.name))[:80]:
                print(f"  {'DIR ' if p.is_dir() else 'FILE'} {p.name}")
            raise RuntimeError("Could not identify CASIA-B subject folders.")

        if args.expected_subjects and len(subject_dirs) != args.expected_subjects:
            print(
                f"WARNING: Expected about {args.expected_subjects} subjects, "
                f"but found {len(subject_dirs)}. Verify the root and dataset version."
            )

        manifest = []
        subject_summaries = []
        all_extensions = Counter()
        condition_counts = Counter()
        view_counts = Counter()
        sequence_name_counts = Counter()
        unknown_conditions = []
        empty_subjects = []
        unreadable_paths = []
        total_frames = 0

        print("\n[2/7] Scanning subject/view/sequence folders...")
        for subject_dir in subject_dirs:
            seqs = find_sequence_dirs(subject_dir)
            if not seqs:
                empty_subjects.append(subject_dir.name)
            subject_frame_count = 0
            subject_seq_count = 0
            for sequence_dir, image_files in seqs:
                view, inferred_condition, sequence_name = identify_view_and_sequence(
                    sequence_dir, subject_dir
                )
                condition = classify_condition(sequence_name)
                if condition == "UNKNOWN":
                    condition = inferred_condition
                image_files = sorted(image_files, key=lambda p: natural_key(p.name))
                ext_counts = Counter(p.suffix.lower() for p in image_files)
                frame_count = len(image_files)
                total_frames += frame_count
                subject_frame_count += frame_count
                subject_seq_count += 1
                condition_counts[condition] += 1
                view_counts[view] += 1
                sequence_name_counts[sequence_name] += 1
                all_extensions.update(ext_counts)
                if condition == "UNKNOWN":
                    unknown_conditions.append(str(sequence_dir))
                manifest.append({
                    "subject_id": subject_dir.name,
                    "view": view,
                    "condition": condition,
                    "sequence": sequence_name,
                    "sequence_path": str(sequence_dir.resolve()),
                    "frame_count": frame_count,
                    "extensions": ";".join(
                        f"{ext}:{count}" for ext, count in sorted(ext_counts.items())
                    ),
                    "first_frame": str(image_files[0].resolve()) if image_files else "",
                    "last_frame": str(image_files[-1].resolve()) if image_files else "",
                })
            subject_summaries.append({
                "subject_id": subject_dir.name,
                "sequence_count": subject_seq_count,
                "frame_count": subject_frame_count,
                "subject_path": str(subject_dir.resolve()),
            })

        manifest_fields = [
            "subject_id", "view", "condition", "sequence", "sequence_path",
            "frame_count", "extensions", "first_frame", "last_frame"
        ]
        manifest_path = output_dir / "casia_b_sequence_manifest.csv"
        write_csv(manifest_path, manifest, manifest_fields)
        write_csv(
            output_dir / "casia_b_subject_summary.csv",
            subject_summaries,
            ["subject_id", "sequence_count", "frame_count", "subject_path"]
        )

        print(f"Subjects scanned     : {len(subject_dirs)}")
        print(f"Sequence folders     : {len(manifest)}")
        print(f"Total image frames   : {total_frames:,}")
        print(f"Image extensions     : {dict(all_extensions)}")
        print(f"Sequence counts by condition: {dict(condition_counts)}")
        print(f"Sequence counts by view     : {dict(view_counts)}")
        print(f"Sequence-name counts         : {dict(sequence_name_counts)}")
        print(f"Manifest saved              : {manifest_path}")
        print(f"Subject summary saved       : {output_dir / 'casia_b_subject_summary.csv'}")

        print("\n[3/7] Coverage and data-quality checks...")
        expected_views = {"000", "018", "036", "054", "072", "090",
                          "108", "126", "144", "162", "180"}
        detected_views = {row["view"] for row in manifest if row["view"] != "UNKNOWN"}
        print(f"Detected views ({len(detected_views)}): {', '.join(sorted(detected_views, key=natural_key)) or '(none)'}")
        missing_expected_views = sorted(expected_views - detected_views, key=natural_key)
        if missing_expected_views:
            print("WARNING: Common CASIA-B views not detected:", ", ".join(missing_expected_views))
        print("Condition sequence counts:")
        for cond in ["normal", "bag", "coat", "UNKNOWN"]:
            print(f"  {cond:8s}: {condition_counts.get(cond, 0)}")
        if unknown_conditions:
            print(f"WARNING: {len(unknown_conditions)} sequence folders have unknown condition names.")
            for p in unknown_conditions[:30]:
                print("  UNKNOWN:", p)
        if empty_subjects:
            print(f"WARNING: {len(empty_subjects)} subject folders contain no detected sequence frames:")
            print("  " + ", ".join(empty_subjects))
        if not manifest:
            print("ERROR: No sequence folders containing image files were detected.")
            raise RuntimeError("Manifest is empty. Check the data-root layout and image extensions.")

        # Check per-subject coverage without assuming every release has identical files.
        per_subject = defaultdict(list)
        for row in manifest:
            per_subject[row["subject_id"]].append(row)
        for sid in subject_ids:
            rows = per_subject.get(sid, [])
            conds = {r["condition"] for r in rows}
            if not {"normal", "bag", "coat"}.issubset(conds):
                print(
                    f"WARNING: Subject {sid} has conditions {sorted(conds)}; "
                    "one or more of normal/bag/coat is absent."
                )

        # Summarize expected sequence labels without claiming every release is identical.
        expected_seq_prefixes = {"nm": "normal", "bg": "bag", "cl": "coat"}
        observed_seq_names = {row["sequence"].lower() for row in manifest}
        for prefix, label in expected_seq_prefixes.items():
            count = sum(1 for name in observed_seq_names if name.startswith(prefix))
            print(f"Distinct {label} sequence labels (prefix {prefix}-): {count}")

        print("\n[4/7] Creating deterministic subject-disjoint split...")
        sizes = {"train": args.train_n, "val": args.val_n, "test": args.test_n}
        assignment, split_warning = split_subjects(subject_ids, sizes, args.seed)
        split_path = output_dir / "casia_b_subject_split.csv"
        if assignment is None:
            print("WARNING:", split_warning)
            print("No split file with guessed sizes will be created.")
            split_rows = []
        else:
            split_rows = [
                {"subject_id": sid, "split": assignment[sid], "seed": args.seed}
                for sid in sorted(subject_ids, key=natural_key)
            ]
            write_csv(split_path, split_rows, ["subject_id", "split", "seed"])
            for split_name in ["train", "val", "test"]:
                ids = [sid for sid in subject_ids if assignment[sid] == split_name]
                print(f"{split_name.upper():5s} ({len(ids):3d}): {', '.join(ids)}")
            print(f"Split assignment saved: {split_path}")

            # Add split column to a separate manifest; preserve the raw manifest.
            split_manifest = []
            for row in manifest:
                updated = dict(row)
                updated["split"] = assignment[row["subject_id"]]
                split_manifest.append(updated)
            write_csv(
                output_dir / "casia_b_sequence_manifest_with_split.csv",
                split_manifest,
                manifest_fields + ["split"]
            )
            sets = {
                name: {sid for sid, sp in assignment.items() if sp == name}
                for name in ["train", "val", "test"]
            }
            assert sets["train"].isdisjoint(sets["val"])
            assert sets["train"].isdisjoint(sets["test"])
            assert sets["val"].isdisjoint(sets["test"])
            print("Leakage check: PASS — no subject appears in more than one split.")

        print("\n[5/7] Saving dataset configuration and audit summary...")
        config = {
            "stage": "Stage 1 — CASIA-B dataset audit and split preparation",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset_root": str(data_root),
            "output_dir": str(output_dir),
            "random_seed": args.seed,
            "expected_subjects": args.expected_subjects,
            "detected_subject_count": len(subject_dirs),
            "requested_split_sizes": sizes,
            "split_created": assignment is not None,
            "image_extensions": sorted(IMAGE_EXTENSIONS),
            "notes": [
                "Dataset files were read only; they were not moved, renamed, or modified.",
                "Split labels are subject-disjoint and deterministic for a fixed subject list and seed.",
                "Confirm the dataset layout and sequence coverage before training.",
                "This stage does not train a model or use a GPU."
            ]
        }
        with (output_dir / "stage1_config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        audit = {
            "started_at": started,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset_root": str(data_root),
            "output_dir": str(output_dir),
            "subject_count": len(subject_dirs),
            "sequence_folder_count": len(manifest),
            "total_image_frames": total_frames,
            "extensions": dict(all_extensions),
            "sequence_counts_by_condition": dict(condition_counts),
            "sequence_counts_by_view": dict(view_counts),
            "distinct_sequence_names": dict(sequence_name_counts),
            "empty_subjects": empty_subjects,
            "unknown_condition_paths": unknown_conditions,
            "detected_views": sorted(detected_views, key=natural_key),
            "missing_common_views": missing_expected_views,
            "split_created": assignment is not None,
            "split_warning": split_warning,
            "warnings": (
                (["Detected subject count differs from expected count."] if args.expected_subjects and len(subject_dirs) != args.expected_subjects else [])
                + (["Some subjects have no detected sequence frames."] if empty_subjects else [])
                + (["Some sequence conditions are unknown."] if unknown_conditions else [])
                + (["Some common views were not detected."] if missing_expected_views else [])
            )
        }
        with (output_dir / "stage1_audit_summary.json").open("w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)

        with (output_dir / "stage1_subject_ids.txt").open("w", encoding="utf-8") as f:
            f.write("\n".join(sorted(subject_ids, key=natural_key)) + "\n")

        print(f"Configuration saved : {output_dir / 'stage1_config.json'}")
        print(f"Audit summary saved : {output_dir / 'stage1_audit_summary.json'}")
        print(f"Subject list saved  : {output_dir / 'stage1_subject_ids.txt'}")

        print("\n[6/7] Final checks...")
        print(f"Manifest exists: {(output_dir / 'casia_b_sequence_manifest.csv').exists()}")
        print(f"Subject summary exists: {(output_dir / 'casia_b_subject_summary.csv').exists()}")
        print(f"Split created: {assignment is not None}")
        print(f"Output files currently present:")
        for p in sorted(output_dir.iterdir(), key=lambda p: p.name.lower()):
            if p.is_file():
                print(f"  {p.name:44s} {p.stat().st_size:>12,} bytes")

        status = "completed" if assignment is not None and not audit["warnings"] else "completed_with_warnings"
        print("\n[7/7] STAGE 1 FINISHED")
        print(f"Status: {status}")
        print("Review the warnings and manifest before moving to Stage 2.")
        print("=" * 88)

    except Exception as exc:
        status = "failed"
        print("\nSTAGE 1 FAILED")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        try:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
