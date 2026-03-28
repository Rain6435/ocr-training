"""
Data preparation script.

Parses IAM, NIST SD19, and EMNIST datasets into a unified CSV manifest
with columns: image_path, transcription, difficulty, split, source
"""

import os
import csv
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_iam_words(iam_dir: str) -> list[dict]:
    """
    Parse IAM words.txt to extract (image_path, transcription) pairs.

    Format: wordID ok/err graylevel #components x y w h grammatical_tag transcription
    Example: a01-000u-00-00 ok 154 1 408 768 27 51 AT A
    """
    words_file = Path(iam_dir) / "ascii" / "words.txt"
    words_dir = Path(iam_dir) / "words"
    entries = []

    if not words_file.exists():
        print(f"Warning: {words_file} not found")
        return entries

    with open(words_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue

            word_id = parts[0]
            status = parts[1]
            transcription = parts[-1]

            # Build image path: a01-000u-00-00 -> a01/a01-000u/a01-000u-00-00.png
            parts_id = word_id.split("-")
            folder1 = parts_id[0]
            folder2 = f"{parts_id[0]}-{parts_id[1]}"
            img_path = words_dir / folder1 / folder2 / f"{word_id}.png"

            if img_path.exists() and status == "ok":
                entries.append({
                    "image_path": str(img_path),
                    "transcription": transcription,
                    "source": "iam",
                    "word_id": word_id,
                })

    return entries


def parse_iam_lines(iam_dir: str) -> list[dict]:
    """Parse IAM lines.txt for line-level data."""
    lines_file = Path(iam_dir) / "ascii" / "lines.txt"
    lines_dir = Path(iam_dir) / "lines"
    entries = []

    if not lines_file.exists():
        print(f"Warning: {lines_file} not found")
        return entries

    with open(lines_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue

            line_id = parts[0]
            status = parts[1]
            transcription = " ".join(parts[8:]).replace("|", " ")

            parts_id = line_id.split("-")
            folder1 = parts_id[0]
            folder2 = f"{parts_id[0]}-{parts_id[1]}"
            img_path = lines_dir / folder1 / folder2 / f"{line_id}.png"

            if img_path.exists() and status == "ok":
                entries.append({
                    "image_path": str(img_path),
                    "transcription": transcription,
                    "source": "iam",
                    "line_id": line_id,
                })

    return entries


def load_iam_splits(iam_dir: str) -> dict[str, set]:
    """Load official IAM writer-independent train/val/test splits."""
    splits = {"train": set(), "val": set(), "test": set()}
    split_files = {
        "train": "trainset.txt",
        "val": "validationset1.txt",
        "test": "testset.txt",
    }

    for split_name, filename in split_files.items():
        path = Path(iam_dir) / "ascii" / filename
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        splits[split_name].add(line)

    return splits


def assign_difficulty(entry: dict) -> str:
    """Assign difficulty based on source and characteristics."""
    source = entry.get("source", "")
    transcription = entry.get("transcription", "")

    if source == "emnist":
        return "easy"
    elif source == "iam" and len(transcription) <= 5:
        return "easy"
    elif source == "iam" and len(transcription) > 20:
        return "hard"
    else:
        return "medium"


def create_manifests(
    output_dir: str = "data/processed",
    iam_dir: str = "data/raw/iam",
):
    """Create train.csv, val.csv, test.csv manifests."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all entries
    all_entries = []

    # IAM words
    iam_words = parse_iam_words(iam_dir)
    print(f"Parsed {len(iam_words)} IAM word entries")
    all_entries.extend(iam_words)

    # IAM lines
    iam_lines = parse_iam_lines(iam_dir)
    print(f"Parsed {len(iam_lines)} IAM line entries")
    all_entries.extend(iam_lines)

    # Load splits
    splits = load_iam_splits(iam_dir)

    # Assign split and difficulty
    for entry in all_entries:
        entry["difficulty"] = assign_difficulty(entry)

        # Determine split from IAM official splits
        entry_id = entry.get("word_id", entry.get("line_id", ""))
        # Extract form ID (e.g., a01-000u from a01-000u-00-00)
        form_id = "-".join(entry_id.split("-")[:2]) if entry_id else ""

        if form_id in splits.get("train", set()):
            entry["split"] = "train"
        elif form_id in splits.get("val", set()):
            entry["split"] = "val"
        elif form_id in splits.get("test", set()):
            entry["split"] = "test"
        else:
            # Random assignment if no official split
            r = np.random.random()
            if r < 0.8:
                entry["split"] = "train"
            elif r < 0.9:
                entry["split"] = "val"
            else:
                entry["split"] = "test"

    # Write CSVs
    fieldnames = ["image_path", "transcription", "difficulty", "split", "source"]

    for split_name in ["train", "val", "test"]:
        split_entries = [e for e in all_entries if e["split"] == split_name]
        output_path = os.path.join(output_dir, f"{split_name}.csv")

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(split_entries)

        print(f"{split_name}: {len(split_entries)} entries -> {output_path}")

    print(f"\nTotal entries: {len(all_entries)}")


if __name__ == "__main__":
    create_manifests()
