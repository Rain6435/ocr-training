"""
Data preparation script.

Parses IAM, NIST SD19, and EMNIST datasets into a unified CSV manifest
with columns: image_path, transcription, difficulty, split, source
"""

import os
import csv
import random
import argparse
import numpy as np
from pathlib import Path


ManifestEntry = dict[str, str]


def interleave_by_source(entries: list[ManifestEntry], seed: int = 42) -> list[ManifestEntry]:
    """Return a randomized source-interleaved list of manifest entries."""
    if not entries:
        return entries

    rng = random.Random(seed)
    source_buckets: dict[str, list[ManifestEntry]] = {}
    for entry in entries:
        source = str(entry.get("source", "unknown"))
        source_buckets.setdefault(source, []).append(entry)

    for bucket in source_buckets.values():
        rng.shuffle(bucket)

    mixed: list[ManifestEntry] = []
    active_sources: list[str] = [s for s, bucket in source_buckets.items() if bucket]
    while active_sources:
        rng.shuffle(active_sources)
        next_active_sources: list[str] = []
        for source in active_sources:
            bucket = source_buckets[source]
            if bucket:
                mixed.append(bucket.pop())
            if bucket:
                next_active_sources.append(source)
        active_sources = next_active_sources

    return mixed


def rebalance_train_entries(
    entries: list[ManifestEntry],
    max_nist_to_iam_ratio: float = 1.0,
    seed: int = 42,
) -> list[ManifestEntry]:
    """
    Rebalance training entries so NIST does not overwhelm IAM sequence learning.

    Keeps all IAM samples and caps NIST count to:
        floor(len(iam) * max_nist_to_iam_ratio)
    """
    if max_nist_to_iam_ratio <= 0:
        return entries

    iam_entries = [e for e in entries if e.get("source") == "iam"]
    nist_entries = [e for e in entries if e.get("source") == "nist_sd19"]
    other_entries = [e for e in entries if e.get("source") not in {"iam", "nist_sd19"}]

    if not iam_entries or not nist_entries:
        return entries

    rng = random.Random(seed)
    rng.shuffle(nist_entries)

    nist_limit = int(len(iam_entries) * max_nist_to_iam_ratio)
    nist_kept = nist_entries[:max(1, nist_limit)]

    return iam_entries + nist_kept + other_entries


def decode_nist_class_label(class_hex: str) -> str | None:
    """Decode NIST SD19 class folder (hex ASCII code) to a single character."""
    try:
        value = int(class_hex.lower(), 16)
    except ValueError:
        return None

    # NIST SD19 by_class folders encode the target character as ASCII hex.
    if 32 <= value <= 126:
        return chr(value)
    return None


def parse_nist_sd19_chars(nist_dir: str) -> list[ManifestEntry]:
    """
    Parse NIST SD19 by_class character images.

    Expected layout:
      nist_dir/by_class/<hex_label>/hsf_<k>/*.png

    Special handling:
      - label is decoded from class folder hex (e.g., 4a -> 'J', 61 -> 'a')
      - split is assigned from hsf partition to avoid random leakage
    """
    by_class_dir = Path(nist_dir) / "by_class"
    entries: list[ManifestEntry] = []

    if not by_class_dir.exists():
        print(f"Warning: {by_class_dir} not found")
        return entries

    for class_dir in sorted(by_class_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        label = decode_nist_class_label(class_dir.name)
        if not label:
            continue

        for hsf_dir in sorted(class_dir.glob("hsf_*")):
            if not hsf_dir.is_dir():
                continue

            hsf_name = hsf_dir.name.lower()
            if hsf_name in {"hsf_0", "hsf_1", "hsf_2", "hsf_3"}:
                split = "train"
            elif hsf_name == "hsf_4":
                split = "val"
            elif hsf_name in {"hsf_6", "hsf_7"}:
                split = "test"
            else:
                # Unknown partitions default to train to preserve data availability.
                split = "train"

            for img_path in hsf_dir.glob("*.png"):
                entries.append({
                    "image_path": str(img_path),
                    "transcription": label,
                    "source": "nist_sd19",
                    "split": split,
                })

    return entries


def parse_iam_words(iam_dir: str) -> list[ManifestEntry]:
    """
    Parse IAM words.txt to extract (image_path, transcription) pairs.

    Format: wordID ok/err graylevel #components x y w h grammatical_tag transcription
    Example: a01-000u-00-00 ok 154 1 408 768 27 51 AT A
    """
    words_file = Path(iam_dir) / "ascii" / "words.txt"
    words_dir = Path(iam_dir) / "words"
    entries: list[ManifestEntry] = []

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


def parse_iam_lines(iam_dir: str) -> list[ManifestEntry]:
    """Parse IAM lines.txt for line-level data."""
    lines_file = Path(iam_dir) / "ascii" / "lines.txt"
    lines_dir = Path(iam_dir) / "lines"
    entries: list[ManifestEntry] = []

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


def load_iam_splits(iam_dir: str) -> dict[str, set[str]]:
    """Load official IAM writer-independent train/val/test splits."""
    splits: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
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


def assign_difficulty(entry: ManifestEntry) -> str:
    """Assign difficulty based on source and characteristics."""
    source = str(entry.get("source", ""))
    transcription = str(entry.get("transcription", ""))

    if source in {"emnist", "nist_sd19"}:
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
    nist_dir: str = "data/raw/nist_sd19",
    seed: int = 42,
    max_nist_to_iam_ratio: float = 1.0,
    datasets: tuple[str, ...] = ("iam",),
):
    """Create train.csv, val.csv, test.csv manifests."""
    os.makedirs(output_dir, exist_ok=True)
    selected_datasets = {d.strip().lower() for d in datasets if d.strip()}

    if not selected_datasets:
        raise ValueError("At least one dataset must be selected")

    unsupported = selected_datasets - {"iam", "nist_sd19"}
    if unsupported:
        raise ValueError(f"Unsupported datasets requested: {sorted(unsupported)}")

    # Collect all entries
    all_entries: list[ManifestEntry] = []

    if "iam" in selected_datasets:
        # IAM words
        iam_words = parse_iam_words(iam_dir)
        print(f"Parsed {len(iam_words)} IAM word entries")
        all_entries.extend(iam_words)

        # IAM lines
        iam_lines = parse_iam_lines(iam_dir)
        print(f"Parsed {len(iam_lines)} IAM line entries")
        all_entries.extend(iam_lines)

    if "nist_sd19" in selected_datasets:
        # NIST SD19 single-character samples
        nist_chars = parse_nist_sd19_chars(nist_dir)
        print(f"Parsed {len(nist_chars)} NIST SD19 character entries")
        all_entries.extend(nist_chars)

    # Load splits
    splits = load_iam_splits(iam_dir)

    # Assign split and difficulty
    for entry in all_entries:
        entry["difficulty"] = assign_difficulty(entry)

        # NIST entries already have deterministic split assignment from hsf_*.
        if entry.get("source") == "nist_sd19" and entry.get("split") in {"train", "val", "test"}:
            continue

        # Determine split from IAM official splits
        entry_id = str(entry.get("word_id", entry.get("line_id", "")))
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

        # Keep sequence-learning signals from IAM visible during training.
        # NIST is single-character; IAM has words/lines. Rebalance and interleave.
        if split_name == "train" and "iam" in selected_datasets and "nist_sd19" in selected_datasets:
            split_entries = rebalance_train_entries(
                split_entries,
                max_nist_to_iam_ratio=max_nist_to_iam_ratio,
                seed=seed,
            )

        split_entries = interleave_by_source(split_entries, seed=seed)

        output_path = os.path.join(output_dir, f"{split_name}.csv")

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(split_entries)

        source_counts: dict[str, int] = {}
        for e in split_entries:
            source = str(e.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1
        print(f"{split_name}: {len(split_entries)} entries -> {output_path} | sources={source_counts}")

    print(f"\nTotal entries: {len(all_entries)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create OCR manifests from selected datasets")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for train/val/test CSV")
    parser.add_argument("--iam-dir", default="data/raw/iam", help="IAM dataset root")
    parser.add_argument("--nist-dir", default="data/raw/nist_sd19", help="NIST SD19 dataset root")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for mixing/shuffling")
    parser.add_argument(
        "--max-nist-to-iam-ratio",
        type=float,
        default=1.0,
        help="Max NIST: IAM ratio for train split (used only when both datasets are enabled)",
    )
    parser.add_argument(
        "--datasets",
        default="iam",
        help="Comma-separated dataset list to include (supported: iam,nist_sd19). Default: iam",
    )

    args = parser.parse_args()
    datasets = tuple(part.strip() for part in args.datasets.split(",") if part.strip())

    create_manifests(
        output_dir=args.output_dir,
        iam_dir=args.iam_dir,
        nist_dir=args.nist_dir,
        seed=args.seed,
        max_nist_to_iam_ratio=args.max_nist_to_iam_ratio,
        datasets=datasets,
    )
