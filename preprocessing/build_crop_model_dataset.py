from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd


# ============================================================
# PlantScan AI
# Crop Model Dataset Builder
#
# Purpose:
#   Build a reproducible crop-specific classification dataset
#   from the frozen Dataset v1.
#
# Input:
#   datasets/processed/cleaned_raw
#
# Taxonomy:
#   datasets/reports/model/final_class_taxonomy_v2.csv
#
# Output:
#   datasets/model/<crop>/
#       train/
#       validation/
#       test/
#       manifest.csv
#
# Dataset v1 is NEVER modified.
# ============================================================


SEED = 42

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "cleaned_raw"
)

TAXONOMY_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "reports"
    / "model"
    / "final_class_taxonomy_v2.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "model"
)


# ============================================================
# Crop configuration
# ============================================================

SUPPORTED_CROPS = {
    "tomato",
    "banana",
    "cotton",
    "mango",
    "rice",
    "potato",
    "apple",
    "bell_pepper",
    "maize",
    "grape",
}


CROP_NAME_MAP = {
    "tomato": "Tomato",
    "banana": "Banana",
    "cotton": "Cotton",
    "mango": "Mango",
    "rice": "Rice",
    "potato": "Potato",
    "apple": "Apple",
    "bell_pepper": "Bell pepper",
    "maize": "Maize",
    "grape": "Grape",
}


# ============================================================
# Helpers
# ============================================================

def normalize_crop_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def is_image(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def reset_output_directory(output_dir: Path, overwrite: bool = False) -> None:
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"\nOutput directory already exists:\n"
                f"{output_dir}\n\n"
                f"Refusing to overwrite an existing model dataset.\n"
                f"Use --overwrite or delete it manually if you intentionally want to rebuild it."
            )


def deterministic_split(items: list[dict]) -> dict[str, list[dict]]:
    rng = random.Random(SEED)

    shuffled = items.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)

    train_count = int(total * TRAIN_RATIO)
    validation_count = int(total * VALIDATION_RATIO)

    train = shuffled[:train_count]

    validation = shuffled[
        train_count:
        train_count + validation_count
    ]

    test = shuffled[
        train_count + validation_count:
    ]

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def copy_items(
    items: list[dict],
    split: str,
    output_dir: Path,
) -> None:

    split_dir = output_dir / split

    for item in items:

        destination_class_dir = (
            split_dir
            / item["canonical_class"]
        )

        destination_class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_path = Path(item["source_path"])

        destination_path = (
            destination_class_dir
            / source_path.name
        )

        if destination_path.exists():
            prefix = hashlib.md5(item["source_path"].encode("utf-8")).hexdigest()[:6]
            destination_path = (
                destination_class_dir
                / f"{source_path.stem}_{prefix}{source_path.suffix}"
            )

        shutil.copy2(
            source_path,
            destination_path,
        )

        item["output_path"] = str(
            destination_path.relative_to(
                PROJECT_ROOT
            )
        )


def write_manifest(
    items: list[dict],
    output_path: Path,
) -> None:

    fieldnames = [
        "split",
        "crop",
        "canonical_class",
        "dataset",
        "raw_class",
        "source_path",
        "output_path",
        "sha256",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for item in items:
            writer.writerow(
                {
                    key: item.get(key, "")
                    for key in fieldnames
                }
            )


# ============================================================
# Taxonomy
# ============================================================

def load_taxonomy(crop_display_name: str) -> pd.DataFrame:

    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError(
            f"Taxonomy file not found:\n"
            f"{TAXONOMY_PATH}"
        )

    taxonomy = pd.read_csv(
        TAXONOMY_PATH
    )

    required_columns = {
        "dataset",
        "raw_class",
        "image_count",
        "crop",
        "category",
        "canonical_class",
        "decision",
        "reason",
    }

    missing = (
        required_columns
        - set(taxonomy.columns)
    )

    if missing:
        raise ValueError(
            f"Taxonomy is missing columns: {sorted(missing)}"
        )

    crop_taxonomy = taxonomy[
        taxonomy["crop"]
        .astype(str)
        .str.strip()
        .str.lower()
        == crop_display_name.lower()
    ].copy()

    crop_taxonomy = crop_taxonomy[
        crop_taxonomy["decision"]
        .astype(str)
        .str.lower()
        == "include"
    ]

    crop_taxonomy = crop_taxonomy[
        crop_taxonomy["canonical_class"]
        .notna()
    ]

    if crop_taxonomy.empty:
        raise ValueError(
            f"No included taxonomy classes found for {crop_display_name}."
        )

    return crop_taxonomy


# ============================================================
# Source mapping
# ============================================================

def build_source_mapping(
    crop_taxonomy: pd.DataFrame,
) -> dict[tuple[str, str], str]:

    mapping = {}

    for _, row in crop_taxonomy.iterrows():

        dataset = str(
            row["dataset"]
        ).strip()

        raw_class = str(
            row["raw_class"]
        ).strip()

        canonical_class = str(
            row["canonical_class"]
        ).strip()

        key = (
            dataset.lower(),
            raw_class.lower(),
        )

        if key in mapping:

            if mapping[key] != canonical_class:
                raise ValueError(
                    "Conflicting taxonomy mapping:\n"
                    f"{key}\n"
                    f"{mapping[key]} vs {canonical_class}"
                )

        mapping[key] = canonical_class

    return mapping


# ============================================================
# Discover source images
# ============================================================

def discover_images(
    crop_display_name: str,
    crop_taxonomy: pd.DataFrame,
) -> list[dict]:

    mapping = build_source_mapping(
        crop_taxonomy
    )

    images = []
    seen_paths = set()

    for _, row in crop_taxonomy.iterrows():

        dataset = str(
            row["dataset"]
        ).strip()

        raw_class = str(
            row["raw_class"]
        ).strip()

        canonical_class = str(
            row["canonical_class"]
        ).strip()

        dataset_root = (
            SOURCE_ROOT / dataset
        )

        if not dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset source directory not found:\n"
                f"{dataset_root}"
            )

        # The cleaned dataset preserves source dataset
        # structure. Search for the exact raw-class folder
        # beneath that dataset.
        matching_directories = [
            p
            for p in dataset_root.rglob("*")
            if p.is_dir()
            and p.name.strip().lower()
            == raw_class.lower()
        ]

        if not matching_directories:
            print(
                f"WARNING: source class directory not found: "
                f"{dataset}/{raw_class}"
            )
            continue

        for class_dir in matching_directories:

            for image_path in class_dir.rglob("*"):

                if not is_image(image_path):
                    continue

                src_str = str(image_path)
                if src_str in seen_paths:
                    continue
                seen_paths.add(src_str)

                images.append(
                    {
                        "crop": crop_display_name,
                        "dataset": dataset,
                        "raw_class": raw_class,
                        "canonical_class": canonical_class,
                        "source_path": src_str,
                        "output_path": "",
                    }
                )

    return images


# ============================================================
# Validation
# ============================================================

def validate_discovered_images(
    images: list[dict],
) -> None:

    if not images:
        raise ValueError(
            "No images were discovered."
        )

    source_paths = [
        item["source_path"]
        for item in images
    ]

    if len(source_paths) != len(set(source_paths)):
        raise ValueError(
            "Duplicate source paths detected."
        )

    print(
        f"Discovered images: {len(images):,}"
    )

    class_counts = Counter(
        item["canonical_class"]
        for item in images
    )

    print("\nCanonical class counts:")

    for class_name in sorted(class_counts):
        print(
            f"  {class_name:40} "
            f"{class_counts[class_name]:6,}"
        )


def validate_splits(
    splits: dict[str, list[dict]],
) -> None:

    split_names = [
        "train",
        "validation",
        "test",
    ]

    paths = {
        split: {
            item["source_path"]
            for item in splits[split]
        }
        for split in split_names
    }

    for i, first in enumerate(split_names):
        for second in split_names[i + 1:]:

            overlap = (
                paths[first]
                & paths[second]
            )

            if overlap:
                raise ValueError(
                    f"Split overlap detected: "
                    f"{first} ∩ {second} = "
                    f"{len(overlap)}"
                )


def validate_output(
    output_dir: Path,
    expected_items: list[dict],
) -> None:

    actual_files = [
        p
        for p in output_dir.rglob("*")
        if is_image(p)
    ]

    if len(actual_files) != len(expected_items):
        raise ValueError(
            "Output image count mismatch:\n"
            f"Expected: {len(expected_items):,}\n"
            f"Found:    {len(actual_files):,}"
        )

    manifest_path = (
        output_dir / "manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Manifest was not created."
        )

    manifest = pd.read_csv(
        manifest_path
    )

    if len(manifest) != len(expected_items):
        raise ValueError(
            "Manifest row count mismatch:\n"
            f"Expected: {len(expected_items):,}\n"
            f"Found:    {len(manifest):,}"
        )

    if manifest["source_path"].duplicated().any():
        raise ValueError(
            "Duplicate source paths in manifest."
        )

    if manifest["output_path"].duplicated().any():
        raise ValueError(
            "Duplicate output paths in manifest."
        )

    print("\nOutput validation: PASS")


# ============================================================
# Main
# ============================================================

def build_single_crop(crop_key: str, overwrite: bool = False) -> None:
    crop_display_name = CROP_NAME_MAP[crop_key]

    print("=" * 80)
    print(f"PlantScan AI - Building {crop_display_name} Model Dataset ({crop_key})")
    print("=" * 80)

    print(f"\nSource dataset:\n{SOURCE_ROOT}")
    print(f"\nTaxonomy:\n{TAXONOMY_PATH}")

    output_dir = OUTPUT_ROOT / crop_key
    print(f"\nOutput:\n{output_dir}")

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Frozen Dataset v1 not found:\n{SOURCE_ROOT}")

    crop_taxonomy = load_taxonomy(crop_display_name)
    print(f"\nIncluded taxonomy rows: {len(crop_taxonomy)}")

    print("\nCanonical classes:")
    for class_name in sorted(crop_taxonomy["canonical_class"].unique()):
        print(f"  {class_name}")

    images = discover_images(crop_display_name, crop_taxonomy)
    validate_discovered_images(images)

    print("\nCalculating source image hashes...")
    hashes = {}
    for item in images:
        digest = file_sha256(Path(item["source_path"]))
        if digest in hashes:
            raise ValueError(
                "Duplicate image content detected inside model dataset:\n"
                f"Existing: {hashes[digest]}\n"
                f"Duplicate: {item['source_path']}"
            )
        hashes[digest] = item["source_path"]
        item["sha256"] = digest

    print(f"Unique image hashes: {len(hashes):,}")

    splits = deterministic_split(images)
    validate_splits(splits)

    print("\nSplit counts:")
    for split_name in ["train", "validation", "test"]:
        print(f"  {split_name:12} {len(splits[split_name]):6,}")

    reset_output_directory(output_dir, overwrite=overwrite)

    for split_name in ["train", "validation", "test"]:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)

    print("\nCopying images...")
    all_items = []
    for split_name in ["train", "validation", "test"]:
        copy_items(splits[split_name], split_name, output_dir)
        for item in splits[split_name]:
            item["split"] = split_name
            all_items.append(item)

    manifest_path = output_dir / "manifest.csv"
    write_manifest(all_items, manifest_path)

    validate_output(output_dir, all_items)

    report_path = output_dir / "class_counts.csv"
    counts = Counter((item["split"], item["canonical_class"]) for item in all_items)

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "canonical_class", "image_count"])
        for split_name in ["train", "validation", "test"]:
            for class_name in sorted({item["canonical_class"] for item in all_items}):
                writer.writerow([split_name, class_name, counts[(split_name, class_name)]])

    print("\n" + "=" * 80)
    print(f"{crop_display_name} MODEL DATASET CREATED")
    print("=" * 80)
    print(f"Images: {len(all_items):,}")
    print(f"Output: {output_dir}")
    print(f"STATUS: PASS\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build crop-specific model datasets from PlantScan AI Dataset v1."
    )
    parser.add_argument(
        "--crop",
        required=True,
        choices=sorted(SUPPORTED_CROPS | {"all"}),
        help="Crop model to build ('all' for all 10 crops).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )

    args = parser.parse_args()

    if args.crop == "all":
        crops_to_build = sorted(SUPPORTED_CROPS)
    else:
        crops_to_build = [args.crop]

    for c_key in crops_to_build:
        build_single_crop(c_key, overwrite=args.overwrite)

    print("\n" + "=" * 80)
    print("ALL CROP MODEL DATASETS PROCESSED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()