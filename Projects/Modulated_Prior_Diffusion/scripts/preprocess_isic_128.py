#!/usr/bin/env python3
"""Resize the ISIC split once and write a file-manager-friendly copy on disk."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train_original_80", "train_crop_80", "test_original_20", "test_crop_20")


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def resize_tree(src_root: Path, dst_root: Path, size: int, overwrite: bool) -> None:
    resampling = Image.Resampling.BILINEAR
    dst_root.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        src_split = src_root / split
        dst_split = dst_root / split
        if not src_split.exists():
            raise FileNotFoundError(f"missing split directory: {src_split}")

        files = list(iter_images(src_split))
        print(f"{split}: {len(files)} files")
        for src_path in tqdm(files, desc=split, unit="img"):
            rel_path = src_path.relative_to(src_split)
            dst_path = dst_split / rel_path
            if dst_path.exists() and not overwrite:
                continue

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src_path) as image:
                resized = image.convert("RGB").resize((size, size), resampling)
                resized.save(dst_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/work-pvc/macw1030/isic_mpd_png"),
        help="raw ISIC split root",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("/work-pvc/macw1030/isic_mpd_png_128"),
        help="output root for resized images",
    )
    parser.add_argument("--size", type=int, default=128, help="resize target for width and height")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing resized files")
    args = parser.parse_args()

    resize_tree(args.src_root, args.dst_root, args.size, args.overwrite)
    print(f"done: wrote resized ISIC files to {args.dst_root}")


if __name__ == "__main__":
    main()
