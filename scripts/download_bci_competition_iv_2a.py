#!/usr/bin/env python3
"""Download BCI Competition IV Dataset 2a from the official source."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

DATASET_URL = (
    "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"
)
ARCHIVE_NAME = "BCICIV_2a_gdf.zip"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"


def download(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / ARCHIVE_NAME

    if not destination.exists():
        print(f"Downloading {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, destination)

    extracted_dir = output_dir / destination.stem
    if not extracted_dir.exists():
        print(f"Extracting to {extracted_dir}")
        shutil.unpack_archive(destination, extracted_dir)

    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        download(args.output_dir)
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
