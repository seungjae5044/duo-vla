#!/usr/bin/env python3
"""Prepare or verify the pinned CALVIN ABC->D archive-direct generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from duo_vla.data.calvin_archive import (  # noqa: E402
    CALVIN_ARCHIVE_BYTES,
    CALVIN_ARCHIVE_NAME,
    CALVIN_ARCHIVE_SHA256,
    CALVIN_ARCHIVE_URL,
    CALVIN_CHECKSUM_URL,
    CalvinArchiveReader,
    prepare_calvin_archive,
)


def _default_data_root() -> Path:
    cache_root = Path(os.environ.get("DUO_VLA_CACHE_ROOT", "/root/.cache/duo-vla"))
    return Path(os.environ.get("CALVIN_DATA_ROOT", str(cache_root / "data" / "calvin")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the authenticated v4 CALVIN ZIP-direct projection without extracting episode NPZ files."
    )
    parser.add_argument("command", choices=("prepare", "verify", "print-contract"))
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.absolute()
    archive = data_root / CALVIN_ARCHIVE_NAME
    if args.command == "print-contract":
        print(
            json.dumps(
                {
                    "archive": CALVIN_ARCHIVE_NAME,
                    "bytes": CALVIN_ARCHIVE_BYTES,
                    "checksum_url": CALVIN_CHECKSUM_URL,
                    "sha256": CALVIN_ARCHIVE_SHA256,
                    "url": CALVIN_ARCHIVE_URL,
                },
                allow_nan=False,
                sort_keys=True,
            )
        )
        return
    if args.command == "prepare":
        prepared = prepare_calvin_archive(archive, data_root)
        report = {
            "archive": str(prepared.archive_path),
            "dataset_root": str(prepared.dataset_root),
            "index": str(prepared.index_path),
            "manifest": str(prepared.manifest_path),
            "manifest_content_sha256": prepared.manifest["content_sha256"],
            "publication_warnings": list(prepared.publication_warnings),
            "status": "prepared" if not prepared.publication_warnings else "prepared-with-warnings",
        }
    else:
        with CalvinArchiveReader.from_manifest(data_root) as reader:
            verified_archive = str(reader.verified_archive_path)
        report = {
            "archive": verified_archive,
            "manifest": str(data_root / "task_ABC_D.manifest.json"),
            "status": "verified",
        }
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"CALVIN archive-direct preparation failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
