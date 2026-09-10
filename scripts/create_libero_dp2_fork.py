#!/usr/bin/env python3
"""Freeze an authenticated TP1-update1000 to DP2 new-run fork document."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if _SOURCE_ROOT.resolve(strict=True) != _SOURCE_ROOT or not stat.S_ISDIR(os.lstat(_SOURCE_ROOT).st_mode):
    raise RuntimeError("project source root must be a canonical real directory")
sys.path.insert(0, str(_SOURCE_ROOT))

from duo_vla.dp2_fork import create_dp2_fork_manifest, write_dp2_fork_manifest  # noqa: E402


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_checkpoint", type=_absolute_path)
    parser.add_argument("child_output_dir", type=_absolute_path)
    parser.add_argument("--config", type=Path, default=_PROJECT_ROOT / "configs/libero_dp2_fused_v2_b32.toml")
    parser.add_argument("--expected-parent-manifest-sha256", required=True)
    parser.add_argument("--expected-parent-source-tree-sha256", required=True)
    parser.add_argument("--expected-parent-run-uuid", required=True)
    parser.add_argument("--child-run-uuid", required=True)
    parser.add_argument("--output", required=True, type=_absolute_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve(strict=True)
    expected_config = _PROJECT_ROOT / "configs/libero_dp2_fused_v2_b32.toml"
    if config_path != expected_config:
        raise ValueError(f"DP2 fork config must be {expected_config}")
    manifest = create_dp2_fork_manifest(
        project_root=_PROJECT_ROOT,
        parent_checkpoint=args.parent_checkpoint,
        expected_parent_manifest_sha256=args.expected_parent_manifest_sha256,
        expected_parent_source_tree_sha256=args.expected_parent_source_tree_sha256,
        expected_parent_run_uuid=args.expected_parent_run_uuid,
        config_path=config_path,
        child_run_uuid=args.child_run_uuid,
        child_output_dir=args.child_output_dir,
    )
    digest = write_dp2_fork_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "child_output_dir": manifest["child"]["output_dir"],
                "child_run_uuid": manifest["child"]["run_uuid"],
                "fork_manifest": str(args.output),
                "fork_manifest_sha256": digest,
                "parent_checkpoint": manifest["parent"]["checkpoint_identity"]["path"],
                "parent_manifest_sha256": manifest["parent"]["manifest_sha256"],
                "status": "frozen",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
