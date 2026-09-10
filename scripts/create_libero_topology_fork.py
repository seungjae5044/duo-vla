#!/usr/bin/env python3
"""Freeze an authenticated LIBERO TP1/DP2 topology-transition manifest."""

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

from duo_vla.topology_fork import create_topology_fork_manifest, write_topology_fork_manifest  # noqa: E402


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_checkpoint", type=_absolute_path)
    parser.add_argument("child_output_dir", type=_absolute_path)
    parser.add_argument("--child-config", required=True, type=_absolute_path)
    parser.add_argument("--child-physical-gpu", action="append", required=True, type=int)
    parser.add_argument("--expected-parent-manifest-sha256", required=True)
    parser.add_argument("--expected-parent-source-tree-sha256", required=True)
    parser.add_argument("--expected-parent-run-uuid", required=True)
    parser.add_argument("--expected-parent-update", required=True, type=int)
    parser.add_argument("--child-run-uuid", required=True)
    parser.add_argument("--output", required=True, type=_absolute_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_topology_fork_manifest(
        project_root=_PROJECT_ROOT,
        parent_checkpoint=args.parent_checkpoint,
        expected_parent_manifest_sha256=args.expected_parent_manifest_sha256,
        expected_parent_source_tree_sha256=args.expected_parent_source_tree_sha256,
        expected_parent_run_uuid=args.expected_parent_run_uuid,
        expected_parent_update=args.expected_parent_update,
        child_config_path=args.child_config,
        child_run_uuid=args.child_run_uuid,
        child_output_dir=args.child_output_dir,
        child_physical_gpu_indices=args.child_physical_gpu,
    )
    digest = write_topology_fork_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "child_output_dir": manifest["child"]["output_dir"],
                "child_run_uuid": manifest["child"]["run_uuid"],
                "child_topology": manifest["child"]["topology"]["kind"],
                "fork_manifest": str(args.output),
                "fork_manifest_sha256": digest,
                "parent_checkpoint": manifest["parent"]["checkpoint_identity"]["path"],
                "parent_manifest_sha256": manifest["parent"]["manifest_sha256"],
                "parent_topology": manifest["parent"]["topology"]["kind"],
                "parent_update": manifest["parent"]["update"],
                "status": "frozen",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
