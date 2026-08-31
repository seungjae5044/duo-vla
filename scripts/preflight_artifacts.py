#!/usr/bin/env python3
"""Read-only storage, access, and immutable-revision checks before large downloads."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from huggingface_hub import HfApi


@dataclass(frozen=True, slots=True)
class HubArtifact:
    repo_type: str
    repo_id: str
    expected_revision: str
    observed_revision: str
    gated: bool | str
    private: bool
    files: int
    bytes: int


def inspect(repo_id: str, revision: str, *, repo_type: str) -> HubArtifact:
    api = HfApi()
    if repo_type == "model":
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
    elif repo_type == "dataset":
        info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
    else:
        raise ValueError(f"unsupported repo_type {repo_type}")
    total = sum((sibling.size or 0) for sibling in info.siblings)
    return HubArtifact(
        repo_type=repo_type,
        repo_id=repo_id,
        expected_revision=revision,
        observed_revision=info.sha,
        gated=info.gated,
        private=info.private,
        files=len(info.siblings),
        bytes=total,
    )


def main() -> None:
    artifacts = (
        inspect(
            "google/diffusiongemma-26B-A4B-it",
            "f7f5b7f5fa82ffc52addd066915886d497f5517b",
            repo_type="model",
        ),
        inspect(
            "HuggingFaceVLA/libero",
            "86958911c0f959db2bbbdb107eb3e17c5f9c798e",
            repo_type="dataset",
        ),
    )
    for artifact in artifacts:
        if artifact.observed_revision != artifact.expected_revision:
            raise RuntimeError(f"revision mismatch for {artifact.repo_id}")
    root_usage = shutil.disk_usage(Path("/"))
    workspace_usage = shutil.disk_usage(Path("/workspace"))
    output = {
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "storage": {
            "root_free_bytes": root_usage.free,
            "workspace_free_bytes": workspace_usage.free,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
