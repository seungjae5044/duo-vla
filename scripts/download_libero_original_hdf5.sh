#!/usr/bin/env bash
set -euo pipefail

readonly project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
readonly hf_home="${HF_HOME:-/root/.cache/huggingface}"
readonly environment_path="${DUO_VLA_TRAIN_VENV:-${cache_root}/venvs/train}"
readonly repository_id="yifengzhu-hf/LIBERO-datasets"
readonly revision="f13aa24a3da8c43c7225569f28c562979fa0e35a"
readonly destination="${LIBERO_ORIGINAL_HDF5_ROOT:-${cache_root}/data/libero/original-hdf5/${revision}}"
readonly metadata="${cache_root}/data/libero/download-metadata/original-hdf5-${revision}"
readonly inventory="${project_dir}/configs/libero_original_hdf5_inventory.json"

if [[ ! -x "${environment_path}/bin/hf" || ! -x "${environment_path}/bin/python" ]]; then
  echo "training environment is missing; run the matching bootstrap_train*_env.sh first" >&2
  exit 1
fi

mkdir -p "${destination}" "$(dirname "${metadata}")" "${hf_home}"

mapfile -t expected_paths < <(
  "${environment_path}/bin/python" -P -B - "${destination}" "${inventory}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
inventory_path = Path(sys.argv[2]).resolve()
inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
expected = sorted(entry["path"] for entry in inventory["files"])
if len(expected) != inventory["file_count"] or len(expected) != len(set(expected)):
    raise RuntimeError("original HDF5 inventory count or path uniqueness differs")

observed = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path.relative_to(root).parts[0] != ".cache"
}
extra = sorted(observed - set(expected))
if extra:
    raise RuntimeError(f"original HDF5 root contains out-of-scope files: {extra[:3]}")

for relative in expected:
    if "\n" in relative or "\r" in relative:
        raise RuntimeError("original HDF5 inventory path contains a line break")
    print(relative)
PY
)
if [[ "${#expected_paths[@]}" -ne 40 ]]; then
  echo "original HDF5 inventory did not produce exactly 40 paths" >&2
  exit 1
fi

if [[ -e "${destination}/.cache" && -e "${metadata}" ]]; then
  echo "download metadata exists in both the content and external roots" >&2
  exit 1
fi
if [[ -d "${metadata}" ]]; then
  mv "${metadata}" "${destination}/.cache"
fi

HF_HOME="${hf_home}" HF_XET_HIGH_PERFORMANCE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/dev/null \
  "${environment_path}/bin/hf" download "${repository_id}" "${expected_paths[@]}" \
  --repo-type dataset \
  --revision "${revision}" \
  --local-dir "${destination}"

if [[ ! -d "${destination}/.cache" ]]; then
  echo "download did not create the expected local metadata cache" >&2
  exit 1
fi
mv "${destination}/.cache" "${metadata}"

"${environment_path}/bin/python" -P -B - "${destination}" "${inventory}" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
inventory_path = Path(sys.argv[2]).resolve()
inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
expected = {entry["path"]: entry for entry in inventory["files"]}
observed = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
if observed != sorted(expected):
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    raise RuntimeError(f"original HDF5 file inventory differs: missing={missing[:3]}, extra={extra[:3]}")

total_bytes = 0
for relative, record in sorted(expected.items()):
    path = root / relative
    identity = os.lstat(path)
    if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
        raise RuntimeError(f"original HDF5 entry is not an independently materialized regular file: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
            total_bytes += len(block)
    if identity.st_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
        raise RuntimeError(f"original HDF5 identity mismatch: {relative}")

if len(observed) != inventory["file_count"] or total_bytes != inventory["total_bytes"]:
    raise RuntimeError("original HDF5 aggregate counts differ")
print(
    json.dumps(
        {
            "content_sha256": inventory["content_sha256"],
            "file_count": len(observed),
            "repository_id": inventory["repository"]["id"],
            "revision": inventory["repository"]["revision"],
            "root": str(root),
            "status": "ok",
            "total_bytes": total_bytes,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
