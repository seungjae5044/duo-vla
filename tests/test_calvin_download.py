from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import warnings
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "calvin" / "download_dataset.sh"
ARCHIVE_BYTES = 555_309_812_705
ARCHIVE_SHA256 = "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74"
CRITICAL = (
    "training/ep_start_end_ids.npy",
    "training/lang_annotations/auto_lang_ann.npy",
    "training/scene_info.npy",
    "training/.hydra/merged_config.yaml",
    "validation/ep_start_end_ids.npy",
    "validation/.hydra/merged_config.yaml",
)


def _zip_info(name: str, kind: str = "file") -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    if kind == "directory":
        info.external_attr = ((stat.S_IFDIR | 0o700) << 16) | 0x10
    elif kind == "symlink":
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
    elif kind == "fifo":
        info.external_attr = (stat.S_IFIFO | 0o600) << 16
    else:
        info.external_attr = (stat.S_IFREG | 0o600) << 16
    return info


def _write_zip(path: Path, entries: Iterable[tuple[str, str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, kind, payload in entries:
                archive.writestr(_zip_info(name, kind), payload)


def _write_valid_archive(path: Path) -> None:
    entries: list[tuple[str, str, bytes]] = [
        ("task_ABC_D/", "directory", b""),
        ("task_ABC_D/training/", "directory", b""),
        ("task_ABC_D/training/lang_annotations/", "directory", b""),
        ("task_ABC_D/training/.hydra/", "directory", b""),
        ("task_ABC_D/validation/", "directory", b""),
        ("task_ABC_D/validation/.hydra/", "directory", b""),
    ]
    entries.extend((f"task_ABC_D/{relative}", "file", f"fixture:{relative}\n".encode()) for relative in CRITICAL)
    entries.append(("task_ABC_D/training/episode_0000000.npz", "file", b"fake-npz-payload"))
    _write_zip(path, entries)


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _fake_tool_environment(tmp_path: Path, *, inject_symlink: bool = False) -> tuple[Path, dict[str, str]]:
    data_root = tmp_path / "calvin"
    archive = data_root / "task_ABC_D.zip"
    tools = tmp_path / "fake-bin"
    tools.mkdir()
    _write_executable(
        tools / "stat",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" == 3 && "$1" == -c && "$2" == %s && "$3" == "${CALVIN_FAKE_ARCHIVE}" ]]; then
  printf '%s\\n' 555309812705
elif [[ "$#" == 4 && "$1" == -f && "$2" == -c && "$3" == '%a %S' ]]; then
  printf '%s\\n' '1000000000000000 1'
else
  exec /usr/bin/stat "$@"
fi
""",
    )
    _write_executable(
        tools / "sha256sum",
        """#!/usr/bin/env bash
set -euo pipefail
cat >/dev/null
exit 0
""",
    )
    _write_executable(
        tools / "mv",
        """#!/usr/bin/env bash
set -euo pipefail
destination="${@: -1}"
if [[ -n "${CALVIN_MV_LOG:-}" ]]; then
  printf '%s\\n' "${destination}" >>"${CALVIN_MV_LOG}"
fi
if [[ -n "${CALVIN_FAIL_PUBLISH_DESTINATION:-}" && "${destination}" == "${CALVIN_FAIL_PUBLISH_DESTINATION}" ]]; then
  exit 73
fi
exec /usr/bin/mv "$@"
""",
    )
    _write_executable(
        tools / "unzip",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${CALVIN_UNZIP_MARKER:-}" ]]; then
  : >"${CALVIN_UNZIP_MARKER}"
fi
/usr/bin/unzip "$@"
if [[ "${CALVIN_INJECT_SYMLINK:-0}" == 1 ]]; then
  destination="${@: -1}"
  ln -s /etc/passwd "${destination}/task_ABC_D/injected-link"
fi
""",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CALVIN_DATA_ROOT": str(data_root),
            "CALVIN_FAKE_ARCHIVE": str(archive),
            "CALVIN_INJECT_SYMLINK": "1" if inject_symlink else "0",
            "PATH": f"{tools}:{environment['PATH']}",
        }
    )
    return archive, environment


def _run_extract(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "extract"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_calvin_download_script_has_pinned_official_contract(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["CALVIN_DATA_ROOT"] = str(tmp_path / "calvin")
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--print-contract"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    contract = json.loads(completed.stdout)
    assert contract == {
        "archive": "task_ABC_D.zip",
        "archive_direct_reserve_bytes": 16 * 1024**3,
        "bytes": ARCHIVE_BYTES,
        "reserve_bytes": 150 * 1024**3,
        "sha256": ARCHIVE_SHA256,
        "url": "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip",
    }


def test_calvin_download_script_parses_as_bash() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_calvin_download_source_has_no_follow_scan_single_hash_and_commit_order() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "/workspace" not in source
    assert "150 * 1024 * 1024 * 1024" in source
    assert "16 * 1024 * 1024 * 1024" in source
    assert '! -e "${aria2_control_path}" && ! -L "${aria2_control_path}"' in source
    assert '"duo-vla-calvin-dataset-manifest-v3"' in source
    assert "O_NOFOLLOW" in source and "dir_fd=" in source
    assert "exact_tree_inventory" in source
    assert 'content_archive.open(info, "r")' in source
    assert "archived_sha256 == extracted_sha256" in source
    assert 'rm -rf -- "${active_stage_root}"' in source
    assert 'rm -rf -- "${final_root}"' not in source
    assert "refusing in-place blessing" in source
    assert "run_archive_direct_tool prepare" in source
    assert "run_archive_direct_tool verify" in source
    assert "download_archive archive-direct" in source
    assert '"${script_dir}/prepare_archive_direct.py"' in source

    all_case = source.split("  all)", 1)[1].split("  download)", 1)[0]
    extract_function = source.split("extract_archive() {", 1)[1].split("print_contract() {", 1)[0]
    assert all_case.count("verify_archive") == 1
    assert "verify_archive" not in extract_function
    archive_direct_case = source.split("  archive-direct)", 1)[1].split("  prepare-archive-direct)", 1)[0]
    prepare_direct_case = source.split("  prepare-archive-direct)", 1)[1].split("  verify-archive-direct)", 1)[0]
    assert "verify_archive" not in archive_direct_case
    assert "verify_archive" not in prepare_direct_case
    assert "require_headroom_for_download archive-direct" in prepare_direct_case

    root_publish = source.index('publish_noreplace "${stage_root}/task_ABC_D" "${final_root}"')
    index_publish = source.index('publish_noreplace "${private_index}" "${member_index_path}"')
    manifest_publish = source.index('publish_noreplace "${private_manifest}" "${manifest_path}"')
    assert root_publish < index_publish < manifest_publish


def test_prepare_archive_direct_mode_delegates_without_extracting_or_blessing_v3(tmp_path: Path) -> None:
    archive, environment = _fake_tool_environment(tmp_path)
    _write_valid_archive(archive)
    marker = tmp_path / "archive-direct-args.json"
    direct_python = tmp_path / "archive-direct-python"
    _write_executable(
        direct_python,
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
Path(os.environ["CALVIN_ARCHIVE_DIRECT_MARKER"]).write_text(json.dumps(sys.argv[1:]))
""",
    )
    environment["CALVIN_ARCHIVE_DIRECT_PYTHON"] = str(direct_python)
    environment["CALVIN_ARCHIVE_DIRECT_MARKER"] = str(marker)
    unzip_marker = tmp_path / "unzip-called"
    environment["CALVIN_UNZIP_MARKER"] = str(unzip_marker)

    completed = subprocess.run(
        ["bash", str(SCRIPT), "prepare-archive-direct"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = json.loads(marker.read_text(encoding="utf-8"))
    assert arguments == [
        str(ROOT / "scripts" / "calvin" / "prepare_archive_direct.py"),
        "prepare",
        "--data-root",
        str(archive.parent),
    ]
    assert not unzip_marker.exists()
    assert not (archive.parent / "task_ABC_D.members.sqlite3").exists()
    assert not (archive.parent / "task_ABC_D.manifest.json").exists()


def test_verify_archive_direct_mode_is_an_idempotent_strict_v4_delegate(tmp_path: Path) -> None:
    data_root = tmp_path / "calvin"
    marker = tmp_path / "archive-direct-verify-args.json"
    direct_python = tmp_path / "archive-direct-python"
    _write_executable(
        direct_python,
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
Path(os.environ["CALVIN_ARCHIVE_DIRECT_MARKER"]).write_text(json.dumps(sys.argv[1:]))
""",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CALVIN_ARCHIVE_DIRECT_MARKER": str(marker),
            "CALVIN_ARCHIVE_DIRECT_PYTHON": str(direct_python),
            "CALVIN_DATA_ROOT": str(data_root),
        }
    )

    completed = subprocess.run(
        ["bash", str(SCRIPT), "verify-archive-direct"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(marker.read_text(encoding="utf-8")) == [
        str(ROOT / "scripts" / "calvin" / "prepare_archive_direct.py"),
        "verify",
        "--data-root",
        str(data_root),
    ]
    assert not (data_root / "task_ABC_D.members.sqlite3").exists()


def test_download_rejects_a_symlink_archive_target_before_network_access(tmp_path: Path) -> None:
    data_root = tmp_path / "calvin"
    data_root.mkdir()
    target = tmp_path / "outside"
    target.write_bytes(b"must-not-change")
    (data_root / "task_ABC_D.zip").symlink_to(target)
    environment = os.environ.copy()
    environment["CALVIN_DATA_ROOT"] = str(data_root)

    completed = subprocess.run(
        ["bash", str(SCRIPT), "archive-direct"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert "no-follow regular file" in completed.stderr
    assert target.read_bytes() == b"must-not-change"


@pytest.mark.parametrize(
    "entries",
    [
        [("task_ABC_D/", "file", b"not-a-directory")],
        [("task_ABC_D//evil", "file", b"x")],
        [(r"task_ABC_D\evil", "file", b"x")],
        [("/task_ABC_D/evil", "file", b"x")],
        [("task_ABC_D/../evil", "file", b"x")],
        [("task_ABC_D/./evil", "file", b"x")],
        [("task_ABC_D/evil", "file", b"x"), ("task_ABC_D/evil", "file", b"y")],
        [("task_ABC_D/evil", "symlink", b"/etc/passwd")],
        [("task_ABC_D/evil", "fifo", b"")],
        [("task_ABC_D/a", "file", b"x"), ("task_ABC_D/a/b", "file", b"y")],
    ],
    ids=(
        "empty-relative-file",
        "double-slash",
        "backslash",
        "absolute",
        "dot-dot",
        "dot",
        "duplicate",
        "symlink",
        "fifo",
        "file-directory-collision",
    ),
)
def test_unsafe_zip_namespace_is_rejected_before_unzip(
    tmp_path: Path,
    entries: list[tuple[str, str, bytes]],
) -> None:
    archive, environment = _fake_tool_environment(tmp_path)
    _write_zip(archive, entries)
    marker = tmp_path / "unzip-called"
    environment["CALVIN_UNZIP_MARKER"] = str(marker)

    completed = _run_extract(environment)

    assert completed.returncode != 0
    assert not marker.exists()
    assert not (archive.parent / "task_ABC_D").exists()
    assert not list(archive.parent.glob(".task_ABC_D.extracting.*"))


def test_valid_archive_is_fully_indexed_and_published_manifest_last(tmp_path: Path) -> None:
    archive, environment = _fake_tool_environment(tmp_path)
    _write_valid_archive(archive)
    move_log = tmp_path / "moves.log"
    environment["CALVIN_MV_LOG"] = str(move_log)

    completed = _run_extract(environment)

    assert completed.returncode == 0, completed.stderr
    data_root = archive.parent
    final_root = data_root / "task_ABC_D"
    index_path = data_root / "task_ABC_D.members.sqlite3"
    manifest_path = data_root / "task_ABC_D.manifest.json"
    assert final_root.is_dir() and index_path.is_file() and manifest_path.is_file()
    assert not list(data_root.glob(".task_ABC_D.extracting.*"))
    assert move_log.read_text(encoding="utf-8").splitlines() == [
        str(final_root),
        str(index_path),
        str(manifest_path),
    ]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content_sha256 = manifest.pop("content_sha256")
    canonical = json.dumps(manifest, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    assert content_sha256 == hashlib.sha256(canonical).hexdigest()
    assert manifest["extraction"]["file_members_verified"] == len(CRITICAL) + 1
    with sqlite3.connect(index_path) as connection:
        assert connection.execute("SELECT count(*) FROM members").fetchone() == (len(CRITICAL) + 1,)
        stored_sha256 = connection.execute(
            "SELECT sha256 FROM members WHERE path = ?",
            ("training/episode_0000000.npz",),
        ).fetchone()
    with zipfile.ZipFile(archive) as source_archive:
        archived_payload = source_archive.read("task_ABC_D/training/episode_0000000.npz")
    assert stored_sha256 == (hashlib.sha256(archived_payload).hexdigest(),)
    assert (
        hashlib.sha256((final_root / "training" / "episode_0000000.npz").read_bytes()).hexdigest() == stored_sha256[0]
    )

    verified_again = _run_extract(environment)
    assert verified_again.returncode == 0, verified_again.stderr
    assert "fully verified" in verified_again.stdout


def test_existing_committed_tree_is_not_accepted_after_noncritical_member_tamper(tmp_path: Path) -> None:
    archive, environment = _fake_tool_environment(tmp_path)
    _write_valid_archive(archive)
    assert _run_extract(environment).returncode == 0
    final_root = archive.parent / "task_ABC_D"
    manifest_path = archive.parent / "task_ABC_D.manifest.json"
    manifest_before = manifest_path.read_bytes()
    (final_root / "training" / "episode_0000000.npz").write_bytes(b"tampered")

    completed = _run_extract(environment)

    assert completed.returncode != 0
    assert final_root.is_dir()
    assert manifest_path.read_bytes() == manifest_before


def test_ordinary_private_scan_failure_cleans_only_unique_staging_tree(tmp_path: Path) -> None:
    archive, environment = _fake_tool_environment(tmp_path, inject_symlink=True)
    _write_valid_archive(archive)

    completed = _run_extract(environment)

    assert completed.returncode != 0
    assert not list(archive.parent.glob(".task_ABC_D.extracting.*"))
    assert not (archive.parent / "task_ABC_D").exists()
    assert not (archive.parent / "task_ABC_D.members.sqlite3").exists()
    assert not (archive.parent / "task_ABC_D.manifest.json").exists()


def test_interrupted_publication_keeps_root_but_cannot_be_blessed_on_retry(tmp_path: Path) -> None:
    archive, environment = _fake_tool_environment(tmp_path)
    _write_valid_archive(archive)
    data_root = archive.parent
    final_root = data_root / "task_ABC_D"
    index_path = data_root / "task_ABC_D.members.sqlite3"
    manifest_path = data_root / "task_ABC_D.manifest.json"
    environment["CALVIN_FAIL_PUBLISH_DESTINATION"] = str(manifest_path)

    interrupted = _run_extract(environment)

    assert interrupted.returncode != 0
    assert final_root.is_dir() and index_path.is_file() and not manifest_path.exists()
    assert not list(data_root.glob(".task_ABC_D.extracting.*"))

    environment.pop("CALVIN_FAIL_PUBLISH_DESTINATION")
    retried = _run_extract(environment)
    assert retried.returncode != 0
    assert "without the index and manifest commit marker" in retried.stderr
    assert final_root.is_dir() and index_path.is_file() and not manifest_path.exists()


def test_nonempty_crash_stage_is_preserved_and_fails_closed(tmp_path: Path) -> None:
    archive, environment = _fake_tool_environment(tmp_path)
    _write_valid_archive(archive)
    stale = archive.parent / ".task_ABC_D.extracting.crashed"
    stale.mkdir()
    sentinel = stale / "operator-recovery-required"
    sentinel.write_text("do not delete crash evidence", encoding="utf-8")

    completed = _run_extract(environment)

    assert completed.returncode != 0
    assert "manual recovery" in completed.stderr
    assert sentinel.read_text(encoding="utf-8") == "do not delete crash evidence"
    assert not (archive.parent / "task_ABC_D").exists()
