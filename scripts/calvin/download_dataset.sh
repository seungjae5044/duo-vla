#!/usr/bin/env bash
set -euo pipefail

readonly archive_name="task_ABC_D.zip"
readonly archive_url="http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"
readonly checksum_url="http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt"
readonly expected_archive_bytes=555309812705
readonly expected_archive_sha256="c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74"
readonly reserve_bytes=$((150 * 1024 * 1024 * 1024))
readonly archive_direct_reserve_bytes=$((16 * 1024 * 1024 * 1024))
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly archive_direct_python="${CALVIN_ARCHIVE_DIRECT_PYTHON:-python3}"
readonly cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
readonly data_root="${CALVIN_DATA_ROOT:-${cache_root}/data/calvin}"
readonly archive_path="${data_root}/${archive_name}"
readonly aria2_control_path="${archive_path}.aria2"
readonly final_root="${data_root}/task_ABC_D"
readonly manifest_path="${data_root}/task_ABC_D.manifest.json"
readonly member_index_path="${data_root}/task_ABC_D.members.sqlite3"
readonly mode="${1:-all}"
active_stage_root=""
verified_archive_identity=""

fail() {
  printf 'CALVIN dataset setup failed: %s\n' "$*" >&2
  exit 1
}

cleanup_on_exit() {
  local status=$?
  trap - EXIT HUP INT TERM
  if [[ -n "${active_stage_root}" ]]; then
    case "${active_stage_root}" in
      "${data_root}"/.task_ABC_D.extracting.*)
        if [[ -d "${active_stage_root}" && ! -L "${active_stage_root}" ]]; then
          rm -rf -- "${active_stage_root}" \
            || printf 'CALVIN dataset setup warning: could not clean private staging tree %s\n' \
              "${active_stage_root}" >&2
        else
          printf 'CALVIN dataset setup warning: refusing to clean unexpected staging path %s\n' \
            "${active_stage_root}" >&2
        fi
        ;;
      *)
        printf 'CALVIN dataset setup warning: refusing to clean out-of-scope path %s\n' \
          "${active_stage_root}" >&2
        ;;
    esac
  fi
  exit "${status}"
}

trap cleanup_on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

fsync_regular_file() {
  python3 - "$1" <<'PY'
import os
import stat
import sys

flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
descriptor = os.open(sys.argv[1], flags)
try:
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise RuntimeError(f"not a no-follow regular file: {sys.argv[1]}")
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}

fsync_directory() {
  python3 - "$1" <<'PY'
import os
import stat
import sys

flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW
descriptor = os.open(sys.argv[1], flags)
try:
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        raise RuntimeError(f"not a no-follow directory: {sys.argv[1]}")
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}

available_bytes() {
  local blocks block_size
  read -r blocks block_size < <(stat -f -c '%a %S' "${data_root}")
  printf '%s\n' "$((blocks * block_size))"
}

archive_size() {
  if [[ -f "${archive_path}" ]]; then
    stat -c '%s' "${archive_path}"
  else
    printf '0\n'
  fi
}

archive_downloaded_bytes() {
  if [[ ! -f "${archive_path}" ]]; then
    printf '0\n'
    return
  fi
  local logical blocks block_size allocated
  read -r logical blocks block_size < <(stat -c '%s %b %B' "${archive_path}")
  allocated=$((blocks * block_size))
  if (( allocated < logical )); then
    printf '%s\n' "${allocated}"
  else
    printf '%s\n' "${logical}"
  fi
}

require_headroom_for_download() {
  local storage_mode="${1:-extracted}"
  local current remaining required available
  # Multi-range aria2 downloads are sparse: logical size can jump near the final
  # object size after the first segment. Count allocated bytes for safe restart headroom.
  current="$(archive_downloaded_bytes)"
  (( current <= expected_archive_bytes )) || fail "partial archive is larger than the pinned object"
  remaining=$((expected_archive_bytes - current))
  if [[ "${storage_mode}" == "archive-direct" ]]; then
    # P0 keeps the verified ZIP and creates only the v2 SQLite index plus six
    # projected metadata files. It never reserves or extracts the 558 GB payload.
    required=$((remaining + archive_direct_reserve_bytes))
  else
    # The outer ZIP contains already-compressed per-frame NPZs. Reserve one archive-sized
    # extraction plus 150 GiB until the exact central-directory total is available.
    required=$((remaining + expected_archive_bytes + reserve_bytes))
  fi
  available="$(available_bytes)"
  (( available >= required )) || fail "insufficient headroom: available=${available}, required=${required}"
}

verify_remote_contract() {
  local content_length published
  content_length="$(curl --fail --silent --show-error --location --head --max-time 60 "${archive_url}" \
    | tr -d '\r' | awk 'tolower($1) == "content-length:" {print $2}' | tail -1)"
  [[ "${content_length}" == "${expected_archive_bytes}" ]] \
    || fail "remote Content-Length changed: ${content_length:-missing}"
  published="$(curl --fail --silent --show-error --location --max-time 60 "${checksum_url}" \
    | awk -v name="${archive_name}" '$2 == name {print $1}')"
  [[ "${published}" == "${expected_archive_sha256}" ]] \
    || fail "published SHA-256 changed: ${published:-missing}"
}

download_archive() {
  local storage_mode="${1:-extracted}"
  mkdir -p "${data_root}"
  if [[ -e "${archive_path}" || -L "${archive_path}" ]]; then
    [[ -f "${archive_path}" && ! -L "${archive_path}" ]] \
      || fail "archive download target is not a no-follow regular file"
    [[ "$(stat -c '%h' "${archive_path}")" == 1 ]] \
      || fail "archive download target has external hard links"
  fi
  [[ ! -L "${aria2_control_path}" ]] || fail "archive download control path is a symlink"
  require_headroom_for_download "${storage_mode}"
  verify_remote_contract
  if [[ "$(archive_size)" == "${expected_archive_bytes}" \
    && ! -e "${aria2_control_path}" && ! -L "${aria2_control_path}" ]]; then
    printf 'Archive already has the pinned byte length; proceeding to SHA-256 verification.\n'
    return
  fi
  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --allow-overwrite=true \
      --auto-file-renaming=false \
      --check-integrity=true \
      --continue=true \
      --dir="${data_root}" \
      --file-allocation=none \
      --max-connection-per-server=16 \
      --max-tries=0 \
      --min-split-size=64M \
      --out="${archive_name}" \
      --retry-wait=10 \
      --split=16 \
      --summary-interval=60 \
      "${archive_url}"
  else
    printf 'aria2c is unavailable; falling back to single-stream resumable curl.\n' >&2
    curl \
      --continue-at - \
      --fail \
      --location \
      --output "${archive_path}" \
      --retry 100 \
      --retry-all-errors \
      --retry-delay 10 \
      --show-error \
      "${archive_url}"
  fi
}

run_archive_direct_tool() {
  local command="$1"
  # This standalone Python 3.11 preparation owns the v4 commit protocol. It
  # accepts only a complete v4 generation on verify and will not promote a v3
  # tree, a mixed generation, or a partially published root in place.
  "${archive_direct_python}" "${script_dir}/prepare_archive_direct.py" "${command}" --data-root "${data_root}"
}

verify_archive() {
  local identity_before identity_after
  [[ -f "${archive_path}" ]] || fail "archive is missing: ${archive_path}"
  [[ "$(archive_size)" == "${expected_archive_bytes}" ]] || fail "archive byte length is incomplete"
  identity_before="$(stat -c '%d:%i:%s:%y:%z' "${archive_path}")"
  printf '%s  %s\n' "${expected_archive_sha256}" "${archive_path}" | sha256sum --check --status \
    || fail "archive SHA-256 mismatch"
  fsync_regular_file "${archive_path}" || fail "cannot durably sync the verified archive"
  fsync_directory "${data_root}" || fail "cannot durably sync the archive directory"
  identity_after="$(stat -c '%d:%i:%s:%y:%z' "${archive_path}")"
  [[ "${identity_before}" == "${identity_after}" ]] || fail "archive changed while SHA-256 was verified"
  verified_archive_identity="${identity_after}"
}

require_verified_archive_unchanged() {
  [[ -n "${verified_archive_identity}" ]] || fail "archive was not verified in this process"
  [[ "$(stat -c '%d:%i:%s:%y:%z' "${archive_path}")" == "${verified_archive_identity}" ]] \
    || fail "archive identity changed after SHA-256 verification"
}

uncompressed_bytes() {
  LC_ALL=C zipinfo -t "${archive_path}" | awk 'END {print $3}'
}

validate_member_paths() {
  python3 - "${archive_path}" <<'PY' || fail "ZIP contains an unsafe or duplicate member"
import stat
import sys
import zipfile

prefix = "task_ABC_D/"
seen_names = set()
seen_relative = set()
files = set()
directories = set()
with zipfile.ZipFile(sys.argv[1]) as archive:
    for info in archive.infolist():
        name = info.filename
        if not name or "\x00" in name or "\\" in name or name.startswith("/") or "//" in name:
            raise RuntimeError(f"non-canonical ZIP member path: {name!r}")
        if name in seen_names or not name.startswith(prefix):
            raise RuntimeError(f"duplicate or out-of-root ZIP member: {name!r}")
        seen_names.add(name)

        relative_with_marker = name[len(prefix) :]
        is_directory = info.is_dir()
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        expected_type = stat.S_IFDIR if is_directory else stat.S_IFREG
        if stat.S_ISLNK(mode) or file_type != expected_type:
            raise RuntimeError(f"ZIP member is not a regular file/directory: {name!r}")
        if info.flag_bits & 1:
            raise RuntimeError(f"encrypted ZIP member is unsupported: {name!r}")
        if not relative_with_marker:
            if not is_directory or name != prefix:
                raise RuntimeError(f"empty relative ZIP member path: {name!r}")
            if info.file_size != 0 or info.compress_size != 0:
                raise RuntimeError("root ZIP directory contains data")
            continue
        if is_directory:
            if not relative_with_marker.endswith("/"):
                raise RuntimeError(f"directory ZIP member lacks one trailing slash: {name!r}")
            if info.file_size != 0 or info.compress_size != 0:
                raise RuntimeError(f"ZIP directory contains data: {name!r}")
            relative = relative_with_marker[:-1]
        else:
            if relative_with_marker.endswith("/"):
                raise RuntimeError(f"regular ZIP member has a trailing slash: {name!r}")
            relative = relative_with_marker
        components = relative.split("/")
        if not relative or any(component in ("", ".", "..") for component in components):
            raise RuntimeError(f"non-canonical relative ZIP member path: {name!r}")
        canonical = prefix + "/".join(components) + ("/" if is_directory else "")
        if name != canonical or relative in seen_relative:
            raise RuntimeError(f"aliased or duplicate ZIP member path: {name!r}")
        seen_relative.add(relative)
        for depth in range(1, len(components)):
            directories.add("/".join(components[:depth]))
        if is_directory:
            directories.add(relative)
        else:
            files.add(relative)

if files & directories:
    raise RuntimeError("ZIP file/directory namespace collision")
PY
}

authenticate_dataset_tree() {
  local action="$1"
  local verified_root="$2"
  local selected_index="$3"
  local selected_manifest="$4"
  python3 - "${action}" "${verified_root}" "${archive_path}" "${selected_index}" \
    "${selected_manifest}" "${expected_archive_bytes}" "${expected_archive_sha256}" \
    "${archive_url}" "${checksum_url}" "$(uncompressed_bytes)" <<'PY'
import hashlib
import json
import os
import sqlite3
import stat
import sys
import zipfile
import zlib
from pathlib import Path

action = sys.argv[1]
root = Path(sys.argv[2])
archive_path = Path(sys.argv[3])
index_path = Path(sys.argv[4])
manifest_path = Path(sys.argv[5])
archive_bytes = int(sys.argv[6])
archive_sha256 = sys.argv[7]
archive_url = sys.argv[8]
checksum_url = sys.argv[9]
reported_uncompressed_bytes = int(sys.argv[10])
prefix = "task_ABC_D/"
index_schema = "duo-vla-calvin-member-index-v1"
critical = (
    "training/ep_start_end_ids.npy",
    "training/lang_annotations/auto_lang_ann.npy",
    "training/scene_info.npy",
    "training/.hydra/merged_config.yaml",
    "validation/ep_start_end_ids.npy",
    "validation/.hydra/merged_config.yaml",
)
directory_flags = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | os.O_DIRECTORY
    | os.O_NOFOLLOW
)
file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical_json_bytes(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("ascii")


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field {name!r}")
        result[name] = value
    return result


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def read_regular_file(path, maximum_bytes=None):
    descriptor = os.open(path, file_flags)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"not a no-follow regular file: {path}")
        require(maximum_bytes is None or before.st_size <= maximum_bytes, f"file is unexpectedly large: {path}")
        chunks = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            require(bool(block), f"file changed while reading: {path}")
            chunks.append(block)
            remaining -= len(block)
        require(not os.read(descriptor, 1), f"file grew while reading: {path}")
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            f"file changed while reading: {path}",
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def regular_file_identity(path, durable=False):
    descriptor = os.open(path, file_flags)
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"not a no-follow regular file: {path}")
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
        if durable:
            os.fsync(descriptor)
        after = os.fstat(descriptor)
        require(
            size == before.st_size
            and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            f"file changed while hashing: {path}",
        )
        return {"bytes": size, "sha256": digest.hexdigest()}
    finally:
        os.close(descriptor)


def open_regular_beneath(root_descriptor, relative):
    components = relative.split("/")
    require(components and all(component not in ("", ".", "..") for component in components), "unsafe path")
    parent = os.dup(root_descriptor)
    try:
        for component in components[:-1]:
            child = os.open(component, directory_flags, dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(components[-1], file_flags, dir_fd=parent)
    finally:
        os.close(parent)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise RuntimeError(f"extracted member is not a private regular file: {relative}")
    return descriptor


def read_extracted_member(root_descriptor, relative, info, durable):
    descriptor = open_regular_beneath(root_descriptor, relative)
    digest = hashlib.sha256()
    checksum = 0
    size = 0
    try:
        before = os.fstat(descriptor)
        require(before.st_size == info.file_size, f"extracted member has the wrong size: {relative}")
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            size += len(block)
            checksum = zlib.crc32(block, checksum)
            digest.update(block)
        if durable:
            os.fsync(descriptor)
        after = os.fstat(descriptor)
        require(
            size == info.file_size
            and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            f"extracted member changed while hashing: {relative}",
        )
        checksum &= 0xFFFFFFFF
        require(checksum == info.CRC, f"extracted member CRC differs from the pinned ZIP: {relative}")
        return size, checksum, digest.hexdigest()
    finally:
        os.close(descriptor)


def exact_tree_inventory(root_descriptor, expected_files, expected_directories):
    file_count = 0
    directory_count = 0

    def visit(descriptor, prefix_path):
        nonlocal file_count, directory_count
        with os.scandir(descriptor) as iterator:
            entries = list(iterator)
        for entry in entries:
            require(entry.name not in ("", ".", "..") and "/" not in entry.name, "unsafe extracted name")
            relative = f"{prefix_path}/{entry.name}" if prefix_path else entry.name
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                require(relative in expected_directories, f"unexpected extracted directory: {relative}")
                child = os.open(entry.name, directory_flags, dir_fd=descriptor)
                try:
                    directory_count += 1
                    visit(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                require(metadata.st_nlink == 1, f"extracted file has external hard links: {relative}")
                require(relative in expected_files, f"unexpected extracted file: {relative}")
                file_count += 1
            else:
                raise RuntimeError(f"extracted path is a symlink or non-regular type: {relative}")

    visit(root_descriptor, "")
    require(file_count == len(expected_files), "extracted file inventory differs from the pinned ZIP")
    require(directory_count == len(expected_directories), "extracted directory inventory differs from the pinned ZIP")


def fsync_tree_directories(descriptor):
    with os.scandir(descriptor) as iterator:
        directory_names = [entry.name for entry in iterator if entry.is_dir(follow_symlinks=False)]
    for name in directory_names:
        child = os.open(name, directory_flags, dir_fd=descriptor)
        try:
            fsync_tree_directories(child)
        finally:
            os.close(child)
    os.fsync(descriptor)


def fsync_parent(path):
    descriptor = os.open(path.parent, directory_flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_zip_info(info, seen_names, seen_relative):
    name = info.filename
    require(name and "\x00" not in name and "\\" not in name, f"non-canonical ZIP member path: {name!r}")
    require(not name.startswith("/") and "//" not in name, f"non-canonical ZIP member path: {name!r}")
    require(name not in seen_names and name.startswith(prefix), f"duplicate/out-of-root ZIP member: {name!r}")
    seen_names.add(name)
    relative_marker = name[len(prefix) :]
    is_directory = info.is_dir()
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    expected_type = stat.S_IFDIR if is_directory else stat.S_IFREG
    require(not stat.S_ISLNK(mode) and file_type == expected_type, f"non-regular ZIP member: {name!r}")
    require(not (info.flag_bits & 1), f"encrypted ZIP member is unsupported: {name!r}")
    if not relative_marker:
        require(is_directory and name == prefix, f"empty relative ZIP member path: {name!r}")
        require(info.file_size == 0 and info.compress_size == 0, "root ZIP directory contains data")
        return None, True
    if is_directory:
        require(relative_marker.endswith("/"), f"directory lacks a trailing slash: {name!r}")
        require(info.file_size == 0 and info.compress_size == 0, f"ZIP directory contains data: {name!r}")
        relative = relative_marker[:-1]
    else:
        require(not relative_marker.endswith("/"), f"regular ZIP member has a trailing slash: {name!r}")
        relative = relative_marker
    components = relative.split("/")
    require(relative and all(component not in ("", ".", "..") for component in components), "unsafe ZIP path")
    canonical = prefix + "/".join(components) + ("/" if is_directory else "")
    require(name == canonical and relative not in seen_relative, f"aliased/duplicate ZIP member: {name!r}")
    seen_relative.add(relative)
    return relative, is_directory


require(action in ("build", "verify"), "invalid dataset authentication action")
require(root.name == "task_ABC_D", f"unexpected dataset staging root: {root}")
expected_files = {}
expected_directories = set()
seen_names = set()
seen_relative = set()
inventory_digest = hashlib.sha256()
with zipfile.ZipFile(archive_path) as archive:
    infos = archive.infolist()
    for info in infos:
        relative, is_directory = validate_zip_info(info, seen_names, seen_relative)
        if relative is not None:
            components = relative.split("/")
            for depth in range(1, len(components)):
                expected_directories.add("/".join(components[:depth]))
            if is_directory:
                expected_directories.add(relative)
            else:
                expected_files[relative] = info
    require(not (set(expected_files) & expected_directories), "ZIP file/directory namespace collision")
    sorted_infos = sorted(infos, key=lambda item: item.filename)

require(expected_files, "ZIP has no dataset file members")
require(set(critical).issubset(expected_files), "ZIP is missing required CALVIN metadata")
root_descriptor = os.open(root, directory_flags)
try:
    require(stat.S_ISDIR(os.fstat(root_descriptor).st_mode), "dataset root is not a no-follow directory")
    exact_tree_inventory(root_descriptor, expected_files, expected_directories)

    manifest_raw = None
    manifest_raw_sha256 = None
    if action == "build":
        require(not os.path.lexists(index_path) and not os.path.lexists(manifest_path), "private output already exists")
        connection = sqlite3.connect(index_path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            "CREATE TABLE members(path TEXT PRIMARY KEY, bytes INTEGER NOT NULL, "
            "crc32 INTEGER NOT NULL, sha256 TEXT NOT NULL) WITHOUT ROWID"
        )
        connection.execute("CREATE TABLE metadata(name TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
        member_cursor = None
    else:
        manifest_raw = read_regular_file(manifest_path, maximum_bytes=16 * 1024 * 1024)
        manifest_raw_sha256 = hashlib.sha256(manifest_raw).hexdigest()
        uri = index_path.resolve().as_uri() + "?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        objects = list(
            connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            )
        )
        require(
            [(row[0], row[1], row[2]) for row in objects]
            == [("table", "members", "members"), ("table", "metadata", "metadata")]
            and all(isinstance(row[3], str) and "WITHOUT ROWID" in row[3].upper() for row in objects),
            "member-index object inventory differs",
        )
        require(
            connection.execute("PRAGMA table_info(members)").fetchall()
            == [
                (0, "path", "TEXT", 1, None, 1),
                (1, "bytes", "INTEGER", 1, None, 0),
                (2, "crc32", "INTEGER", 1, None, 0),
                (3, "sha256", "TEXT", 1, None, 0),
            ],
            "member-index members schema differs",
        )
        require(
            connection.execute("PRAGMA table_info(metadata)").fetchall()
            == [(0, "name", "TEXT", 1, None, 1), (1, "value", "TEXT", 1, None, 0)],
            "member-index metadata schema differs",
        )
        require(connection.execute("PRAGMA integrity_check").fetchone() == ("ok",), "member-index integrity failed")
        member_cursor = connection.execute("SELECT path, bytes, crc32, sha256 FROM members ORDER BY path")

    critical_hashes = {}
    compressed_bytes = 0
    member_uncompressed_bytes = 0
    npz_member_count = 0
    file_member_count = 0
    try:
        with zipfile.ZipFile(archive_path) as content_archive:
            for info in sorted_infos:
                record = {
                    "compress_size": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "file_size": info.file_size,
                    "is_dir": info.is_dir(),
                    "name": info.filename,
                }
                inventory_digest.update(canonical_json_bytes(record) + b"\n")
                if info.is_dir():
                    continue
                relative = info.filename[len(prefix) :]
                size, checksum, extracted_sha256 = read_extracted_member(
                    root_descriptor,
                    relative,
                    info,
                    durable=action == "build",
                )
                archived_digest = hashlib.sha256()
                archived_size = 0
                with content_archive.open(info, "r") as archived_member:
                    while True:
                        block = archived_member.read(8 * 1024 * 1024)
                        if not block:
                            break
                        archived_size += len(block)
                        archived_digest.update(block)
                archived_sha256 = archived_digest.hexdigest()
                require(archived_size == size, f"ZIP/extracted member byte lengths differ: {relative}")
                require(
                    archived_sha256 == extracted_sha256,
                    f"ZIP/extracted member SHA-256 differs despite CRC match: {relative}",
                )
                row = (relative, size, checksum, archived_sha256)
                if action == "build":
                    connection.execute(
                        "INSERT INTO members(path, bytes, crc32, sha256) VALUES (?, ?, ?, ?)",
                        row,
                    )
                else:
                    require(member_cursor.fetchone() == row, f"member-index row differs: {relative}")
                if relative in critical:
                    critical_hashes[relative] = archived_sha256
                file_member_count += 1
                compressed_bytes += info.compress_size
                member_uncompressed_bytes += size
                if relative.endswith(".npz"):
                    npz_member_count += 1

        require(set(critical_hashes) == set(critical), "required critical-file hashes are incomplete")
        require(member_uncompressed_bytes == reported_uncompressed_bytes, "ZIP uncompressed byte totals differ")
        if action == "build":
            connection.executemany(
                "INSERT INTO metadata(name, value) VALUES (?, ?)",
                (("file_member_count", str(file_member_count)), ("schema", index_schema)),
            )
            connection.commit()
            require(connection.execute("PRAGMA integrity_check").fetchone() == ("ok",), "member-index integrity failed")
        else:
            require(member_cursor.fetchone() is None, "member-index contains extra rows")
            metadata = dict(connection.execute("SELECT name, value FROM metadata"))
            require(
                metadata == {"file_member_count": str(file_member_count), "schema": index_schema},
                "member-index metadata differs",
            )
    finally:
        connection.close()

    if action == "build":
        index_identity = regular_file_identity(index_path, durable=True)
    else:
        index_identity = regular_file_identity(index_path)

    payload = {
        "archive": {
            "bytes": archive_bytes,
            "member_inventory": {
                "compressed_bytes": compressed_bytes,
                "file_member_count": file_member_count,
                "member_count": len(sorted_infos),
                "npz_member_count": npz_member_count,
                "sha256": inventory_digest.hexdigest(),
                "uncompressed_bytes": member_uncompressed_bytes,
            },
            "sha256": archive_sha256,
            "uncompressed_bytes": reported_uncompressed_bytes,
            "url": archive_url,
        },
        "checksum_url": checksum_url,
        "critical_files": critical_hashes,
        "dataset": "task_ABC_D",
        "extraction": {
            "file_members_verified": file_member_count,
            "member_index": {
                "bytes": index_identity["bytes"],
                "path": "task_ABC_D.members.sqlite3",
                "schema": index_schema,
                "sha256": index_identity["sha256"],
            },
            "verification": "size-and-crc32-against-every-pinned-zip-member",
        },
        "schema": "duo-vla-calvin-dataset-manifest-v3",
    }
    payload["content_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    if action == "build":
        manifest_bytes = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(
            manifest_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(manifest_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        fsync_tree_directories(root_descriptor)
        fsync_parent(index_path)
    else:
        try:
            observed_manifest = json.loads(
                manifest_raw.decode("utf-8"),
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("published manifest is not strict finite UTF-8 JSON") from exc
        require(observed_manifest == payload, "published manifest differs from the fully verified dataset")
        require(
            hashlib.sha256(read_regular_file(manifest_path, maximum_bytes=16 * 1024 * 1024)).hexdigest()
            == manifest_raw_sha256,
            "published manifest changed during verification",
        )
        require(regular_file_identity(index_path) == index_identity, "member index changed during verification")

    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
finally:
    os.close(root_descriptor)
PY
}

clear_empty_or_reject_stale_artifacts() {
  local candidate
  local stale_stages=()
  local stale_files=()
  shopt -s nullglob
  stale_stages=("${data_root}"/.task_ABC_D.extracting.*)
  stale_files=(
    "${data_root}"/.task_ABC_D.members.sqlite3.tmp-*
    "${data_root}"/.task_ABC_D.manifest.json.tmp-*
  )
  shopt -u nullglob
  for candidate in "${stale_stages[@]}"; do
    if [[ -d "${candidate}" && ! -L "${candidate}" ]] && rmdir -- "${candidate}" 2>/dev/null; then
      fsync_directory "${data_root}" || fail "cannot sync cleanup of empty staging directory"
    else
      fail "stale non-empty extraction staging tree requires manual recovery: ${candidate}"
    fi
  done
  ((${#stale_files[@]} == 0)) \
    || fail "stale extraction artifact requires manual recovery: ${stale_files[0]}"
}

publish_noreplace() {
  local source="$1"
  local destination="$2"
  [[ -e "${source}" || -L "${source}" ]] || fail "private publication source is missing: ${source}"
  [[ ! -e "${destination}" && ! -L "${destination}" ]] \
    || fail "refusing to replace existing publication target: ${destination}"
  mv --no-clobber --no-target-directory -- "${source}" "${destination}"
  [[ ! -e "${source}" && ! -L "${source}" ]] \
    || fail "publication target appeared concurrently: ${destination}"
  [[ -e "${destination}" || -L "${destination}" ]] || fail "publication rename did not create ${destination}"
  fsync_directory "${data_root}" || fail "cannot durably sync publication of ${destination}"
}

verify_or_reject_existing_publication() {
  local root_present=0
  local index_present=0
  local manifest_present=0
  if [[ -e "${final_root}" || -L "${final_root}" ]]; then
    root_present=1
  fi
  if [[ -e "${member_index_path}" || -L "${member_index_path}" ]]; then
    index_present=1
  fi
  if [[ -e "${manifest_path}" || -L "${manifest_path}" ]]; then
    manifest_present=1
  fi

  if ((root_present)); then
    [[ -d "${final_root}" && ! -L "${final_root}" ]] \
      || fail "published dataset root is not a no-follow directory: ${final_root}"
    ((index_present && manifest_present)) \
      || fail "dataset root exists without the index and manifest commit marker; refusing in-place blessing"
    [[ -f "${member_index_path}" && ! -L "${member_index_path}" ]] \
      || fail "published member index is not a no-follow regular file"
    [[ -f "${manifest_path}" && ! -L "${manifest_path}" ]] \
      || fail "published manifest is not a no-follow regular file"
    authenticate_dataset_tree verify "${final_root}" "${member_index_path}" "${manifest_path}" \
      || fail "existing dataset failed full archive/member/index/manifest verification"
    require_verified_archive_unchanged
    printf 'Dataset already extracted and fully verified: %s\n' "${final_root}"
    return 0
  fi

  ((!index_present && !manifest_present)) \
    || fail "index/manifest exists without the dataset root; refusing inconsistent publication state"
  return 1
}

extract_archive() {
  local expanded available required stage_root private_index private_manifest
  require_verified_archive_unchanged
  clear_empty_or_reject_stale_artifacts
  validate_member_paths
  if verify_or_reject_existing_publication; then
    return
  fi
  expanded="$(uncompressed_bytes)"
  [[ "${expanded}" =~ ^[0-9]+$ ]] || fail "cannot determine ZIP uncompressed byte total"
  available="$(available_bytes)"
  required=$((expanded + reserve_bytes))
  (( available >= required )) || fail "insufficient extraction headroom: available=${available}, required=${required}"
  stage_root="$(mktemp -d "${data_root}/.task_ABC_D.extracting.XXXXXX")"
  active_stage_root="${stage_root}"
  unzip -q "${archive_path}" -d "${stage_root}"
  private_index="${stage_root}/task_ABC_D.members.sqlite3"
  private_manifest="${stage_root}/task_ABC_D.manifest.json"
  authenticate_dataset_tree build "${stage_root}/task_ABC_D" "${private_index}" "${private_manifest}" \
    || fail "private extracted tree failed full archive-member authentication"
  require_verified_archive_unchanged

  # The root becomes visible first, then its authenticated index.  The manifest
  # is the last atomic commit marker.  A crash before it is durable leaves a
  # state that the next invocation refuses rather than blessing in place.
  publish_noreplace "${stage_root}/task_ABC_D" "${final_root}"
  publish_noreplace "${private_index}" "${member_index_path}"
  publish_noreplace "${private_manifest}" "${manifest_path}"
  rmdir -- "${stage_root}"
  active_stage_root=""
  fsync_directory "${data_root}" || fail "cannot durably sync staging cleanup"
}

print_contract() {
  printf '{"archive":"%s","archive_direct_reserve_bytes":%s,"bytes":%s,"reserve_bytes":%s,"sha256":"%s","url":"%s"}\n' \
    "${archive_name}" "${archive_direct_reserve_bytes}" "${expected_archive_bytes}" "${reserve_bytes}" \
    "${expected_archive_sha256}" "${archive_url}"
}

mkdir -p "${data_root}"
exec 9>"${data_root}/.task_ABC_D.download.lock"
flock -n 9 || fail "another task_ABC_D setup process holds the dataset lock"

case "${mode}" in
  all)
    download_archive
    verify_archive
    extract_archive
    ;;
  download)
    download_archive
    verify_archive
    ;;
  archive-direct)
    download_archive archive-direct
    run_archive_direct_tool prepare
    ;;
  prepare-archive-direct)
    require_headroom_for_download archive-direct
    run_archive_direct_tool prepare
    ;;
  verify-archive-direct)
    run_archive_direct_tool verify
    ;;
  verify)
    verify_remote_contract
    verify_archive
    ;;
  extract)
    verify_archive
    extract_archive
    ;;
  --print-contract)
    print_contract
    ;;
  *)
    fail "usage: $0 [all|download|verify|extract|archive-direct|prepare-archive-direct|verify-archive-direct|--print-contract]"
    ;;
esac
