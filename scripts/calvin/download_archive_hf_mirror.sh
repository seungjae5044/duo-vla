#!/usr/bin/env bash
set -euo pipefail

readonly archive_name="task_ABC_D.zip"
readonly expected_archive_bytes=555309812705
readonly expected_archive_sha256="c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74"
readonly mirror_repo="myendless/calvin_abc_d"
readonly mirror_revision="cab96dbc6432795736dd718ec3b927a50bb6f924"
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
readonly data_root="${CALVIN_DATA_ROOT:-${cache_root}/data/calvin}"
readonly hf_home="${HF_HOME:-/root/.cache/huggingface}"
readonly train_venv="${DUO_VLA_TRAIN_VENV:-${cache_root}/venvs/train}"
readonly hf_cli="${CALVIN_HF_CLI:-${train_venv}/bin/hf}"
readonly archive_path="${data_root}/${archive_name}"
readonly aria2_control_path="${archive_path}.aria2"
readonly mirror_root="${data_root}/mirror-parts/${mirror_revision}"
readonly stage_path="${data_root}/.${archive_name}.mirror-assembling"

readonly -a part_sizes=(
  55530981271 55530981271 55530981271 55530981271 55530981271
  55530981270 55530981270 55530981270 55530981270 55530981270
)
readonly -a part_sha256=(
  39b71e5150d42ff70891f6899271a83b2a2fd50221ea870047e12844761ea2ae
  07242fd4114b8b7cfa313ab93d36aa4c0dca485319d6d7b2d50fd2dccee16863
  b4ca43cc19283b6aa796360ce65ed1666dfd6d569c70e14175826ce62724c460
  a3df61c4386f4b9bfc77568b2cac0142b64916f7090f218588bba3b19c60d6cd
  43e2029d01d209da1fff060b4f90ab3345e241014ff432b77e41ae155106ae38
  2e7b349b68bc4e841c1eb2fd46a50a39786482a8cf9a8daf87a66bc3a0d910b2
  7f9305ea73748042efa568ff3159af0363d63c6a8071f27a4528bc2978c03d5c
  4532ce8a7c1e1bb5976e1995f21e008f7f4a5a3533ad1665d9d22428c7f8e43e
  19939f47562a12c6762074940ffe2aec9235cef505310be81245db599308aa57
  60b72292a9b07e05ae566546165699eba7573431514a78dea4a1cbcca4f57425
)

fail() {
  printf 'CALVIN Hugging Face mirror setup failed: %s\n' "$*" >&2
  exit 1
}

mkdir -p "${data_root}" "${mirror_root}" "${hf_home}"
exec 9>"${data_root}/.task_ABC_D.download.lock"
flock -n 9 || fail "another task_ABC_D setup process holds the dataset lock"

read -r available_blocks filesystem_block_size < <(stat -f -c '%a %S' "${data_root}")
available_bytes=$((available_blocks * filesystem_block_size))
required_bytes=$((2 * expected_archive_bytes + 16 * 1024 * 1024 * 1024))
((available_bytes >= required_bytes)) \
  || fail "insufficient headroom for pinned parts plus assembled archive: available=${available_bytes}, required=${required_bytes}"

[[ -x "${hf_cli}" ]] || fail "Hugging Face CLI is missing: ${hf_cli}"
if [[ -e "${archive_path}" || -L "${archive_path}" ]]; then
  [[ -f "${archive_path}" && ! -L "${archive_path}" && "$(stat -c '%h' "${archive_path}")" == 1 ]] \
    || fail "archive target is not a no-follow, singly linked regular file"
fi
[[ ! -L "${aria2_control_path}" ]] || fail "aria2 control path is a symlink"

if [[ -f "${archive_path}" && "$(stat -c '%s' "${archive_path}")" == "${expected_archive_bytes}" \
  && ! -e "${aria2_control_path}" ]]; then
  printf 'A complete-size archive is already present; deferring full authentication to the canonical preparer.\n'
else
  download_names=()
  for index in "${!part_sizes[@]}"; do
    printf -v suffix '%02d' "${index}"
    name="${archive_name}.part${suffix}"
    part="${mirror_root}/${name}"
    if [[ -e "${part}" || -L "${part}" ]]; then
      [[ -f "${part}" && ! -L "${part}" && "$(stat -c '%h' "${part}")" == 1 ]] \
        || fail "existing mirror part is not a no-follow, singly linked regular file: ${part}"
      if [[ "$(stat -c '%s' "${part}")" == "${part_sizes[index]}" ]] \
        && printf '%s  %s\n' "${part_sha256[index]}" "${part}" | sha256sum --check --status; then
        printf 'Reusing authenticated mirror part: %s\n' "${name}"
        continue
      fi
      recovery_root="${data_root}/recovery_quarantine/replaced-mirror-part-$(date -u +%Y%m%dT%H%M%SZ)-$$"
      mkdir -p "${recovery_root}"
      mv -- "${part}" "${recovery_root}/"
    fi
    download_names+=("${name}")
  done

  if ((${#download_names[@]} > 0)); then
    printf 'Downloading %d missing mirror part(s): %s\n' "${#download_names[@]}" "${download_names[*]}"
  HF_HOME="${hf_home}" HF_XET_HIGH_PERFORMANCE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/dev/null \
      "${hf_cli}" download "${mirror_repo}" "${download_names[@]}" \
    --repo-type dataset \
    --revision "${mirror_revision}" \
    --local-dir "${mirror_root}"
  else
    printf 'All mirror parts are already authenticated; skipping network download.\n'
  fi

  parts=()
  total=0
  for index in "${!part_sizes[@]}"; do
    printf -v suffix '%02d' "${index}"
    part="${mirror_root}/${archive_name}.part${suffix}"
    [[ -f "${part}" && ! -L "${part}" && "$(stat -c '%h' "${part}")" == 1 ]] \
      || fail "mirror part is not a no-follow, singly linked regular file: ${part}"
    [[ "$(stat -c '%s' "${part}")" == "${part_sizes[index]}" ]] \
      || fail "mirror part byte length mismatch: ${part}"
    printf '%s  %s\n' "${part_sha256[index]}" "${part}" | sha256sum --check --status \
      || fail "mirror part SHA-256 mismatch: ${part}"
    parts+=("${part}")
    total=$((total + part_sizes[index]))
  done
  ((total == expected_archive_bytes)) || fail "mirror part byte total differs from the official archive"

  if [[ -e "${stage_path}" || -L "${stage_path}" ]]; then
    [[ -f "${stage_path}" && ! -L "${stage_path}" ]] || fail "mirror staging path is unsafe"
    recovery_root="${data_root}/recovery_quarantine/mirror-stage-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mkdir -p "${recovery_root}"
    mv -- "${stage_path}" "${recovery_root}/"
  fi

  "${CALVIN_ARCHIVE_DIRECT_PYTHON:-python3}" - \
    "${stage_path}" "${expected_archive_sha256}" "${parts[@]}" <<'PY'
import hashlib
import os
import stat
import sys

destination, expected_sha256, *parts = sys.argv[1:]
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
descriptor = os.open(destination, flags, 0o600)
digest = hashlib.sha256()
try:
    with os.fdopen(descriptor, "wb", closefd=False) as output:
        for part in parts:
            source_descriptor = os.open(
                part,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
            )
            try:
                if not stat.S_ISREG(os.fstat(source_descriptor).st_mode):
                    raise RuntimeError(f"mirror part is not regular: {part}")
                while block := os.read(source_descriptor, 16 * 1024 * 1024):
                    output.write(block)
                    digest.update(block)
            finally:
                os.close(source_descriptor)
        output.flush()
        os.fsync(descriptor)
finally:
    os.close(descriptor)
if digest.hexdigest() != expected_sha256:
    raise RuntimeError(f"assembled archive SHA-256 mismatch: {digest.hexdigest()}")
PY

  [[ "$(stat -c '%s' "${stage_path}")" == "${expected_archive_bytes}" ]] \
    || fail "assembled archive byte length mismatch"
  recovery_root="${data_root}/recovery_quarantine/replaced-partial-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  if [[ -e "${archive_path}" || -L "${archive_path}" || -e "${aria2_control_path}" ]]; then
    mkdir -p "${recovery_root}"
    [[ ! -e "${archive_path}" && ! -L "${archive_path}" ]] || mv -- "${archive_path}" "${recovery_root}/"
    [[ ! -e "${aria2_control_path}" && ! -L "${aria2_control_path}" ]] \
      || mv -- "${aria2_control_path}" "${recovery_root}/"
  fi
  mv -- "${stage_path}" "${archive_path}"
  python3 - "${data_root}" <<'PY'
import os
import sys

descriptor = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
fi

flock -u 9
exec 9>&-
exec env \
  DUO_VLA_CACHE_ROOT="${cache_root}" \
  CALVIN_DATA_ROOT="${data_root}" \
  CALVIN_ARCHIVE_DIRECT_PYTHON="${CALVIN_ARCHIVE_DIRECT_PYTHON:-python3}" \
  "${script_dir}/download_dataset.sh" archive-direct
