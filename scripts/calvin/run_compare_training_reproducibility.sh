#!/usr/bin/env -S -i DUO_VLA_CLOSED_LAUNCHER_ENTRY=1 /bin/bash --noprofile --norc
set -euo pipefail

if [[ "${DUO_VLA_CLOSED_LAUNCHER_ENTRY:-}" != "1" ]]; then
  echo "Invoke this qualification launcher directly; do not run it through bash" >&2
  exit 1
fi
unset DUO_VLA_CLOSED_LAUNCHER_ENTRY

readonly launcher_path="$(/usr/bin/readlink -f -- "${BASH_SOURCE[0]}")"
readonly script_dir="$(cd -- "$(/usr/bin/dirname -- "${launcher_path}")" && pwd)"
readonly project_dir="$(cd -- "${script_dir}/../.." && pwd)"
readonly cache_root="/hdd2/hyunbin/vla/cache"
readonly environment_path="${cache_root}/venvs/train"
readonly comparator_path="${project_dir}/scripts/compare_calvin_training_reproducibility.py"

hash_file() {
  local output
  output="$(/usr/bin/sha256sum -- "$1")"
  if [[ ! "${output%% *}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Cannot hash qualification source: $1" >&2
    exit 1
  fi
  printf '%s' "${output%% *}"
}

readonly launcher_start_sha256="$(hash_file "${launcher_path}")"

if [[ ! -x "${environment_path}/bin/python" ]]; then
  echo "Train environment is missing; run scripts/bootstrap_train_env.sh" >&2
  exit 1
fi

if [[ ! -f "${comparator_path}" || -L "${comparator_path}" ]]; then
  echo "Qualification comparator must be a real file: ${comparator_path}" >&2
  exit 1
fi

# The isolated, no-site bootstrap manually adds exactly the canonical venv
# package directory and project source directory. It reads and compiles one
# authenticated comparator byte string, injecting a process-local global
# capability that a direct ``python comparator.py`` invocation cannot receive.
readonly python_bootstrap='import hashlib as _hashlib
import os as _os
import pathlib as _pathlib
import stat as _stat
import sys as _sys

def _stable_regular_bytes(_path):
    _before = _os.stat(_path, follow_symlinks=False)
    if not _stat.S_ISREG(_before.st_mode):
        raise SystemExit(f"qualification bootstrap source must be a regular file: {_path}")
    _descriptor = _os.open(_path, _os.O_RDONLY | _os.O_NONBLOCK | _os.O_NOFOLLOW | _os.O_CLOEXEC)
    try:
        _opened = _os.fstat(_descriptor)
        if not _stat.S_ISREG(_opened.st_mode) or (_opened.st_dev, _opened.st_ino) != (_before.st_dev, _before.st_ino):
            raise SystemExit(f"qualification bootstrap source changed while opening: {_path}")
        _blocks = []
        while True:
            _block = _os.read(_descriptor, 1024 * 1024)
            if not _block:
                break
            _blocks.append(_block)
        _after_descriptor = _os.fstat(_descriptor)
    finally:
        _os.close(_descriptor)
    _after_path = _os.stat(_path, follow_symlinks=False)
    _fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    if any(getattr(_opened, _field) != getattr(_after_descriptor, _field) for _field in _fields) or any(getattr(_opened, _field) != getattr(_after_path, _field) for _field in _fields):
        raise SystemExit(f"qualification bootstrap source changed while reading: {_path}")
    _raw = b"".join(_blocks)
    if len(_raw) != _opened.st_size:
        raise SystemExit(f"qualification bootstrap source size changed while reading: {_path}")
    return _raw

_expected_venv_argument = _pathlib.Path(_sys.argv[1])
_expected_venv = _expected_venv_argument.resolve(strict=True)
if not _expected_venv_argument.is_absolute() or _expected_venv_argument != _expected_venv:
    raise SystemExit("qualification train environment path is not canonical")
if _sys.implementation.name != "cpython" or _sys.version_info[:3] != (3, 11, 15):
    raise SystemExit("qualification requires exact Python 3.11.15")
_expected_executable = _expected_venv / "bin/python"
if _pathlib.Path(_sys.executable) != _expected_executable or _sys.prefix != _sys.base_prefix:
    raise SystemExit("qualification did not start from the canonical no-site train Python entry point")
_flags = {"dont_write_bytecode": _sys.flags.dont_write_bytecode, "ignore_environment": _sys.flags.ignore_environment, "isolated": _sys.flags.isolated, "no_site": _sys.flags.no_site, "no_user_site": _sys.flags.no_user_site, "safe_path": _sys.flags.safe_path}
_expected_flags = {"dont_write_bytecode": 1, "ignore_environment": 1, "isolated": 1, "no_site": 1, "no_user_site": 1, "safe_path": True}
if _flags != _expected_flags or "site" in _sys.modules:
    raise SystemExit("qualification requires -I -S -B before bootstrap")
_forbidden = sorted(set(("_virtualenv", "_cuda_bindings_redirector", "_distutils_hack")) & set(_sys.modules))
if _forbidden:
    raise SystemExit(f"qualification bootstrap preloaded forbidden site hooks: {_forbidden}")
_pyvenv_path = _expected_venv / "pyvenv.cfg"
_pyvenv_raw = _stable_regular_bytes(_pyvenv_path)
_pyvenv = {}
for _line in _pyvenv_raw.decode("utf-8").splitlines():
    _key, _separator, _value = _line.partition("=")
    _key = _key.strip()
    _value = _value.strip()
    if not _separator or not _key or _key in _pyvenv:
        raise SystemExit("qualification pyvenv.cfg is malformed")
    _pyvenv[_key] = _value
if _pyvenv.get("implementation") != "CPython" or _pyvenv.get("version_info") != "3.11.15" or _pyvenv.get("include-system-site-packages") != "false":
    raise SystemExit("qualification pyvenv.cfg identity differs")
if _pathlib.Path(_pyvenv.get("home", "")).resolve(strict=True) != (_pathlib.Path(_sys.base_prefix) / "bin").resolve(strict=True):
    raise SystemExit("qualification pyvenv.cfg base interpreter differs")
_site_packages = _expected_venv / "lib/python3.11/site-packages"
_project_src = _pathlib.Path(_sys.argv[2]).resolve(strict=True)
if not _stat.S_ISDIR(_os.stat(_site_packages, follow_symlinks=False).st_mode) or not _stat.S_ISDIR(_os.stat(_project_src, follow_symlinks=False).st_mode):
    raise SystemExit("qualification import roots must be real directories")
if str(_site_packages) in _sys.path or str(_project_src) in _sys.path:
    raise SystemExit("qualification import roots were unexpectedly preloaded")
_sys.path.extend((str(_project_src), str(_site_packages)))
_launcher = _pathlib.Path(_sys.argv[3]).resolve(strict=True)
_launcher_start = _sys.argv[4]
_launcher_raw = _stable_regular_bytes(_launcher)
_launcher_sha256 = _hashlib.sha256(_launcher_raw).hexdigest()
if _launcher_sha256 != _launcher_start:
    raise SystemExit("qualification launcher changed after its start snapshot")
_comparator = _pathlib.Path(_sys.argv[5]).resolve(strict=True)
_comparator_raw = _stable_regular_bytes(_comparator)
_comparator_sha256 = _hashlib.sha256(_comparator_raw).hexdigest()
globals()["__duo_vla_qualification_bootstrap_capability__"] = {
    "comparator_sha256": _comparator_sha256,
    "expected_train_venv": str(_expected_venv),
    "forbidden_modules_preloaded": _forbidden,
    "interpreter_flags": _flags,
    "launcher_sha256": _launcher_sha256,
    "mode": "launcher-exact-comparator-bytes-isolated-v2",
    "project_src": str(_project_src),
    "pyvenv_cfg_sha256": _hashlib.sha256(_pyvenv_raw).hexdigest(),
    "site_packages": str(_site_packages),
    "sys_path": list(_sys.path),
}
_sys.argv = [str(_comparator), *_sys.argv[6:]]
globals()["__file__"] = str(_comparator)
globals()["__cached__"] = None
globals()["__package__"] = None
globals()["__spec__"] = None
exec(compile(_comparator_raw, str(_comparator), "exec"), globals(), globals())'

readonly launcher_pre_exec_sha256="$(hash_file "${launcher_path}")"
if [[ "${launcher_start_sha256}" != "${launcher_pre_exec_sha256}" ]]; then
  echo "Qualification launcher changed between startup and exec" >&2
  exit 1
fi

# The comparator only deserializes already-authenticated rank state onto CPU.
# A clean environment prevents model/dataset lookup and hides every GPU.
exec /usr/bin/env -i \
  CUDA_VISIBLE_DEVICES= \
  DUO_VLA_CACHE_ROOT="${cache_root}" \
  HF_HUB_OFFLINE=1 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PATH="${environment_path}/bin:/usr/bin:/bin" \
  PYTHONHASHSEED=0 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/dev/null \
  TRANSFORMERS_OFFLINE=1 \
  "${environment_path}/bin/python" \
  -I -S -B -c "${python_bootstrap}" \
  "${environment_path}" \
  "${project_dir}/src" \
  "${launcher_path}" \
  "${launcher_start_sha256}" \
  "${comparator_path}" \
  "$@"
