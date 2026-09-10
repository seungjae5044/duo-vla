#!/bin/bash -p
set -euo pipefail

readonly requested_cache_root="${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}"
readonly requested_hf_home="${HF_HOME:-/root/.cache/huggingface}"
export PATH="/usr/bin:/bin"
unset BASH_ENV CDPATH ENV GLOBIGNORE LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONINSPECT PYTHONPATH PYTHONSTARTUP

readonly project_dir="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd -P)"
readonly cache_root="$(realpath -m -- "${requested_cache_root}")"
readonly hf_home="$(realpath -m -- "${requested_hf_home}")"
readonly environment_path="${cache_root}/venvs/train-single-gpu"
readonly config_path="${project_dir}/configs/libero_dp2_fused_v2_b32.toml"
readonly expected_gpu0_uuid="GPU-30424b03-3051-615a-832e-186511378a61"
readonly expected_gpu1_uuid="GPU-84fa4004-92fb-8f86-cc65-01d62a27950e"

if [[ ! -x "${environment_path}/bin/python" ]]; then
  echo "DP2 requires the pinned train-single-gpu environment; run ./scripts/bootstrap_train_single_gpu_env.sh" >&2
  exit 1
fi

readonly train_seed="0"
fork_from=""
topology_fork_from=""
resume_from=""
expected_fork_manifest_sha256=""
expected_topology_fork_manifest_sha256=""
fork_count=0
topology_fork_count=0
resume_count=0
expected_fork_manifest_sha256_count=0
expected_topology_fork_manifest_sha256_count=0
help_only="false"
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  argument="${arguments[index]}"
  case "${argument}" in
    --help|-h)
      help_only="true"
      ;;
    --fork-from)
      ((index + 1 < ${#arguments[@]})) || {
        echo "--fork-from requires the absolute frozen fork-manifest JSON path" >&2
        exit 2
      }
      fork_from="${arguments[index + 1]}"
      ((fork_count += 1))
      ((index += 1))
      ;;
    --fork-from=*)
      fork_from="${argument#--fork-from=}"
      ((fork_count += 1))
      ;;
    --expected-fork-manifest-sha256)
      ((index + 1 < ${#arguments[@]})) || {
        echo "--expected-fork-manifest-sha256 requires the preregistered 64-hex digest" >&2
        exit 2
      }
      expected_fork_manifest_sha256="${arguments[index + 1]}"
      ((expected_fork_manifest_sha256_count += 1))
      ((index += 1))
      ;;
    --expected-fork-manifest-sha256=*)
      expected_fork_manifest_sha256="${argument#--expected-fork-manifest-sha256=}"
      ((expected_fork_manifest_sha256_count += 1))
      ;;
    --topology-fork-from)
      ((index + 1 < ${#arguments[@]})) || {
        echo "--topology-fork-from requires the absolute frozen topology-fork JSON path" >&2
        exit 2
      }
      topology_fork_from="${arguments[index + 1]}"
      ((topology_fork_count += 1))
      ((index += 1))
      ;;
    --topology-fork-from=*)
      topology_fork_from="${argument#--topology-fork-from=}"
      ((topology_fork_count += 1))
      ;;
    --expected-topology-fork-manifest-sha256)
      ((index + 1 < ${#arguments[@]})) || {
        echo "--expected-topology-fork-manifest-sha256 requires a preregistered 64-hex digest" >&2
        exit 2
      }
      expected_topology_fork_manifest_sha256="${arguments[index + 1]}"
      ((expected_topology_fork_manifest_sha256_count += 1))
      ((index += 1))
      ;;
    --expected-topology-fork-manifest-sha256=*)
      expected_topology_fork_manifest_sha256="${argument#--expected-topology-fork-manifest-sha256=}"
      ((expected_topology_fork_manifest_sha256_count += 1))
      ;;
    --resume)
      ((index + 1 < ${#arguments[@]})) || {
        echo "--resume requires a child DP2 checkpoint path" >&2
        exit 2
      }
      resume_from="${arguments[index + 1]}"
      ((resume_count += 1))
      ((index += 1))
      ;;
    --resume=*)
      resume_from="${argument#--resume=}"
      ((resume_count += 1))
      ;;
    --config|--config=*|--seed|--seed=*|--task|--task=*|--total-updates|--total-updates=*|--warmup-updates|--warmup-updates=*|--microbatch-size|--microbatch-size=*|--gradient-accumulation-steps|--gradient-accumulation-steps=*|--validation-interval|--validation-interval=*|--validation-samples|--validation-samples=*|--checkpoint-interval|--checkpoint-interval=*|--permanent-checkpoint-interval|--permanent-checkpoint-interval=*|--log-interval|--log-interval=*|--max-cached-files|--max-cached-files=*)
      echo "DP2 launcher seals seed0, config, rank B32, global B64, accumulation=1, and cache=377; ${argument} is forbidden" >&2
      exit 2
      ;;
  esac
done

if [[ "${help_only}" != "true" ]]; then
  if ((fork_count > 1 || topology_fork_count > 1 || resume_count > 1)); then
    echo "DP2 restore mode flags may each appear at most once" >&2
    exit 2
  fi
  if ((fork_count + topology_fork_count + resume_count > 1)); then
    echo "--fork-from, --topology-fork-from, and --resume are mutually exclusive" >&2
    exit 2
  fi
  if ((fork_count + topology_fork_count + resume_count == 0)); then
    echo "DP2 requires exactly one of --fork-from FROZEN_FORK_MANIFEST.json, --topology-fork-from FROZEN_TOPOLOGY_FORK.json, or --resume CHILD_CHECKPOINT" >&2
    exit 2
  fi
  if [[ -n "${fork_from}" ]]; then
    if ((expected_fork_manifest_sha256_count != 1)) \
      || [[ ! "${expected_fork_manifest_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "fork mode requires exactly one preregistered --expected-fork-manifest-sha256 64-hex digest" >&2
      exit 2
    fi
    if [[ "${fork_from}" != /* || ! -f "${fork_from}" || -L "${fork_from}" || ! -f "${fork_from}.sha256" || -L "${fork_from}.sha256" ]]; then
      echo "--fork-from must name an absolute regular non-symlink fork manifest with adjacent .sha256" >&2
      exit 2
    fi
    fork_manifest_name="${fork_from##*/}"
    observed_sidecar="$(/usr/bin/cat -- "${fork_from}.sha256")"
    expected_sidecar="${expected_fork_manifest_sha256}  ${fork_manifest_name}"
    observed_sidecar_bytes="$(/usr/bin/stat --format=%s -- "${fork_from}.sha256")"
    expected_sidecar_bytes="$((64 + 2 + ${#fork_manifest_name} + 1))"
    observed_manifest_sha256="$(/usr/bin/sha256sum -- "${fork_from}")"
    observed_manifest_sha256="${observed_manifest_sha256%% *}"
    if [[ "${observed_sidecar}" != "${expected_sidecar}" \
      || "${observed_sidecar_bytes}" != "${expected_sidecar_bytes}" \
      || "${observed_manifest_sha256}" != "${expected_fork_manifest_sha256}" ]]; then
      echo "fork manifest JSON or sidecar differs from the preregistered SHA-256" >&2
      exit 2
    fi
  elif ((expected_fork_manifest_sha256_count != 0)); then
    echo "--expected-fork-manifest-sha256 is forbidden in resume mode or topology fork mode" >&2
    exit 2
  fi
  if [[ -n "${topology_fork_from}" ]]; then
    if ((expected_topology_fork_manifest_sha256_count != 1)) \
      || [[ ! "${expected_topology_fork_manifest_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "topology fork mode requires exactly one preregistered --expected-topology-fork-manifest-sha256" >&2
      exit 2
    fi
    if [[ "${topology_fork_from}" != /* || ! -f "${topology_fork_from}" || -L "${topology_fork_from}" || ! -f "${topology_fork_from}.sha256" || -L "${topology_fork_from}.sha256" ]]; then
      echo "--topology-fork-from must name an absolute regular non-symlink manifest with adjacent .sha256" >&2
      exit 2
    fi
    topology_manifest_name="${topology_fork_from##*/}"
    topology_observed_sidecar="$(/usr/bin/cat -- "${topology_fork_from}.sha256")"
    topology_expected_sidecar="${expected_topology_fork_manifest_sha256}  ${topology_manifest_name}"
    topology_observed_sha256="$(/usr/bin/sha256sum -- "${topology_fork_from}")"
    topology_observed_sha256="${topology_observed_sha256%% *}"
    if [[ "${topology_observed_sidecar}" != "${topology_expected_sidecar}" \
      || "${topology_observed_sha256}" != "${expected_topology_fork_manifest_sha256}" ]]; then
      echo "topology fork manifest JSON or sidecar differs from the preregistered SHA-256" >&2
      exit 2
    fi
  elif ((expected_topology_fork_manifest_sha256_count != 0)); then
    echo "--expected-topology-fork-manifest-sha256 is forbidden outside topology fork mode" >&2
    exit 2
  fi

  mapfile -t observed_gpus < <(
    /usr/bin/nvidia-smi --id=0,1 --query-gpu=index,uuid --format=csv,noheader,nounits \
      | /usr/bin/sed 's/[[:space:]]//g'
  )
  if [[ "${#observed_gpus[@]}" -ne 2 \
    || "${observed_gpus[0]}" != "0,${expected_gpu0_uuid}" \
    || "${observed_gpus[1]}" != "1,${expected_gpu1_uuid}" ]]; then
    echo "physical GPUs 0,1 do not match the sealed DP2 UUID inventory" >&2
    exit 1
  fi
  observed_compute_apps="$(
    /usr/bin/nvidia-smi --id=0,1 --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits
  )" || {
    echo "cannot attest compute-process occupancy for physical GPUs 0,1" >&2
    exit 1
  }
  if [[ -n "${observed_compute_apps//[[:space:]]/}" ]]; then
    echo "physical GPUs 0,1 are not idle; refusing to preempt existing compute processes" >&2
    echo "${observed_compute_apps}" >&2
    exit 1
  fi
fi

canonical_environment=(
  "BLIS_NUM_THREADS=1"
  "CUBLAS_WORKSPACE_CONFIG=:4096:8"
  "CUDA_DEVICE_ORDER=PCI_BUS_ID"
  "CUDA_VISIBLE_DEVICES=0,1"
  "DUO_VLA_CACHE_ROOT=${cache_root}"
  "DUO_VLA_PROJECT_ROOT=${project_dir}"
  "DUO_VLA_TRAIN_VENV=${environment_path}"
  "HF_HOME=${hf_home}"
  "HF_HUB_DISABLE_PROGRESS_BARS=1"
  "HF_HUB_OFFLINE=1"
  "HOME=/root"
  "LANG=C.UTF-8"
  "LC_ALL=C.UTF-8"
  "MKL_NUM_THREADS=1"
  "NUMEXPR_NUM_THREADS=1"
  "OMP_DYNAMIC=FALSE"
  "OMP_NUM_THREADS=1"
  "OPENBLAS_NUM_THREADS=1"
  "PATH=/usr/bin:/bin"
  "PYTHONHASHSEED=${train_seed}"
  "PYTHONNOUSERSITE=1"
  "PYTHONPYCACHEPREFIX=/dev/null"
  "PYTHONSAFEPATH=1"
  "PYTHONDONTWRITEBYTECODE=1"
  "RAYON_NUM_THREADS=1"
  "TOKENIZERS_PARALLELISM=false"
  "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
  "TRANSFORMERS_OFFLINE=1"
  "TZ=UTC"
  "VECLIB_MAXIMUM_THREADS=1"
)

cd "${project_dir}"
if [[ "${help_only}" == "true" ]]; then
  exec /usr/bin/env -i "${canonical_environment[@]}" \
    "${environment_path}/bin/python" -P -B -X pycache_prefix=/dev/null \
    "${project_dir}/scripts/train_libero.py" --config "${config_path}" "$@"
fi

exec /usr/bin/env -i "${canonical_environment[@]}" \
  "${environment_path}/bin/python" -P -B -X pycache_prefix=/dev/null \
  -m torch.distributed.run \
  --standalone \
  --nproc-per-node=2 \
  "${project_dir}/scripts/train_libero.py" \
  --config "${config_path}" \
  --max-cached-files 377 \
  "$@"
