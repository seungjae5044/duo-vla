# Pinned LIBERO simulator setup

The simulator is deliberately separate from the Duo-VLA model environment. It uses the Hugging Face-maintained
`hf-libero` 0.1.4 release (source commit `8561c60eea2fb93096146f240194649df73d8b1e`), robosuite 1.4.0, MuJoCo
3.8.1, and NVIDIA EGL. The environment lock is in `envs/libero-eval/uv.lock`. The bootstrap sets CMake's documented
`CMAKE_POLICY_VERSION_MINIMUM=3.5` compatibility floor while building the legacy `egl-probe` dependency pulled by
robomimic; it does not patch the upstream source.

Simulator meshes and textures come from `lerobot/libero-assets` at immutable revision
`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`. These are runtime assets, not demonstrations. The bootstrap never calls
LIBERO's dataset downloader and does not download HDF5, LeRobot, RLDS, or parquet training data.

Run from the repository root:

```bash
export DUO_VLA_CACHE_ROOT=/root/.cache/duo-vla
export HF_HOME=/root/.cache/huggingface
export MUJOCO_EGL_DEVICE_ID=0
./scripts/bootstrap_libero_env.sh
```

The command is safe to rerun. It refuses to replace an unexpected source checkout, asset link, or runtime path. It
creates the Python environment at `/root/.cache/duo-vla/venvs/libero-eval`, the pinned source and assets at
`/root/.cache/duo-vla/simulators/libero`, and an empty simulator dataset path under `/root/.cache/duo-vla/data/libero`.
The final preflight imports LIBERO, verifies 40 tasks and all 50 fixed states per task, selects `mujoco.egl`, constructs
one 256x256 two-camera OSC_POSE environment at 20 Hz, checks a deterministic fixed-state reset, and takes one open-
gripper no-op step. It does not run a policy, a benchmark episode, or training.

Rerun the read-only validation with:

```bash
./scripts/run_libero_preflight.sh
```

Use `./scripts/run_libero_preflight.sh --imports-only` on a host where EGL construction is intentionally unavailable.
The imports-only mode is not an approval substitute for the full environment-construction preflight.

## Pre-register a clean-development state bank

Generate development reset states only through `run_generate_libero_dev_states_single_gpu.sh`. The launcher reuses the
pinned `libero-eval` environment, starts from `env -i`, removes loader and Python injection variables, and maps the
same selected physical device into `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID`. `DUO_VLA_PHYSICAL_GPU` is
fail-closed to `0` or `1` and defaults to `0`.

The Spatial gate pre-registration is one bank with this exact identity: `libero_spatial`, all task IDs `0-9`, two
states per task, base seed `20260829`, at most 200 attempts per task, and one predeclared physical GPU and output path.
This produces 20 clean-development states. Pre-register the complete tuple before execution; changing the GPU, path,
or any generator argument defines a different bank. The output path must not already exist, and the generator rejects
official-state matches, cross-bank duplicates, initially successful states, and states that become successful during
the mandatory settling trajectory.

The exact future GPU-1 invocation is:

```bash
export DUO_VLA_CACHE_ROOT=/hdd2/hyunbin/vla/cache
DUO_VLA_PHYSICAL_GPU=1 \
./scripts/run_generate_libero_dev_states_single_gpu.sh \
  "$DUO_VLA_CACHE_ROOT/evaluation/libero-spatial-clean-dev-seed-20260829-n2" \
  --suite libero_spatial \
  --task-ids all \
  --base-seed 20260829 \
  --states-per-task 2 \
  --max-attempts-per-task 200
```

Use `DUO_VLA_PHYSICAL_GPU=0` only if GPU 0 was the device named in the pre-registration. Preserve the emitted
`manifest_sha256`, `root_sha256`, and exclusive output directory as the frozen development-reset identity. Bank
generation constructs MuJoCo environments and therefore must not be launched on a GPU reserved by a training run.

For official pre-registration, preserve the pure full report with
`./scripts/run_libero_preflight.sh --output-json /read-only/freeze/libero-simulator-attestation.json`. The v2 report
content-addresses the evaluator lock and sources, clean Git tree, module origins and verified distribution RECORDs,
complete assets, every task BDDL/instruction, and all 2,000 official reset-state byte hashes. The official evaluator
regenerates this report and requires its canonical SHA-256 to equal the frozen pre-registration before policy contact.
The preflight parses the runtime manifest as duplicate-free exact-schema JSON, checks the source URL/revision/path and
clean live Git origin, verifies semantic config paths resolve BDDL/reset files from the pinned package, compares
the full installed name/version closure to a frozen identity, checks behavior-critical distribution RECORD inventories
and file contents, rejects unregistered startup hooks/site-packages files/symlinks, and compares the complete asset and
task/reset trees to frozen content hashes. The full report also records the OpenGL vendor/renderer/driver strings and
pixel SHA-256 values before and after the smoke step. An imports-only report cannot be used to create an official
pre-registration.

The full attestation is also an authenticated input to the separate
[40-task expert replay qualification](libero_expert_replay_qualification.md). That gate is stronger than the one-task
preflight smoke: it requires regenerated successful evidence for every canonical task plus alignment, camera,
controller, reset, and pre-dispatch integrity evidence. The pinned inventory and two-stage collector/binder now exist;
the 31.47-GiB original-HDF5 corpus is not locally materialized and a real 40-task report is still outstanding.

Every simulator subprocess must set `LIBERO_CONFIG_PATH`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and
`MUJOCO_EGL_DEVICE_ID` before importing MuJoCo, robosuite, or LIBERO. Keep policy inference in the training environment
and exchange copied raw RGB/state arrays and `float32[7]` actions over the eventual lossless local IPC boundary, as
required by `docs/environment.md` and `docs/libero_protocol.md`.
