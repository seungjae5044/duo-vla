# Repository Guidelines

## Project Structure & Module Organization

Duo-VLA implements continuous robot-action policies on DiffusionGemma using PyTorch.

- `src/duo_vla/`: policies, objectives, training, checkpointing, and runtime contracts; `backbones/` contains adapters, `data/` dataset utilities, and `benchmarks/` LIBERO/CALVIN integrations.
- `tests/`: pytest unit, contract, and regression tests.
- `scripts/`: training, evaluation, download, and preflight tools; CALVIN utilities live in `scripts/calvin/`.
- `configs/`: TOML recipes and pinned artifact identities.
- `envs/`: separately locked training and simulator environments.
- `docs/`: architecture, setup, and protocols; `reports/qualifications/` holds reviewed evidence.

## Build, Test, and Development Commands

Use Python 3.11 and run from the repository root. Choose an external cache directory:

```bash
export DUO_VLA_CACHE_ROOT=/path/to/large/cache
export UV_PROJECT_ENVIRONMENT="$DUO_VLA_CACHE_ROOT/venvs/dev"
uv sync --frozen --extra dev --extra data
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
python -m pytest -q                    # Run tests.
python -m ruff check src scripts tests # Check style and imports.
uv build                              # Build wheel and source distributions.
python -m duo_vla.experiments.synthetic_overfit --device cpu --steps 20
```

The final command runs a synthetic training check. For GPU training, `./scripts/bootstrap_train_env.sh` installs locked dependencies and checks CUDA/BF16. Follow `docs/training.md` for artifact preparation and launchers, or `docs/single_gpu_g6.md` for single-GPU setup.

## Coding Style & Naming Conventions

Use four-space indentation, type hints, and concise docstrings. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Ruff targets Python 3.11 with a 120-character line limit and import-order checks; see `pyproject.toml`.

## Testing Guidelines

Name files `test_*.py` and functions `test_<behavior>`. Prefer deterministic fixtures, fixed seeds, and tiny backbones. Add regression tests for changed contracts and failure paths. Run focused checks with `python -m pytest tests/test_policy.py -q`. Optional-dependency and CUDA tests may skip; report skips. No numeric coverage threshold is configured.

## Commit & Pull Request Guidelines

History uses descriptive subjects such as `Isolate CALVIN evaluator JSON output`; no mandatory prefix convention is evident. Keep commits focused. PRs should explain the problem, resulting behavior, affected configs/protocols, and validation results; link relevant issues and evidence for performance claims.

## Artifacts & Benchmark Integrity

Keep credentials, weights, datasets, simulator assets, and run outputs outside Git. Preserve pinned revisions and lockfiles; isolate simulator environments. Never use CALVIN scene D for training, normalization, tuning, or checkpoint selection. Follow the compute-approval and official-evaluation gates in `docs/single_gpu_g6.md`.
