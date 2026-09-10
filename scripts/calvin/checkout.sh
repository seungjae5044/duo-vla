#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=revisions.env
source "$script_dir/revisions.env"

cache_root=${DUO_VLA_CACHE_ROOT:-/root/.cache/duo-vla}
source_root=${CALVIN_SOURCE_ROOT:-$cache_root/simulators/calvin}

if [[ -e "$source_root" && ! -d "$source_root/.git" ]]; then
  echo "Refusing to replace non-git path: $source_root" >&2
  exit 1
fi

if [[ ! -d "$source_root/.git" ]]; then
  mkdir -p "$(dirname -- "$source_root")"
  git clone --filter=blob:none "$CALVIN_REPOSITORY_URL" "$source_root"
fi

actual_remote=$(git -C "$source_root" remote get-url origin)
case "$actual_remote" in
  "$CALVIN_REPOSITORY_URL"|https://github.com/mees/calvin|git@github.com:mees/calvin.git) ;;
  *)
    echo "Unexpected CALVIN origin: $actual_remote" >&2
    exit 1
    ;;
esac

if [[ -n "$(git -C "$source_root" status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing to change a CALVIN checkout with tracked modifications: $source_root" >&2
  exit 1
fi

git -C "$source_root" fetch --no-tags origin "$CALVIN_REVISION"
git -C "$source_root" checkout --detach "$CALVIN_REVISION"
git -C "$source_root" submodule sync --recursive
git -C "$source_root" submodule update --init --recursive

actual_parent=$(git -C "$source_root" rev-parse HEAD)
actual_env=$(git -C "$source_root/calvin_env" rev-parse HEAD)
actual_tacto=$(git -C "$source_root/calvin_env/tacto" rev-parse HEAD)
[[ "$actual_parent" == "$CALVIN_REVISION" ]]
[[ "$actual_env" == "$CALVIN_ENV_REVISION" ]]
[[ "$actual_tacto" == "$CALVIN_TACTO_REVISION" ]]

printf 'CALVIN source ready at %s\n' "$source_root"
printf '  parent: %s\n  calvin_env: %s\n  tacto: %s\n' "$actual_parent" "$actual_env" "$actual_tacto"
