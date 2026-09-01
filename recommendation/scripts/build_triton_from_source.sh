#!/usr/bin/env bash
# Build Triton from upstream source at the revision sglang's gfx1250-rocm7_14
# flavor uses for its MI450 A0 bring-up (docker/rocm.Dockerfile TRITON_COMMIT_DEFAULT).
set -euxo pipefail

TRITON_COMMIT="${TRITON_COMMIT:-76940ad348795521b3dc9f6c79acd7309ff924e3}"
SRC=/opt/triton-custom

export MAX_JOBS="${MAX_JOBS:-64}"
export PIP_NO_CACHE_DIR=1

if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/triton-lang/triton.git "$SRC"
fi
cd "$SRC"
git fetch --all --tags
git checkout "$TRITON_COMMIT"
git log -1 --oneline

pip install -r python/requirements.txt
pip uninstall -y triton || true
pip install -e .
if [ -d python/triton_kernels ]; then
  pip install -e python/triton_kernels --no-deps || true
fi

python -c "import triton; print('BUILD OK', triton.__version__, triton.__file__)"
echo "BUILD_SCRIPT_DONE"
