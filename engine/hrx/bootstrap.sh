#!/usr/bin/env bash
# HRX / Loom for this engine: the pinned upstream revision (engine/hrx/PIN) plus the patches in engine/hrx/patches.
#
#   engine/hrx/bootstrap.sh            clone (if needed), check out the pin, apply the patches, build what the engine uses
#   engine/hrx/bootstrap.sh --check    verify the checkout: pinned revision, every patch applied, no other edits
#
# The checkout is $YAH_HRX (default: external/hrx in this repo, gitignored; a symlink to an existing checkout works).
# The build tree is $YAH_HRX/build/cmake. Building HRX from scratch takes a while, so the engine's own build never does
# it behind your back: run this script, then engine/build_hrx.sh.
# Patches apply idempotently: a patch already in the tree (or taken upstream) is left alone, and one that no longer fits
# the base stops the script. Each patch says at its top what it is for.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
hrx="${YAH_HRX:-$root/external/hrx}"
repo="$(awk '$1 == "repo" { print $2 }' "$here/PIN")"
ref="$(awk '$1 == "ref" { print $2 }' "$here/PIN")"
patches=("$here"/patches/*.patch)

check() {
  local ok=0 p files
  if [ ! -d "$hrx/.git" ] && [ ! -f "$hrx/.git" ]; then
    echo "hrx: no checkout at $hrx (run engine/hrx/bootstrap.sh)"
    return 1
  fi
  if [ "$(git -C "$hrx" rev-parse HEAD)" != "$ref" ]; then
    echo "hrx: $hrx is at $(git -C "$hrx" rev-parse --short HEAD), the pin is ${ref:0:9}"
    ok=1
  fi
  for p in "${patches[@]}"; do
    if ! git -C "$hrx" apply --check --reverse "$p" 2>/dev/null; then
      echo "hrx: not applied: $(basename "$p")"
      ok=1
    fi
  done
  # Edits outside the patches make a build nobody else can reproduce.
  files="$(cat "${patches[@]}" | sed -n 's|^+++ b/||p' | sort -u)"
  local extra
  extra="$(git -C "$hrx" status --porcelain | awk '{ print $2 }' | sort -u | comm -23 - <(echo "$files"))"
  if [ -n "$extra" ]; then
    echo "hrx: changes outside engine/hrx/patches:"
    echo "$extra" | sed 's/^/  /'
    ok=1
  fi
  [ "$ok" = 0 ] && echo "hrx: $hrx at ${ref:0:9} with ${#patches[@]} patches"
  return "$ok"
}

if [ "${1:-}" = "--check" ]; then
  check
  exit $?
fi

if [ ! -e "$hrx" ]; then
  echo "hrx: cloning $repo into $hrx"
  git clone "$repo" "$hrx"
fi
if [ "$(git -C "$hrx" rev-parse HEAD)" != "$ref" ]; then
  if [ -n "$(git -C "$hrx" status --porcelain)" ]; then
    echo "hrx: $hrx has local changes and is not at the pin; commit, stash or move them first" >&2
    exit 1
  fi
  git -C "$hrx" rev-parse --verify --quiet "$ref^{commit}" >/dev/null || git -C "$hrx" fetch --quiet origin
  git -C "$hrx" checkout --quiet --detach "$ref"
fi
for p in "${patches[@]}"; do
  if git -C "$hrx" apply --check --reverse "$p" 2>/dev/null; then
    echo "hrx: already applied: $(basename "$p")"
  elif git -C "$hrx" apply "$p"; then
    echo "hrx: applied: $(basename "$p")"
  else
    echo "hrx: $(basename "$p") does not apply to ${ref:0:9}; refresh it against the pin" >&2
    exit 1
  fi
done

# ROCm's clang builds the AMDGPU device code; the same toolchain builds the host side. Dependencies, including the HSA
# headers, are HRX's pinned copies: a system ROCm's headers can be older than the runtime HRX was written against.
rocm="${ROCM_PATH:-/opt/rocm}"
cc="${CC:-$rocm/llvm/bin/clang}"
cxx="${CXX:-$rocm/llvm/bin/clang++}"
build="$hrx/build/cmake"
if [ ! -f "$build/CMakeCache.txt" ]; then
  cmake -S "$hrx" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" -DIREE_ROCM_PATH="$rocm" \
    -DIREE_DEPENDENCY_MODE=pinned -DIREE_ROCM_DEPENDENCY_MODE=pinned \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_HAL_DRIVER_TASK=ON \
    -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF -DIREE_ENABLE_LIBBACKTRACE=OFF \
    -DLIBHRX_BUILD=ON -DLIBHRX_BUILD_HIP_BINDING=ON -DLIBHRX_BUILD_CTS=OFF \
    -DLOOM_BUILD=ON -DLOOM_TARGET_AMDGPU=ON -DLOOM_TARGET_AMDGPU_TARGETS=loom_defaults \
    -DAMDF_BUILD=ON -DAMDF_FAMILY_XDNA=ON -DLOOM_TARGET_XDNA=ON -DLOOM_TARGET_ARCH_XDNA=ON -DLOOM_EMIT_XDNA=ON
fi
# What the engine uses: the runtime, its HIP binding, the HAL emitter, the compiler (compile reports), the profiler,
# and the NPU runner (tools/npu_gemm_check.py).
ninja -C "$build" libhrx/src/libhrx/libhrx.so libhrx/src/binding/hip/libamdhip64.so \
  loom/src/loom/tools/iree-run-loom/iree-run-loom loom/src/loom/tools/loom-compile/loom-compile \
  runtime/src/iree/tools/iree-profile/iree-profile experimental/xdna/iree-xdna-run
check
