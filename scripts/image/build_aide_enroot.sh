#!/bin/bash
# Build the AIDE execution image with Enroot on an allocated NVIDIA GPU.
# Run from any directory. OUT_SQSH sets the output image path.
set -euxo pipefail

REPO="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Per-arch image paths. $HOME is shared NFS across beta (aarch64/B200) and alpha
# (x86_64/H100+H200), and an .sqsh only runs on the arch it was built on -- so
# the two images MUST NOT share a filename. This also guards the `[ -f
# $BASE_SQSH ]` cache check below: with a single default the x86_64 build would
# silently reuse the existing aarch64 ubuntu base and produce a broken image.
if [ "$(uname -m)" = "x86_64" ]; then
  BASE_SQSH="${BASE_SQSH:-$HOME/ubuntu2204.x86_64.sqsh}"
  BUILD_NAME="${BUILD_NAME:-aide-build-x86_64}"
  OUT_SQSH="${OUT_SQSH:-$HOME/aide.x86_64.sqsh}"
else
  BASE_SQSH="${BASE_SQSH:-$HOME/ubuntu2204.sqsh}"
  BUILD_NAME="${BUILD_NAME:-aide-build}"
  OUT_SQSH="${OUT_SQSH:-$HOME/aide.sqsh}"
fi
PIP_CACHE="${PIP_CACHE:-$HOME/.enroot_build_cache/pip}"  # persists wheels across retries
mkdir -p "$PIP_CACHE"

# 1. base image (cached after first import)
# ubuntu 22.04 (not the repo's 20.04): focal's gcc-9 lacks arm_sve.h and many
# arm64 wheels need newer glibc, forcing fragile source builds. Jammy's gcc-11 +
# glibc 2.35 fix both. The conda env still pins python=3.11, so this only changes
# the system toolchain, not the agent's Python stack.
[ -f "$BASE_SQSH" ] || enroot import -o "$BASE_SQSH" docker://ubuntu:22.04

# 2. fresh writable sandbox
enroot remove -f "$BUILD_NAME" 2>/dev/null || true
enroot create --name "$BUILD_NAME" "$BASE_SQSH"

# 3. replay the Dockerfiles (root, writable, repo + pip-cache mounted, GPU live)
# NVIDIA_DRIVER_CAPABILITIES=compute,utility: without `compute`, the enroot
# nvidia hook injects nvidia-smi/NVML but NOT libcuda.so, so torch.cuda is
# unavailable. Needed for the in-build GPU verify.
# GPU passthrough differs per cluster. Beta has libnvidia-container, so the
# enroot nvidia hook works off NVIDIA_VISIBLE_DEVICES. Alpha ships no
# libnvidia-container (and no pyxis), so that hook aborts `enroot start`
# outright -- stage the driver user-space into a private dir and bind it plus
# the device nodes, exactly as agents/runtime_enroot.py does at run time. This
# only matters for the inner script's final torch.cuda verify, but a build that
# skips that verify is how a silently GPU-less image ships.
GPU_ARGS=()
GPU_ENV=()
if command -v nvidia-container-cli >/dev/null 2>&1; then
  export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
else
  echo "[build] no libnvidia-container -> manual driver injection"
  NVDRV=/tmp/nvdrv-build-${SLURM_JOB_ID:-local}
  mkdir -p "$NVDRV"
  for d in /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib64; do
    [ -d "$d" ] || continue
    for g in 'libcuda.so*' 'libnvidia-ml.so*' 'libnvidia-nvvm.so*' 'libnvidia-ptxjitcompiler.so*'; do
      cp -a $d/$g "$NVDRV"/ 2>/dev/null || true
    done
  done
  cp -a "$(command -v nvidia-smi)" "$NVDRV"/ 2>/dev/null || true
  [ -e "$NVDRV/libcuda.so.1" ] || { echo "ERROR: could not stage libcuda.so.1" >&2; exit 1; }
  GPU_ARGS=(--mount "$NVDRV:/opt/nvdrv")
  for dev in /dev/nvidia*; do
    [ -e "$dev" ] && [ ! -d "$dev" ] && GPU_ARGS+=(--mount "$dev:$dev")
  done
  GPU_ENV=(--env "LD_LIBRARY_PATH=/opt/nvdrv" --env "PATH=/opt/nvdrv:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
fi

enroot start --root --rw \
    --mount "$REPO:/work" \
    --mount "$PIP_CACHE:/root/.cache/pip" \
    "${GPU_ARGS[@]}" "${GPU_ENV[@]}" \
    "$BUILD_NAME" bash /work/scripts/image/_build_aide_inner.sh

# 4. export the finished image -- to NODE-LOCAL disk first, never straight to
# $HOME. mksquashfs streaming the ~94 GB image directly onto NFS saturates the
# filer head and hangs logins cluster-wide (alpha1 was frozen 16:50-17:29 EDT
# on 2026-08-13 by exactly this line). Export locally, then trickle the image
# to $HOME at a rate the filer absorbs. EXPORT_DIR needs ~110 GiB free ON TOP
# of the unpacked rootfs already in ENROOT_DATA_PATH; the guard refuses to run
# rather than silently falling back to the NFS-direct write.
EXPORT_DIR="${EXPORT_DIR:-${ENROOT_TEMP_PATH:-/tmp}}"
command -v rsync >/dev/null || { echo "ERROR: rsync needed for the throttled copy to \$HOME" >&2; exit 1; }
avail_kb=$(df -Pk "$EXPORT_DIR" | awk 'NR==2 {print $4}')
avail_gib=$(( avail_kb / 1024 / 1024 ))
if [ "$avail_gib" -lt 110 ]; then
  echo "ERROR: $EXPORT_DIR has ${avail_gib} GiB free, need ~110 GiB for the local export." >&2
  echo "Set EXPORT_DIR to a node-local path with space; do NOT export directly onto NFS." >&2
  exit 1
fi
OUT_LOCAL="$EXPORT_DIR/$(basename "$OUT_SQSH").export.$$"
rm -f "$OUT_LOCAL"
enroot export -o "$OUT_LOCAL" "$BUILD_NAME"
ls -lh "$OUT_LOCAL"
rm -f "$OUT_SQSH"
rsync --inplace --bwlimit=200M "$OUT_LOCAL" "$OUT_SQSH"
rm -f "$OUT_LOCAL"
ls -lh "$OUT_SQSH"
echo "BUILD_DONE: $OUT_SQSH"
