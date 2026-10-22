#!/bin/bash
# Inner build: replays environment/Dockerfile + agents/aide/Dockerfile
# inside a writable enroot sandbox (ubuntu:22.04 arm64), to produce the same
# image for arm64+Blackwell. The repo is mounted at /work. Run as root.
#
# Only deviations from the Dockerfiles (each forced by arm64/Blackwell, or noted):
#   1. Miniconda installer: Linux-x86_64.sh  -> Linux-aarch64.sh
#   2. GPU torch stack (torch/torchvision/torchaudio==2.2.0, torchtext==0.17.0 --
#      pinned in BOTH the Dockerfile AND requirements.txt): replaced by torch/
#      vision/audio from the cu128 index (Blackwell sm_100); torchtext dropped
#      (no arm64 wheel). Filtered out of requirements.txt at build time.
#   3. tensorflow stack (tensorflow/tensorflow-hub/tf-keras in requirements.txt;
#      tensorflow[and-cuda] in the Dockerfile) -> dropped entirely (per request).
#   4. skip the tuna (China) PyPI mirror in the aide layer -> default PyPI
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

CONDA_ENV_NAME=agent
PYTHON_VERSION=3.11

# ===================== environment/Dockerfile =====================
apt-get update && apt-get install -y \
    curl wget git vim nano unzip zip p7zip-full \
    python3 python3-pip python3-venv python3-dev python-is-python3 \
    build-essential openssh-server tmux gettext sudo \
    ffmpeg libsm6 libxext6
pip install jupyter
# arm64 build deps not needed on amd64 (where these had prebuilt wheels):
# pkg-config + HDF5 headers so h5py builds from source on aarch64.
apt-get install -y pkg-config libhdf5-dev
rm -rf /var/lib/apt/lists/*

python -m pip install --upgrade pip
pip install virtualenv

# EDIT 1: arch-matched, PYTHON-PINNED Miniconda -- aarch64 on beta/B200,
# x86_64 on alpha/H100+H200. `uname -m` is exactly Anaconda's installer suffix
# on both arches, so one script builds either image. Everything else below is
# arch-neutral: the arm-driven filters (torchtext/h5py/gdcm/tensorflow drops)
# are KEPT on x86_64 on purpose -- the agent's package set must match the beta
# image or runs from the two clusters are not comparable.
#
# ⚠️ PINNED, not `latest`. `Miniconda3-latest` now ships Python 3.14, and the
# base env is where `pip install -e /mlebench` lands -- mlebench requires
# tensorflow>=2.16, which has no cp314 wheels, so the 08-13 x86_64 build died
# with "No matching distribution found for tensorflow". The July aarch64 image
# built fine only because `latest` was an older Python then; that is exactly
# the kind of silent drift a pin removes. py312 has full wheel coverage for
# the mlebench dependency set on both arches.
MINICONDA_VERSION="${MINICONDA_VERSION:-Miniconda3-py312_25.9.1-1}"
wget "https://repo.anaconda.com/miniconda/${MINICONDA_VERSION}-Linux-$(uname -m).sh" -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/conda
rm /tmp/miniconda.sh
/opt/conda/bin/conda init

cp /work/environment/requirements.txt /tmp/requirements.txt

/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
/opt/conda/bin/conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" -y
# newer pip recognizes more aarch64 manylinux wheel tags -> fewer source builds
/opt/conda/bin/conda run -n "${CONDA_ENV_NAME}" pip install --upgrade pip

# requirements.txt itself pins the GPU/TF stack; drop those lines (cu128 torch
# below handles torch/vision/audio for Blackwell, torchtext has no arm64 wheel,
# and the tensorflow ecosystem is dropped per request). Filtered copy only --
# the repo requirements.txt is untouched, still used by the docker/amd64 path.
grep -viE '^(torch|torchvision|torchaudio|torchtext|tensorflow|tensorflow-hub|tf-keras)==' \
    /tmp/requirements.txt > /tmp/requirements.filtered.txt
# h5py 3.11.0 has NO aarch64 wheel (available: ...3.10.0, 3.12.1, 3.13.0...), so
# pip fell back to the sdist, which fails to Cython-build. Bump to the nearest
# wheel-available release (numpy-1.26 compatible). Filtered copy only.
sed -i 's/^h5py==.*/h5py==3.12.1/' /tmp/requirements.filtered.txt
# --only-binary=h5py: take that prebuilt wheel rather than ever building the sdist.
/opt/conda/bin/conda run -n "${CONDA_ENV_NAME}" pip install -r /tmp/requirements.filtered.txt --only-binary=h5py
/opt/conda/bin/conda clean -afy
# NOTE: Blackwell torch is installed LAST (after the aide editable install
# below), because fastai==2.7.17 caps torch<2.5 and any pip resolve downgrades
# torch back to a CPU build to satisfy that cap.

export PATH="/opt/conda/bin:${PATH}"

# grading server + mlebench package into the conda BASE env (COPY . /mlebench; pip install -e)
mkdir -p /mlebench
# ⚠️ EXCLUDES MATTER: this bakes the host tree into the image, and the repo
# accumulates enormous in-tree dirs. The 08-13 x86_64 build shipped a 94G
# .sqsh (vs 12.5G for the July aarch64 one) purely because `runs/` was copied
# in. That is not just waste: `enroot create` unpacks the FULL image per
# container, so at 9 workers a 94G image needs ~846G of node-local disk
# against ~447G available -- the run cannot start. Competition data is
# bind-mounted at /home/data at run time and must never be baked in either.
( cd /work && tar -cf - \
    --exclude=.git --exclude=logs --exclude=runs --exclude=data --exclude=data_privateonly \
    --exclude=__pycache__ --exclude=.pytest_cache --exclude='*.sqsh' \
    --exclude=.venv --exclude=.env --exclude='.env.*' --exclude=cache \
    --exclude=models --exclude=build --exclude=dist --exclude='*.egg-info' \
    agents configs environment mlebench pyproject.toml README.md LICENSE ) \
  | ( cd /mlebench && tar -xf - )
pip install flask
pip install -e /mlebench

mkdir -p /private && chmod 700 /private
cp /work/environment/grading_server.py /private/grading_server.py
cp /work/environment/instructions.txt /home/instructions.txt
cp /work/environment/instructions_obfuscated.txt /home/instructions_obfuscated.txt
cp /work/environment/validate_submission.sh /home/validate_submission.sh
cp /work/environment/entrypoint.sh /entrypoint.sh

useradd -m nonroot || true
mkdir -p /home/submission
chmod +x /entrypoint.sh

# ===================== agents/aide/Dockerfile =====================
export SUBMISSION_DIR=/home/submission LOGS_DIR=/home/logs CODE_DIR=/home/code AGENT_DIR=/home/agent
mkdir -p "${LOGS_DIR}" "${CODE_DIR}" "${AGENT_DIR}"
apt-get update && apt-get install -y ffmpeg libsm6 libxext6
rm -rf /var/lib/apt/lists/*
# EDIT 4: default PyPI (skip tuna mirror)
python -m pip install --upgrade pip
( cd /work/agents/aide && tar --exclude=__pycache__ --exclude='*.egg-info' -cf - src ) \
  | ( cd "${AGENT_DIR}" && tar -xf - )
# aide/src/requirements.txt (read by setup.py as install_requires) re-pins the
# GPU stack to amd64 versions and lists gdcm==1.1 (absent from PyPI) plus
# torchtext/h5py==3.11.0 (no arm64 wheels). Strip those from the in-container
# copy so the editable install keeps the cu128 torch + env h5py and only adds
# aide's own deps (arrow, omegaconf, loguru, openai 1.x, ...). Repo untouched.
# Filter each dependency group as well as the aggregate requirements file.
for aide_requirements in "${AGENT_DIR}"/src/requirements*.txt; do
    grep -viE '^(torch|torchvision|torchaudio|torchtext|h5py|gdcm)==' \
        "$aide_requirements" > /tmp/aide_reqs.filtered.txt
    cp /tmp/aide_reqs.filtered.txt "$aide_requirements"
done
/opt/conda/bin/conda run -n "${CONDA_ENV_NAME}" pip install -e "${AGENT_DIR}/src/."
/opt/conda/bin/conda clean -afy
cp /work/agents/aide/start.sh /work/agents/aide/snapshot_check.sh \
    /work/agents/aide/additional_notes.txt "${AGENT_DIR}/"

# EDIT 2: Blackwell torch as the ABSOLUTE LAST pip op. Both the env-requirements
# step and the aide editable install pull a CPU torch<2.5 (fastai==2.7.17's cap),
# and the last pip resolve wins -- so the cu128 install must come after them all.
# Uninstall the CPU trio first (an unpinned install would see torch satisfied and
# skip it). fastai's cap is necessarily violated on Blackwell (flagged).
/opt/conda/bin/conda run -n "${CONDA_ENV_NAME}" pip uninstall -y torch torchvision torchaudio
/opt/conda/bin/conda run -n "${CONDA_ENV_NAME}" pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
/opt/conda/bin/conda clean -afy

# ===================== verify torch on the Blackwell GPU =====================
# --no-capture-output so a CUDA failure actually propagates (conda run otherwise
# swallows the exception and returns 0, masking a broken build).
/opt/conda/bin/conda run --no-capture-output -n "${CONDA_ENV_NAME}" python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available in agent env"
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
x = torch.randn(64, 64, device="cuda")
print("gpu matmul ok, sum =", float((x @ x).sum()))
PY
echo "BUILD_INNER_OK"
