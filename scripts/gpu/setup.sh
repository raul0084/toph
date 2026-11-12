#!/usr/bin/env bash
# toph GPU runner: sets everything up, runs 3D detection on KITTI Tracking, packages results.
#
# One-liner (Linux / inside WSL):
#   curl -fsSL https://raw.githubusercontent.com/raul0084/toph/main/scripts/gpu/setup.sh | bash
#
# Safe to re-run: every finished step is skipped, downloads resume.
# Everything lives in $TOPH_HOME (default ~/toph-gpu). Delete that folder to uninstall.
#
# Overridable env vars:
#   TOPH_HOME   install dir                     (default ~/toph-gpu)
#   TOPH_MODEL  pv_rcnn | pointpillar           (default pv_rcnn)
#   TOPH_REPO   git URL of the toph repo
#   TOPH_SPLITS "training testing"              (which KITTI tracking splits to run)

main() {
set -Eeuo pipefail

TOPH_HOME="${TOPH_HOME:-$HOME/toph-gpu}"
TOPH_MODEL="${TOPH_MODEL:-pv_rcnn}"
TOPH_REPO="${TOPH_REPO:-https://github.com/raul0084/toph.git}"
TOPH_SPLITS="${TOPH_SPLITS:-training testing}"
KITTI_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_velodyne.zip"
NEED_GB=90

mkdir -p "$TOPH_HOME"
LOG="$TOPH_HOME/setup.log"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n\033[1;36m==> [%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
done_mark() { touch "$TOPH_HOME/.done_$1"; }
is_done() { [[ -f "$TOPH_HOME/.done_$1" ]]; }
fail() {
  printf '\n\033[1;31m✗ Something went wrong.\033[0m\n'
  printf 'Please send this file to the person who sent you this script:\n  %s\n' "$LOG"
  printf 'Re-running the same command later will pick up where it left off.\n'
}
trap fail ERR

case "$TOPH_MODEL" in
  pv_rcnn)     CFG="cfgs/kitti_models/pv_rcnn.yaml";     GDRIVE_ID="1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-" ;;
  pointpillar) CFG="cfgs/kitti_models/pointpillar.yaml"; GDRIVE_ID="1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm" ;;
  *) echo "Unknown TOPH_MODEL=$TOPH_MODEL"; exit 1 ;;
esac

# ---------------------------------------------------------------- checks
step "Checking the machine"
export PATH="/usr/lib/wsl/lib:$PATH"   # nvidia-smi location inside WSL
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  echo "No working NVIDIA GPU driver found."
  echo "On Windows: install the latest driver from nvidia.com (Game Ready or Studio), reboot, re-run."
  false
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
CAP_MAJOR=${CAP%%.*}
echo "GPU: $GPU_NAME (compute capability $CAP)"

FREE_GB=$(df -BG --output=avail "$TOPH_HOME" | tail -1 | tr -dc '0-9')
if ! is_done data && (( FREE_GB < NEED_GB )); then
  echo "Need ~${NEED_GB} GB free disk space, only ${FREE_GB} GB available in $TOPH_HOME."
  false
fi

# Pick a CUDA/PyTorch stack the GPU supports.
if (( CAP_MAJOR >= 12 )); then
  # RTX 50xx (Blackwell) needs CUDA 12.8+. Less battle-tested with OpenPCDet.
  echo "Blackwell GPU detected -> using CUDA 12.8 stack (experimental)"
  CUDA_LABEL="cuda-12.8.0"; GXX="13"
  TORCH_PKGS="torch==2.7.1 torchvision==0.22.1"; TORCH_INDEX="https://download.pytorch.org/whl/cu128"
  SPCONV="spconv-cu126"
else
  CUDA_LABEL="cuda-11.8.0"; GXX="11"
  TORCH_PKGS="torch==2.1.2 torchvision==0.16.2"; TORCH_INDEX="https://download.pytorch.org/whl/cu118"
  SPCONV="spconv-cu118"
fi

# ---------------------------------------------------------------- system packages
if ! is_done apt; then
  step "Installing system packages"
  SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"
  $SUDO apt-get update -y
  $SUDO apt-get install -y git wget unzip ca-certificates
  done_mark apt
fi

# ---------------------------------------------------------------- python env
MF="$TOPH_HOME/miniforge"
if [[ ! -x "$MF/bin/conda" ]]; then
  step "Installing Miniforge (private Python, doesn't touch the system)"
  wget -q -O "$TOPH_HOME/miniforge.sh" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$TOPH_HOME/miniforge.sh" -b -p "$MF"
  rm "$TOPH_HOME/miniforge.sh"
fi
# shellcheck disable=SC1091
source "$MF/etc/profile.d/conda.sh"

if ! is_done env; then
  step "Creating Python environment with CUDA toolkit (takes a while)"
  conda env remove -y -n toph >/dev/null 2>&1 || true
  conda create -y -n toph python=3.10
  conda activate toph
  conda install -y -c "nvidia/label/$CUDA_LABEL" cuda-toolkit
  conda install -y -c conda-forge "gxx_linux-64=$GXX"
  pip install $TORCH_PKGS --index-url "$TORCH_INDEX"
  pip install "$SPCONV" "numpy<1.24" "numba<0.59" llvmlite tensorboardX easydict pyyaml \
    "scikit-image<0.22" tqdm SharedArray opencv-python-headless pyquaternion gdown
  done_mark env
fi
conda activate toph
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="$CAP"

python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'; print('PyTorch sees', torch.cuda.get_device_name(0))"

# ---------------------------------------------------------------- code
REPO="$TOPH_HOME/toph"
step "Getting the latest toph code"
if [[ -d "$REPO/.git" ]]; then
  git -C "$REPO" pull --ff-only
  git -C "$REPO" submodule update --init --recursive
else
  git clone --recursive "$TOPH_REPO" "$REPO"
fi
PCDET="$REPO/third_party/OpenPCDet"

if ! is_done build; then
  step "Building OpenPCDet CUDA ops (5-15 min)"
  (cd "$PCDET" && python setup.py develop)
  done_mark build
fi
python -c "import pcdet, spconv; from pcdet.ops.iou3d_nms import iou3d_nms_cuda; print('OpenPCDet OK')"

# ---------------------------------------------------------------- model
CKPT="$TOPH_HOME/models/$TOPH_MODEL.pth"
if [[ ! -s "$CKPT" ]]; then
  step "Downloading pretrained $TOPH_MODEL model"
  mkdir -p "$TOPH_HOME/models"
  if ! gdown "$GDRIVE_ID" -O "$CKPT"; then
    echo "Google Drive refused the download (it rate-limits popular files). Try again in a few hours."
    false
  fi
fi

# ---------------------------------------------------------------- data
KITTI="$TOPH_HOME/kitti_tracking"
if ! is_done data; then
  step "Downloading KITTI Tracking LiDAR (~35 GB, resumes if interrupted)"
  mkdir -p "$KITTI"
  wget -c --progress=dot:giga -O "$KITTI/velodyne.zip" "$KITTI_URL"
  step "Unzipping"
  unzip -q -o "$KITTI/velodyne.zip" -d "$KITTI"
  rm "$KITTI/velodyne.zip"
  done_mark data
fi

# ---------------------------------------------------------------- detection
OUT="$TOPH_HOME/results/$TOPH_MODEL"
step "Running $TOPH_MODEL on KITTI Tracking ($TOPH_SPLITS)"
# shellcheck disable=SC2086
python "$REPO/scripts/gpu/run_detection.py" \
  --openpcdet "$PCDET" --cfg "$CFG" --ckpt "$CKPT" \
  --kitti-root "$KITTI" --out "$OUT" --splits $TOPH_SPLITS
cp "$LOG" "$OUT/setup.log"

ZIP="$TOPH_HOME/toph_detections_$TOPH_MODEL.zip"
python -c "import shutil; shutil.make_archive('${ZIP%.zip}', 'zip', '$TOPH_HOME/results', '$TOPH_MODEL')"

trap - ERR
printf '\n\033[1;32m✓ All done!\033[0m\n'
printf 'Please send this file back: %s (%s)\n' "$ZIP" "$(du -h "$ZIP" | cut -f1)"
}

main "$@"
