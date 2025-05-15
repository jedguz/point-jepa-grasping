#!/usr/bin/env bash
# Pre-train Voxel-MAE on TSDF volumes
env_name=voxelmae_env  # name of conda/venv if used

# ensure ext/voxel-mae is on PYTHONPATH
echo "Adding voxel-mae to PYTHONPATH"
export PYTHONPATH="$PYTHONPATH:$(pwd)/ext/voxel-mae"

CONFIG=configs/tsdf_masked_voxelmae.py
GPUS=1

# single-GPU launch
python ext/voxel-mae/tools/train.py $CONFIG --work-dir work_dirs/tsdf_voxelmae