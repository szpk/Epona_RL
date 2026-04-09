#!/bin/bash

export NODES_NUM=1
export GPUS_NUM=1
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona:$PYTHONPATH"

# 检查并确保NumPy版本兼容性
echo "Checking NumPy version..."
python3 -c "import numpy; print('Current NumPy version:', numpy.__version__)"

# 设置环境变量
export HYDRA_FULL_ERROR=1
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/inspire/hdd/project/roboticsystem2/public/nuplan/dataset/maps"
export NAVSIM_EXP_ROOT="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/exp"
export NAVSIM_DEVKIT_ROOT="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona"
export OPENSCENE_DATA_ROOT="/inspire/hdd/project/roboticsystem2/public/OpenScene/openscene-v1.1"

# 设置单机单卡环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29502
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

echo "Using ${GPUS_NUM} GPU(s): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# 默认参数
CHECKPOINT="${CHECKPOINT:-exp/ckpt/train-navsim-grpo3/tvar_1500.pkl}"
EXP_NAME="${EXP_NAME:-navsim-video-$(date +%Y%m%d-%H%M%S)}"
START_ID="${START_ID:-0}"
END_ID="${END_ID:-10}"
SPLIT="${SPLIT:-val}"
MODE="${MODE:-teacher}"  # teacher or free

if [ "$MODE" = "free" ]; then
    SCRIPT="scripts/test/test_navsim_free.py"
    echo "Running in FREE generation mode (autoregressive, self-predicted trajectory)"
else
    SCRIPT="scripts/test/test_navsim.py"
    echo "Running in TEACHER FORCING mode (using ground truth trajectory)"
fi

echo "========================================="
echo "NavSim Video Generation"
echo "========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Experiment Name: $EXP_NAME"
echo "Start ID: $START_ID"
echo "End ID: $END_ID"
echo "Dataset Split: $SPLIT"
echo "Generation Mode: $MODE"
echo "Script: $SCRIPT"
echo "========================================="

python $SCRIPT \
    --exp_name "$EXP_NAME" \
    --config configs/dit_config_dcae_navsim.py \
    --resume_path "$CHECKPOINT" \
    --split "$SPLIT" \
    --start_id "$START_ID" \
    --end_id "$END_ID"

echo ""
echo "========================================="
echo "Video generation completed!"
echo "Videos saved to: test_videos/$EXP_NAME/"
echo "========================================="
