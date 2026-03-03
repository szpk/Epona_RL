#!/bin/bash
# GRPO Training Script for Epona TrajDiT using NavSim dataset
# This script trains only the trajectory output (TrajDiT) using GRPO while freezing STT and DiT

export NODES_NUM=1
export GPUS_NUM=8

# 设置环境变量
export HYDRA_FULL_ERROR=1
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/inspire/hdd/project/roboticsystem2/public/nuplan/dataset/maps"
export NAVSIM_EXP_ROOT="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/exp"
export NAVSIM_DEVKIT_ROOT="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona"
export OPENSCENE_DATA_ROOT="/inspire/hdd/project/roboticsystem2/public/OpenScene/openscene-v1.1"

# NCCL 配置
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
# 设置NCCL超时时间（单位：秒），默认600秒（10分钟），设置为3600秒（60分钟）
export NCCL_TIMEOUT=1000000

# CUDA调试配置（可选，用于调试CUDA错误）
export CUDA_LAUNCH_BLOCKING=1

# 添加 recogdrive 到 PYTHONPATH (用于 PDM scorer)
export PYTHONPATH="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/recogdrive:${PYTHONPATH}"

# 预训练模型路径 (用于初始化和作为参考策略)
PRETRAINED_MODEL="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/exp/ckpt/train-navsim/tvar_90000.pkl"

# 切换到项目根目录
cd $NAVSIM_DEVKIT_ROOT

torchrun --nnodes=$NODES_NUM --nproc_per_node=$GPUS_NUM \
scripts/train_deepspeed.py \
  --batch_size 4 \
  --lr 1e-5 \
  --exp_name "train-navsim-grpo1" \
  --config /inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/configs/dit_config_dcae_navsim_grpo.py \
  --iter 40000 \
  --eval_steps 2000 \
  --resume_path "$PRETRAINED_MODEL"
  # 注意：
  # 1. GRPO 训练使用较小的 batch_size 和学习率
  # 2. 配置文件中已设置 fix_stt=True 和 fix_dit=True，只训练 TrajDiT
  # 3. 继续训练时使用 --load_from_deepspeed 和 --resume_step
  # --load_from_deepspeed "/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/exp/ckpt/train-navsim-grpo/10000" \
  # --resume_step 10000
