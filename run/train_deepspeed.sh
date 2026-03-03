# source /inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/miniconda/etc/profile.d/conda.sh
# conda activate epona

# cd /inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona
# export PYTHONPATH="$(pwd):$PYTHONPATH"

export NODES_NUM=1
export GPUS_NUM=8

# 设置环境变量
export HYDRA_FULL_ERROR=1
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/inspire/hdd/project/roboticsystem2/public/nuplan/dataset/maps"
export NAVSIM_EXP_ROOT="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/exp"
export NAVSIM_DEVKIT_ROOT="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona"
export OPENSCENE_DATA_ROOT="/inspire/hdd/project/roboticsystem2/public/OpenScene/openscene-v1.1"

# 设置NCCL超时时间（单位：秒），默认600秒（10分钟），设置为3600秒（60分钟）
export NCCL_TIMEOUT=1000000
# 可选：启用NCCL调试信息（如果需要调试）
# export NCCL_DEBUG=INFO
# export TORCH_NCCL_TRACE_BUFFER_SIZE=1000000

torchrun --nnodes=$NODES_NUM --nproc_per_node=$GPUS_NUM \
scripts/train_deepspeed.py \
  --batch_size 2 \
  --lr 2e-5 \
  --exp_name "train-navsim" \
  --config /inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/configs/dit_config_dcae_navsim.py \
  --iter 100000 \
  --eval_steps 10000 \
  --resume_path "/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/checkpoints/Epona/epona_nuplan.pkl"
  # 注意：继续训练时使用 --load_from_deepspeed 和 --resume_step，不需要 --resume_path
  # --load_from_deepspeed "/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/exp/ckpt/1/38000" \
  # --resume_step 38000 \