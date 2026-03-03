# 1. 激活conda环境
source /inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/miniconda/etc/profile.d/conda.sh
conda activate epona

# 2. 设置工作目录
cd /inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 3. 设置分布式训练所需的环境变量
export NODES_NUM=4            # 设置节点数量
export GPUS_NUM=8             # 每个节点的GPU数量
export PET_NODE_RANK=${PET_NODE_RANK}  # 当前节点的rank（可以通过环境变量传递）
export MASTER_ADDR=${MASTER_ADDR}      # 主节点IP地址
export MASTER_PORT=${MASTER_PORT}      # 主节点端口
export WORLD_SIZE=$(($NODES_NUM * $GPUS_NUM)) # 总进程数，节点数 * 每节点GPU数

# 设置其他环境变量
export HYDRA_FULL_ERROR=1
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/inspire/hdd/project/roboticsystem2/public/nuplan/dataset/maps"
export NAVSIM_EXP_ROOT="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/exp"
export NAVSIM_DEVKIT_ROOT="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona"
export OPENSCENE_DATA_ROOT="/inspire/hdd/project/roboticsystem2/public/OpenScene/openscene-v1.1"

# 4. 启动分布式训练
torchrun --nnodes=$NODES_NUM \
         --node_rank=$PET_NODE_RANK \
         --nproc_per_node=$GPUS_NUM \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         scripts/train_deepspeed.py \
         --batch_size 2 \
         --lr 2e-5 \
         --exp_name "train-navsim" \
         --config /inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/configs/dit_config_dcae_navsim.py \
         --iter 100000 \
         --eval_steps 10000 \
         --resume_path "/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/checkpoints/Epona/epona_nuplan.pkl"
