export NODES_NUM=1
export GPUS_NUM=1
export CUDA_VISIBLE_DEVICES=0 #,1,2,3
export PYTHONPATH="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona:$PYTHONPATH"

# 检查并确保NumPy版本兼容性
echo "Checking NumPy version..."
python3 -c "import numpy; print('Current NumPy version:', numpy.__version__)"
# 如果NumPy版本大于等于2.0,则降级到1.24.3
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" | cut -d. -f1)
if [ "$NUMPY_VERSION" -ge "2" ]; then
    echo "NumPy version is $NUMPY_VERSION.x, downgrading to 1.26.3..."
    pip install numpy==1.26.3
fi

# 设置地图路径环境变量(使用本地复制的地图文件以避免FUSE文件系统的SQLite I/O问题)
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

#-m debugpy --connect 5678 --wait-for-client
# python3 scripts/test/test_traj.py \
# --exp_name "test-nuplan" \
# --start_id 0 --end_id 100 \
# --resume_path "/lpai/volumes/ad-agent-vol-ga/wugaoqiang/code/wm_rl/Epona_ckpt/epona_nuplan+nusc.pkl" \
# --config configs/dit_config_dcae_nuplan.py

python scripts/test/run_navsim_eval_pipeline1.py \
--exp_name "navsim-traj" \
--resume_path "/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/exp/ckpt/train-navsim/tvar_90000.pkl" \
--config configs/dit_config_dcae_nuplan.py \
--navsim_log_path /inspire/hdd/project/roboticsystem2/public/OpenScene/openscene-v1.1/navsim_logs/test \
--sensor_blobs_path /inspire/hdd/project/roboticsystem2/public/OpenScene/openscene-v1.1/sensor_blobs/test \
--navsim_exp_root /inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/exp \
--start_id 0 --end_id 500

# cat /inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/exp/pdm_scores/2025.12.27.22.30.29.csv # pdm score
