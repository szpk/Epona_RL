#!/bin/bash

# 默认配置
NUM_GPUS=1
DATA_SPLIT=test
MAX_SCENARIOS=99999

# 切换到项目根目录
cd /inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona

# 设置环境变量
export HYDRA_FULL_ERROR=1
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/inspire/hdd/project/roboticsystem2/public/nuplan/dataset/maps"
export NAVSIM_EXP_ROOT="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/exp"
export NAVSIM_DEVKIT_ROOT="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona"
export OPENSCENE_DATA_ROOT="/inspire/hdd/project/roboticsystem2/public/OpenScene/openscene-v1.1"

# 设置 metric_cache_path
if [[ "$DATA_SPLIT" == "test" || "$DATA_SPLIT" == "navtest" ]]; then
    METRIC_CACHE_PATH="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/exp/metric_cache"
else
    METRIC_CACHE_PATH="/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/recogdrive/exp/metric_cache_train"
fi

# 运行评估
python3 scripts/test/run_navsim_eval_pipeline.py \
  train_test_split=${DATA_SPLIT} \
  +agent.config_path=configs/dit_config_dcae_nuplan.py \
  +agent.checkpoint_path=/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/checkpoints/Epona/epona_nuplan.pkl \
  +agent.use_sde=false \
  experiment_name=epona_eval \
  output_dir=./exp/navsim_eval \
  metric_cache_path=${METRIC_CACHE_PATH} \
  worker=ray_distributed \
  +worker.threads_per_node=${NUM_GPUS} \
  +num_gpus=${NUM_GPUS} \
  +max_scenarios=${MAX_SCENARIOS}
