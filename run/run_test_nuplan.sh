#!/bin/bash

# 切换到 Epona 项目根目录
cd /inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona

# 运行测试脚本
# 使用 --use_sde 参数：true 表示 SDE 采样8条轨迹，false（不添加该参数）表示 ODE 采样1条轨迹
python3 scripts/test/test_traj.py \
  --exp_name "test-traj-5steps" \
  --start_id 0 --end_id 100 \
  --resume_path "/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/checkpoints/Epona/epona_nuplan.pkl" \
  --config configs/dit_config_dcae_nuplan.py \
  # --use_sde
