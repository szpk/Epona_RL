# run_navsim_eval_pipeline.py 完整流程说明

## 一、脚本目的

评估 Epona 模型在 nuPlan 测试集上的轨迹预测性能，使用 **PDM (Planning Domain Model) 评分系统**进行标准化评估。

**这不是训练过程，而是纯评估（evaluation）流程**，只进行模型推理，不更新模型参数。

**重要：PDM 评分系统必须使用自行车模型仿真**，这是 PDM 评分系统的核心设计。仿真用于模拟车辆实际执行预测轨迹时的行为，考虑车辆动力学约束，使评分更真实。recogdrive 等其他模型的评估也使用同样的仿真机制。

---

## 二、输入数据

### 1. 数据集路径（通过环境变量配置）

- **场景标注数据**：`${OPENSCENE_DATA_ROOT}/navsim_logs/test`
  - 包含：ego poses, annotations, 场景元数据等
  
- **传感器数据**：`${OPENSCENE_DATA_ROOT}/sensor_blobs/test`
  - 包含：图像（cameras）、LiDAR 点云等传感器数据

### 2. Metric Cache（预计算数据）

- **路径**：`/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/recogdrive/exp/metric_cache_train`
- **内容**：每个场景的预计算指标（**ground truth 数据**）
  - `ego_state`：初始 ego 状态
  - `trajectory`：ground truth 轨迹（真实车辆行驶轨迹）
  - `observation`：观测数据
  - `centerline`：中心线
  - `route_lane_ids`：路线车道 ID
  - `drivable_area_map`：可行驶区域地图

**注意**：metric_cache 存储的是场景的真实数据（ground truth），用于评分时的参考标准。但评估流程中仍需要**仿真（PDMSimulator）**，因为仿真是对**模型预测轨迹**的执行模拟，用于评估预测轨迹在实际执行时的可行性和安全性。

### 3. 模型检查点

- **路径**：`checkpoints/Epona/epona_nuplan.pkl`
- **内容**：训练好的 Epona 模型权重

### 4. 配置文件

- **模型配置**：`configs/dit_config_dcae_nuplan.py`
- **评估配置**：`navsim/planning/script/config/pdm_scoring/default_run_pdm_score.yaml`

---

## 三、完整处理流程（按步骤）

### 阶段 1：初始化（main 函数）

```
1. 加载 Hydra 配置
   - 读取 default_run_pdm_score.yaml
   - 解析 agent、simulator、scorer 等配置

2. 初始化 Ray worker（支持 GPU 并行）
   - 使用 ray_distributed worker
   - 为每个任务分配 1 个 GPU（num_gpus=1.0）

3. 创建 SceneLoader
   - 加载测试场景列表（navtest split）
   - 过滤场景，获取要评估的 tokens

4. 创建 MetricCacheLoader
   - 加载预计算的 metric cache
   - 建立 token 到 cache 路径的映射

5. 计算要评估的 tokens
   - 取 scene_loader.tokens 和 metric_cache_loader.tokens 的交集
   - 确保每个场景都有对应的 metric cache
```

### 阶段 2：并行评估（run_pdm_score 函数，每个 Ray worker 执行）

对每个场景 token，执行以下步骤：

#### 步骤 1：加载场景数据

```python
# 从 metric_cache 加载
- ego_state: 初始 ego 状态
- trajectory: ground truth 轨迹
- observation: 观测数据
- centerline: 中心线
- route_lane_ids: 路线车道 ID
- drivable_area_map: 可行驶区域地图

# 从 scene_loader 加载
- agent_input: 包含 cameras 和 ego_statuses
  * cameras: 前置摄像头图像序列
  * ego_statuses: ego 状态序列（位置、速度、加速度等）
```

#### 步骤 2：模型推理（EponaAgent.compute_trajectory）

**a) 图像预处理：**
```python
- 提取 condition_frames（默认3帧）的前置摄像头图像
- 转换为 tensor，resize 到 (512, 1024)
- Stack: (condition_frames, C, H, W) → (1, condition_frames, C, H, W)
```

**b) 位姿提取：**
```python
- 从 agent_input.ego_statuses 提取 condition_frames+1 个位姿
- 计算相对位姿（pose, yaw）
- 转换为 tensor: (1, condition_frames+1, 3)
```

**c) 模型前向传播：**
```python
1. VAE Tokenizer 编码图像 → start_latents
2. TrainTransformersDiT.step_eval() 生成轨迹：
   - 如果 use_sde=true：生成 8 条轨迹（SDE 采样）
   - 如果 use_sde=false：生成 1 条轨迹（ODE 采样）
3. 输出：相对坐标系下的轨迹 (horizon, 3) [x, y, heading]
```

#### 步骤 3：PDM 评分（pdm_score 函数）

**核心流程：**
```python
1. 模型推理输出预测轨迹（model_trajectory）
   - 和 test_traj.py 一样的流程：使用 epona_nuplan.pkl 模型
   - 输入：OPENSCENE_DATA_ROOT/test 数据集的图像和位姿
   - 输出：相对坐标系下的预测轨迹 (horizon, 3) [x, y, heading]

2. 将预测轨迹和 metric_cache 的 ground truth 轨迹一起传给 PDM 评分系统
   - metric_cache.trajectory：ground truth 轨迹（真实车辆行驶轨迹）
   - model_trajectory：模型预测轨迹
   - 两个轨迹都会被处理并评分

3. PDM 评分系统处理流程：
   a) 轨迹转换：将两个轨迹都转换为全局坐标系，插值到固定时间点
   b) 仿真（PDMSimulator）：对两个轨迹都进行仿真
   c) 评分（PDMScorer）：基于仿真后的状态计算各项指标
```

**a) 轨迹转换：**
```python
- 将预测轨迹和 ground truth 轨迹都转换为全局坐标系
- 创建 InterpolatedTrajectory 对象
- 插值到固定时间点（40个点，0.1秒间隔，4秒时间范围）
```

**b) 仿真（PDMSimulator）：**
```python
# 代码逻辑（pdm_score.py 第 107-109 行）：
trajectory_states = np.concatenate([
    pdm_states[None, ...],      # ground truth 轨迹（metric_cache.trajectory）
    pred_states[None, ...]      # 预测轨迹（model_trajectory）
], axis=0)

simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)
# 对两个轨迹都进行仿真，生成考虑车辆动力学约束的模拟状态

# 评分器使用仿真后的状态（pdm_score.py 第 111 行）：
scores = scorer.score_proposals(
    simulated_states,  # 注意：使用的是 simulated_states，不是原始的 trajectory_states
    ...
)
```

**重要说明：**
- **PDM 评分系统必须使用仿真**，这是 PDM 评分系统的核心设计，不是可选的
- 评分器（PDMScorer）的 `score_proposals()` 方法接收的是 `simulated_states`，不是原始的轨迹点
- 仿真使用自行车模型（bicycle model）和 LQR 跟踪器，模拟车辆实际执行轨迹时的行为
- 模型输出的轨迹点可能不符合车辆动力学约束（如急转弯、急加速），仿真可以更真实地评估轨迹的可行性
- **recogdrive 的评估脚本也使用了同样的仿真**（`run_recogdrive_agent_pdm_score_evaluation_2b.sh` 也调用 `pdm_score` 函数）
- 评分时使用预测轨迹的仿真结果（pred_idx=1），ground truth 轨迹用于对比

**c) 评分（PDMScorer）：**
```python
计算多个指标：
1. no_at_fault_collisions：无碰撞标志（0/1）
2. drivable_area_compliance：可行驶区域合规性
3. ego_progress：前进距离（加权：5.0）
4. time_to_collision_within_bound：碰撞时间（加权：5.0）
5. comfort：舒适度（加权：2.0）
6. driving_direction_compliance：行驶方向合规性

综合得分 = weighted sum of all metrics
```

**d) 返回 PDMResults：**
```python
- 包含所有子指标和总分
- 每个场景一个结果
```

#### 步骤 4：结果收集

```python
- 每个场景的结果存入 score_row
- 如果出错，标记 valid=False
- 收集所有场景的结果
```

### 阶段 3：结果汇总（main 函数）

```
1. 合并所有 worker 的结果
   - 收集所有场景的评分结果

2. 计算统计信息：
   - 成功场景数：num_successful_scenarios
   - 失败场景数：num_failed_scenarios
   - 平均得分：所有有效场景的平均 score

3. 保存 CSV 文件：
   - 路径：./exp/navsim_eval/{timestamp}.csv
   - 格式：每行一个场景，包含 token, valid, score, 各子指标
   - 最后一行是平均值
```

---

## 四、输出内容

### CSV 文件格式

| 列名 | 说明 |
|------|------|
| `token` | 场景 ID（唯一标识符） |
| `valid` | 是否成功处理（True/False） |
| `score` | **综合得分**（主要评估指标） |
| `no_at_fault_collisions` | 无碰撞标志（0/1） |
| `drivable_area_compliance` | 可行驶区域合规性 |
| `ego_progress` | 前进距离 |
| `time_to_collision_within_bound` | 碰撞时间 |
| `comfort` | 舒适度 |
| `driving_direction_compliance` | 行驶方向合规性 |

**最后一行**：所有场景的平均值（token="average"）

### 输出位置

- **目录**：`./exp/navsim_eval/`
- **文件名**：`{timestamp}.csv`（例如：`2024.01.15.14.30.25.csv`）
- **日志**：`./exp/navsim_eval/logs/`（每个 worker 的日志）

---

## 五、运行时间估算

### 影响因素

1. **场景数量**：取决于测试集大小（navtest split）
2. **并行度**：Ray worker 数量（取决于 GPU 数量和配置）
3. **模型推理速度**：每个场景约 0.1-0.5 秒（取决于 GPU）
4. **PDM 评分**：每个场景约 0.01-0.1 秒

### 估算示例

假设：
- 1000 个场景
- 4 个 GPU worker
- 每个场景推理时间 0.2 秒

**总时间** ≈ 1000 / 4 × 0.2 ≈ **50 秒**

**实际时间可能更长**（数据加载、I/O、初始化等）

---

## 六、关键参数

### Agent 参数

- `condition_frames=3`：使用 3 帧历史图像
- `use_sde=false`：ODE 采样，生成 1 条轨迹
- `use_sde=true`：SDE 采样，生成 8 条轨迹（但评估时只用第一条）

### 评估参数

- `proposal_sampling`：轨迹采样参数
  - `num_poses=40`：40 个轨迹点
  - `interval_length=0.1`：0.1 秒间隔
  - `time_horizon=4.0`：4 秒时间范围

### 评分权重

- `progress_weight=5.0`：前进距离权重
- `ttc_weight=5.0`：碰撞时间权重
- `comfortable_weight=2.0`：舒适度权重
- `driving_direction_weight=0.0`：行驶方向权重（当前为0）

---

## 七、注意事项

### 1. 这不是训练过程

- **只进行推理**，不更新模型参数
- 模型处于 `eval()` 模式
- 使用 `torch.no_grad()` 禁用梯度计算

### 2. Metric Cache 需要预先计算

- Metric cache 包含场景的 ground truth 和地图信息
- 需要提前运行 metric caching 脚本生成
- 如果某个场景没有 cache，会被跳过

### 3. GPU 必需

- 模型推理需要 GPU（CUDA）
- 每个 Ray worker 分配 1 个 GPU
- 如果 GPU 不足，worker 会等待

### 4. 并行处理

- 使用 Ray 在多个 GPU 上并行评估
- 场景被分配到不同的 worker 处理
- 结果最后汇总

### 5. 错误处理

- 如果某个场景处理失败，会标记 `valid=False`
- 失败的场景不会影响其他场景的处理
- 错误信息会记录在日志中

---

## 八、使用示例

### 基本运行

```bash
bash run/run_navsim_eval_pipeline.sh
```

### 自定义参数

```bash
python3 scripts/test/run_navsim_eval_pipeline.py \
  +agent.config_path=configs/dit_config_dcae_nuplan.py \
  +agent.checkpoint_path=/path/to/checkpoint.pkl \
  +agent.condition_frames=3 \
  +agent.use_sde=false \
  experiment_name=epona_eval \
  output_dir=./exp/navsim_eval \
  metric_cache_path=/path/to/metric_cache \
  worker=ray_distributed
```

### 使用 SDE 采样

```bash
# 修改 use_sde=true 可以生成多条轨迹（但评估时只用第一条）
+agent.use_sde=true
```

---

## 九、评估指标说明

### PDM Score（综合得分）

PDM Score 是加权综合得分，计算公式：

```
score = progress_weight × ego_progress
      + ttc_weight × time_to_collision_within_bound
      + comfortable_weight × comfort
      + driving_direction_weight × driving_direction_compliance
```

### 各子指标含义

1. **no_at_fault_collisions**：无碰撞标志
   - 1：无碰撞
   - 0：发生碰撞

2. **drivable_area_compliance**：可行驶区域合规性
   - 轨迹是否在可行驶区域内

3. **ego_progress**：前进距离
   - 车辆沿路径前进的距离（米）

4. **time_to_collision_within_bound**：碰撞时间
   - 如果预测到碰撞，计算碰撞时间
   - 值越大越好

5. **comfort**：舒适度
   - 基于加速度和转向的舒适度评分

6. **driving_direction_compliance**：行驶方向合规性
   - 轨迹是否符合正确的行驶方向

---

## 十、故障排查

### 常见错误

1. **ModuleNotFoundError: No module named 'navsim'**
   - 检查路径设置，确保项目根目录在 `sys.path` 中

2. **AttributeError: 'ConfigDict' object has no attribute 'batch_size'**
   - 已在 `epona_agent.py` 中设置默认值 `batch_size=1`

3. **RuntimeError: No CUDA GPUs are available**
   - 检查 GPU 是否可用
   - 确保使用 `worker=ray_distributed`（支持 GPU）

4. **KeyError: 'batch_size'**
   - 确保配置文件中包含必要参数，或代码中设置了默认值

### 调试建议

1. 设置 `HYDRA_FULL_ERROR=1` 查看完整错误堆栈
2. 检查日志文件：`./exp/navsim_eval/logs/`
3. 先用单个场景测试：修改 `scene_filter` 限制场景数量

---

## 十一、相关文件

- **评估脚本**：`scripts/test/run_navsim_eval_pipeline.py`
- **启动脚本**：`run/run_navsim_eval_pipeline.sh`
- **Agent 实现**：`navsim/agents/epona/epona_agent.py`
- **配置目录**：`navsim/planning/script/config/pdm_scoring/`
- **评分函数**：`navsim/evaluate/pdm_score.py`

---

## 十二、总结

`run_navsim_eval_pipeline.py` 是一个**端到端的评估流程**：

1. **输入**：测试场景 + 预计算的 metric cache + 训练好的模型
2. **处理**：模型推理生成轨迹 → PDM 仿真 → 多指标评分
3. **输出**：CSV 格式的评估结果，包含每个场景的详细指标和综合得分

整个过程是**并行化**的，使用 Ray 在多个 GPU 上同时处理多个场景，提高评估效率。
