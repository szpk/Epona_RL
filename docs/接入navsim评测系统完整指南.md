# 接入NavSim评测系统完整指南

本文档详细说明如何将自己的模型接入NavSim评测系统，包括需要编写的文件、需要修改的配置参数，以及完整的实现步骤。

---

## 一、核心文件结构

将navsim评测系统迁移到自己的模型仓库后，需要创建以下文件结构：

```
your_model_repo/
├── navsim/                          # navsim评测系统代码（从Epona仓库复制）
│   ├── agents/
│   │   ├── abstract_agent.py        # 抽象基类（已存在）
│   │   └── your_model/              # 【需要创建】你的模型agent目录
│   │       ├── __init__.py
│   │       └── your_model_agent.py  # 【需要创建】你的agent实现
│   ├── common/
│   │   └── dataclasses.py           # 数据类定义（已存在）
│   ├── planning/
│   │   └── script/
│   │       └── config/
│   │           ├── common/
│   │           │   └── agent/
│   │           │       └── your_model_agent.yaml  # 【需要创建】agent配置文件
│   │           └── pdm_scoring/
│   │               └── default_run_pdm_score.yaml # 评测配置（已存在）
│   └── evaluate/
│       └── pdm_score.py             # PDM评分函数（已存在）
├── scripts/
│   └── test/
│       └── run_navsim_eval_pipeline.py  # 评测脚本（已存在，可能需要修改）
├── run/
│   └── run_navsim_eval_pipeline.sh     # 【需要创建】评测启动脚本
└── configs/
    └── your_model_config.py            # 你的模型配置文件（已存在）
```

---

## 二、必须实现的文件

### 1. Agent类实现（核心文件）

**文件路径**：`navsim/agents/your_model/your_model_agent.py`

这是最核心的文件，需要实现一个继承自`AbstractAgent`的类。

#### 2.1 必须实现的方法

```python
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
import torch
import numpy as np

class YourModelAgent(AbstractAgent):
    """
    你的模型Agent实现
    """
    
    def __init__(
        self,
        config_path: str,              # 模型配置文件路径
        checkpoint_path: str,          # 模型检查点路径
        # 其他模型特定参数...
    ):
        super().__init__(requires_scene=False)  # 如果不需要地图信息，设为False
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        # 初始化模型相关变量
        self._model = None
        self._initialized = False
    
    def name(self) -> str:
        """返回agent名称"""
        return "your_model_agent"
    
    def get_sensor_config(self) -> SensorConfig:
        """
        定义需要加载的传感器类型
        返回：SensorConfig对象，指定需要哪些摄像头和LiDAR
        """
        return SensorConfig(
            cam_f0=True,      # 前置摄像头（通常需要）
            cam_l0=False,    # 左侧摄像头
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,    # 右侧摄像头
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,    # 后置摄像头
            lidar_pc=False,  # LiDAR点云
        )
        # 或者使用便捷方法：
        # return SensorConfig.build_all_sensors(include=True)  # 加载所有传感器
        # return SensorConfig.build_no_sensors()  # 不加载传感器
    
    def initialize(self) -> None:
        """
        初始化模型（加载权重、设置设备等）
        这个方法会在评测开始前被调用一次
        """
        if self._initialized:
            return
        
        # 1. 加载模型配置
        # 根据你的配置系统调整
        cfg = load_your_config(self.config_path)
        
        # 2. 初始化模型
        self._model = YourModel(cfg)
        
        # 3. 加载检查点
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self._model.load_state_dict(checkpoint['model_state_dict'])
        
        # 4. 移动到GPU并设置为评估模式
        self._model = self._model.cuda()
        self._model.eval()
        
        self._initialized = True
    
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        核心方法：根据输入计算轨迹
        
        :param agent_input: AgentInput对象，包含：
            - ego_statuses: List[EgoStatus]，历史帧的ego状态
                - ego_pose: np.array([x, y, heading])，相对坐标系下的位置和朝向
                - ego_velocity: np.array([vx, vy])，速度
                - ego_acceleration: np.array([ax, ay])，加速度
                - driving_command: np.array，驾驶命令
            - cameras: List[Cameras]，历史帧的摄像头数据
                - cam_f0.image: np.array，形状(H, W, 3)，RGB图像，numpy数组
                - 其他摄像头类似
            - lidars: List[Lidar]，历史帧的LiDAR数据
                - lidar_pc: np.array，形状(6, N)，点云数据
        
        :return: Trajectory对象，包含：
            - poses: np.array，形状(num_poses, 3)，每行是[x, y, heading]
                - x, y: 相对坐标系下的位置（米）
                - heading: 朝向角度（弧度）
            - trajectory_sampling: TrajectorySampling对象，定义轨迹的时间参数
        """
        if not self._initialized:
            self.initialize()
        
        # ========== 步骤1：提取和处理输入数据 ==========
        
        # 1.1 提取图像数据
        images = []
        num_history_frames = len(agent_input.cameras)
        # 通常使用最后几帧作为输入
        condition_frames = 3  # 根据你的模型调整
        start_idx = max(0, num_history_frames - condition_frames)
        
        for i in range(start_idx, num_history_frames):
            camera = agent_input.cameras[i]
            if camera.cam_f0.image is not None:
                # 图像是numpy数组，形状(H, W, 3)，RGB格式
                img = camera.cam_f0.image.copy()
                # 转换为tensor并预处理（根据你的模型调整）
                # img_tensor = your_preprocess(img)  # 例如：resize, normalize等
                images.append(img)
        
        # 如果帧数不足，进行padding
        while len(images) < condition_frames:
            if images:
                images.insert(0, images[0])  # 重复第一帧
            else:
                # 创建黑色图像
                images.append(np.zeros((H, W, 3), dtype=np.uint8))
        
        # 1.2 提取ego状态
        ego_statuses = agent_input.ego_statuses[start_idx:]
        # 提取位置、速度等信息
        poses = []
        for status in ego_statuses:
            x, y, heading = status.ego_pose  # 相对坐标系
            poses.append([x, y, heading])
        
        # ========== 步骤2：模型推理 ==========
        
        # 2.1 准备模型输入（根据你的模型调整）
        # 例如：将图像转换为tensor，stack成batch等
        # input_tensor = prepare_model_input(images, poses)
        
        # 2.2 模型前向传播
        with torch.no_grad():
            # 根据你的模型接口调用
            # output = self._model(input_tensor)
            # 或者
            # output = self._model.inference(input_tensor)
            pass
        
        # ========== 步骤3：后处理输出 ==========
        
        # 3.1 提取轨迹
        # 模型输出可能是各种格式，需要转换为标准格式
        # trajectory_poses = extract_trajectory(output)  # 形状：(num_poses, 3)
        
        # 示例：假设模型输出是相对坐标系的轨迹
        # trajectory_poses = output.cpu().numpy()  # 形状：(horizon, 3)
        
        # 3.2 确保轨迹格式正确
        # - 形状必须是 (num_poses, 3)
        # - 每行是 [x, y, heading]
        # - x, y: 相对位置（米）
        # - heading: 朝向角度（弧度）
        
        # 如果模型输出的是度数，需要转换为弧度
        # trajectory_poses[:, 2] = np.deg2rad(trajectory_poses[:, 2])
        
        # 3.3 创建TrajectorySampling对象
        # NavSim标准：4秒时间范围，0.5秒间隔，共8个点
        num_poses = trajectory_poses.shape[0]
        trajectory_sampling = TrajectorySampling(
            num_poses=num_poses,
            interval_length=0.5  # 0.5秒间隔
        )
        
        # 如果轨迹点数不匹配，需要resample
        if num_poses != 8:
            trajectory_poses = resample_trajectory(trajectory_poses, target_num_poses=8)
            trajectory_sampling = TrajectorySampling(
                num_poses=8,
                interval_length=0.5
            )
        
        # ========== 步骤4：返回Trajectory对象 ==========
        
        return Trajectory(
            poses=trajectory_poses.astype(np.float32),
            trajectory_sampling=trajectory_sampling
        )
```

#### 2.2 关键数据格式说明

**AgentInput结构**：
```python
@dataclass
class AgentInput:
    ego_statuses: List[EgoStatus]  # 历史帧的ego状态
    cameras: List[Cameras]         # 历史帧的摄像头数据
    lidars: List[Lidar]            # 历史帧的LiDAR数据

@dataclass
class EgoStatus:
    ego_pose: np.array([x, y, heading])      # 相对坐标系，heading是弧度
    ego_velocity: np.array([vx, vy])         # 速度（m/s）
    ego_acceleration: np.array([ax, ay])     # 加速度（m/s²）
    driving_command: np.array                # 驾驶命令

@dataclass
class Cameras:
    cam_f0: Camera  # 前置摄像头
    # cam_f0.image: np.array，形状(H, W, 3)，RGB格式，uint8类型

@dataclass
class Camera:
    image: np.array  # 形状(H, W, 3)，RGB格式，numpy数组（不是tensor）
```

**Trajectory输出格式**：
```python
@dataclass
class Trajectory:
    poses: np.array  # 形状(num_poses, 3)，每行是[x, y, heading]
                    # x, y: 相对位置（米）
                    # heading: 朝向角度（弧度）
    trajectory_sampling: TrajectorySampling  # 时间参数
```

**重要注意事项**：
1. **坐标系**：所有pose都是**相对坐标系**（相对于最后一帧的ego位置）
2. **角度单位**：heading必须是**弧度**，不是度数
3. **图像格式**：`camera.cam_f0.image`是numpy数组，形状(H, W, 3)，RGB格式，uint8类型
4. **轨迹点数**：标准是8个点（4秒，0.5秒间隔），但可以不同，系统会自动处理

---

### 2. Agent配置文件（YAML）

**文件路径**：`navsim/planning/script/config/common/agent/your_model_agent.yaml`

```yaml
_target_: navsim.agents.your_model.your_model_agent.YourModelAgent
_convert_: 'all'
config_path: ${oc.env:YOUR_MODEL_ROOT}/configs/your_model_config.py
checkpoint_path: ${oc.env:YOUR_MODEL_ROOT}/checkpoints/your_model.pkl
# 其他模型特定参数...
```

**说明**：
- `_target_`：指向你的Agent类的完整路径（模块路径.类名）
- `_convert_`：Hydra配置转换选项，通常设为`'all'`
- 其他参数：你的Agent的`__init__`方法需要的参数

**如果使用命令行参数覆盖**（推荐方式）：
```yaml
# 配置文件可以留空或设置默认值
# 实际参数通过命令行传入：
# +agent.config_path=configs/your_model_config.py
# +agent.checkpoint_path=checkpoints/your_model.pkl
```

---

### 3. 评测启动脚本（Shell脚本）

**文件路径**：`run/run_navsim_eval_pipeline.sh`

```bash
#!/bin/bash

# ========== 配置参数 ==========
NUM_GPUS=1                    # 使用的GPU数量
DATA_SPLIT=test               # 数据集划分：test, navtest, navtrain等
MAX_SCENARIOS=200             # 最大评估场景数（0表示全部）

# ========== 路径配置 ==========
# 切换到项目根目录
cd /path/to/your_model_repo

# ========== 环境变量配置（必须） ==========
export HYDRA_FULL_ERROR=1
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/nuplan/dataset/maps"           # NuPlan地图路径
export NAVSIM_EXP_ROOT="/path/to/your_model_repo/exp"            # 评测结果输出根目录
export NAVSIM_DEVKIT_ROOT="/path/to/your_model_repo"            # NavSim代码根目录
export OPENSCENE_DATA_ROOT="/path/to/openscene-v1.1"            # OpenScene数据根目录

# ========== Metric Cache路径配置 ==========
# Metric Cache是预计算的场景数据，必须提前生成
if [[ "$DATA_SPLIT" == "test" || "$DATA_SPLIT" == "navtest" ]]; then
    METRIC_CACHE_PATH="/path/to/metric_cache/test"              # 测试集metric cache
else
    METRIC_CACHE_PATH="/path/to/metric_cache/train"             # 训练集metric cache
fi

# ========== 运行评测 ==========
python3 scripts/test/run_navsim_eval_pipeline.py \
  train_test_split=${DATA_SPLIT} \
  +agent.config_path=configs/your_model_config.py \
  +agent.checkpoint_path=/path/to/checkpoints/your_model.pkl \
  +agent.other_param=value \
  experiment_name=your_model_eval \
  output_dir=./exp/navsim_eval \
  metric_cache_path=${METRIC_CACHE_PATH} \
  worker=ray_distributed \
  +worker.threads_per_node=${NUM_GPUS} \
  +num_gpus=${NUM_GPUS} \
  +max_scenarios=${MAX_SCENARIOS}
```

**关键环境变量说明**：
- `NUPLAN_MAPS_ROOT`：NuPlan地图数据路径
- `NAVSIM_EXP_ROOT`：评测结果输出根目录
- `NAVSIM_DEVKIT_ROOT`：NavSim代码根目录（通常是项目根目录）
- `OPENSCENE_DATA_ROOT`：OpenScene数据集根目录
- `METRIC_CACHE_PATH`：预计算的metric cache路径（必须提前生成）

---

## 三、需要修改的配置参数

### 1. 评测脚本中的Agent导入（如果需要）

**文件**：`scripts/test/run_navsim_eval_pipeline.py`

如果你的Agent需要特殊初始化（类似EponaAgent），可能需要修改评测脚本：

```python
# 在文件开头添加导入
from navsim.agents.your_model.your_model_agent import YourModelAgent

# 在run_pdm_score函数中，可能需要特殊处理
if hasattr(cfg.agent, "config_path"):
    # 你的Agent特殊初始化逻辑
    agent = YourModelAgent(
        config_path=cfg.agent.config_path,
        checkpoint_path=cfg.agent.get("checkpoint_path", None),
        # 其他参数...
    )
else:
    # 使用Hydra自动实例化（标准方式）
    agent = instantiate(cfg.agent)
```

**注意**：大多数情况下，如果Agent实现正确，不需要修改评测脚本，Hydra会自动实例化。

---

### 2. 数据路径配置

确保以下路径正确配置：

#### 2.1 环境变量
```bash
export OPENSCENE_DATA_ROOT="/path/to/openscene-v1.1"
export NUPLAN_MAPS_ROOT="/path/to/nuplan/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/your_model_repo/exp"
export NAVSIM_DEVKIT_ROOT="/path/to/your_model_repo"
```

#### 2.2 数据目录结构
```
${OPENSCENE_DATA_ROOT}/
├── navsim_logs/
│   ├── test/          # 测试集场景标注（.pkl文件）
│   └── train/         # 训练集场景标注
└── sensor_blobs/
    ├── test/          # 测试集传感器数据（图像、LiDAR等）
    └── train/         # 训练集传感器数据
```

#### 2.3 Metric Cache路径
```
/path/to/metric_cache/
├── test/              # 测试集metric cache
│   ├── token1.pkl.xz
│   ├── token2.pkl.xz
│   └── ...
└── train/             # 训练集metric cache
    └── ...
```

**注意**：Metric Cache必须提前生成，可以使用NavSim提供的metric caching脚本。

---

### 3. 评测配置参数

**文件**：`navsim/planning/script/config/pdm_scoring/default_run_pdm_score.yaml`

通常不需要修改，但如果需要自定义，可以修改以下参数：

```yaml
# 默认配置
defaults:
  - default_common
  - default_evaluation
  - default_scoring_parameters
  - agent: your_model_agent  # 改为你的agent配置
  - _self_
  - override train_test_split: navtest  # 数据集划分

metric_cache_path: ${oc.env:NAVSIM_EXP_ROOT}/metric_cache
```

---

## 四、完整实现示例

### 示例1：最简单的Agent实现

假设你的模型只需要前置摄像头图像：

```python
import torch
import numpy as np
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

class SimpleImageAgent(AbstractAgent):
    def __init__(self, checkpoint_path: str):
        super().__init__(requires_scene=False)
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._initialized = False
    
    def name(self) -> str:
        return "simple_image_agent"
    
    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(cam_f0=True)  # 只需要前置摄像头
    
    def initialize(self) -> None:
        if self._initialized:
            return
        # 加载模型
        self._model = YourModel()
        checkpoint = torch.load(self.checkpoint_path)
        self._model.load_state_dict(checkpoint)
        self._model = self._model.cuda().eval()
        self._initialized = True
    
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        if not self._initialized:
            self.initialize()
        
        # 提取最后一帧图像
        image = agent_input.cameras[-1].cam_f0.image  # (H, W, 3)
        
        # 预处理
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).cuda()
        
        # 模型推理
        with torch.no_grad():
            output = self._model(image_tensor)
        
        # 转换为numpy
        trajectory_poses = output[0].cpu().numpy()  # (8, 3)
        
        # 创建Trajectory
        trajectory_sampling = TrajectorySampling(
            num_poses=8,
            interval_length=0.5
        )
        return Trajectory(trajectory_poses.astype(np.float32), trajectory_sampling)
```

---

### 示例2：使用多帧历史信息的Agent

```python
class MultiFrameAgent(AbstractAgent):
    def __init__(self, config_path: str, checkpoint_path: str, num_frames: int = 3):
        super().__init__(requires_scene=False)
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.num_frames = num_frames
        self._model = None
        self._initialized = False
    
    def name(self) -> str:
        return "multi_frame_agent"
    
    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(cam_f0=True)
    
    def initialize(self) -> None:
        if self._initialized:
            return
        cfg = load_config(self.config_path)
        self._model = YourModel(cfg, num_frames=self.num_frames)
        checkpoint = torch.load(self.checkpoint_path)
        self._model.load_state_dict(checkpoint['model'])
        self._model = self._model.cuda().eval()
        self._initialized = True
    
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        if not self._initialized:
            self.initialize()
        
        # 提取多帧图像
        images = []
        num_history = len(agent_input.cameras)
        start_idx = max(0, num_history - self.num_frames)
        
        for i in range(start_idx, num_history):
            img = agent_input.cameras[i].cam_f0.image
            images.append(img)
        
        # Padding
        while len(images) < self.num_frames:
            images.insert(0, images[0] if images else np.zeros((256, 512, 3)))
        
        # 转换为tensor
        image_stack = np.stack(images)  # (T, H, W, 3)
        image_tensor = torch.from_numpy(image_stack).permute(0, 3, 1, 2).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).cuda()  # (1, T, C, H, W)
        
        # 提取ego状态
        poses = []
        for i in range(start_idx, num_history):
            x, y, heading = agent_input.ego_statuses[i].ego_pose
            poses.append([x, y, heading])
        
        while len(poses) < self.num_frames:
            poses.insert(0, poses[0] if poses else [0, 0, 0])
        
        pose_tensor = torch.from_numpy(np.array(poses)).float().unsqueeze(0).cuda()
        
        # 模型推理
        with torch.no_grad():
            output = self._model(image_tensor, pose_tensor)
        
        # 后处理
        trajectory_poses = output[0].cpu().numpy()  # (8, 3)
        
        # 确保heading是弧度
        if trajectory_poses[:, 2].max() > np.pi:
            trajectory_poses[:, 2] = np.deg2rad(trajectory_poses[:, 2])
        
        trajectory_sampling = TrajectorySampling(num_poses=8, interval_length=0.5)
        return Trajectory(trajectory_poses.astype(np.float32), trajectory_sampling)
```

---

## 五、常见问题和解决方案

### 1. 坐标系问题

**问题**：轨迹方向错误或位置不对

**解决**：
- 确保所有pose都是**相对坐标系**（相对于最后一帧ego位置）
- 确保heading是**弧度**，不是度数
- 检查模型输出的坐标系定义

### 2. 图像格式问题

**问题**：图像加载失败或格式不对

**解决**：
- `camera.cam_f0.image`是numpy数组，形状(H, W, 3)，RGB格式
- 需要转换为tensor：`torch.from_numpy(image).permute(2, 0, 1)`
- 注意数据类型：uint8需要归一化到[0, 1]或[-1, 1]

### 3. 轨迹点数不匹配

**问题**：轨迹点数不是8个

**解决**：
- 可以使用`TrajectorySampling`指定任意点数
- 或者resample到8个点：
```python
def resample_trajectory(poses, target_num_poses):
    if poses.shape[0] == target_num_poses:
        return poses
    src = np.linspace(0, 1, poses.shape[0])
    dst = np.linspace(0, 1, target_num_poses)
    x_interp = np.interp(dst, src, poses[:, 0])
    y_interp = np.interp(dst, src, poses[:, 1])
    yaw_unwrapped = np.unwrap(poses[:, 2])
    yaw_interp = np.interp(dst, src, yaw_unwrapped)
    yaw_interp = (yaw_interp + np.pi) % (2 * np.pi) - np.pi
    return np.stack([x_interp, y_interp, yaw_interp], axis=1)
```

### 4. 模型加载问题

**问题**：检查点加载失败

**解决**：
- 检查路径是否正确
- 检查checkpoint格式（可能是`checkpoint['model']`或`checkpoint['model_state_dict']`）
- 确保模型结构匹配

### 5. GPU内存不足

**问题**：CUDA out of memory

**解决**：
- 减少batch size（评测时通常是1）
- 使用`torch.no_grad()`禁用梯度
- 使用混合精度：`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`

### 6. 导入路径问题

**问题**：`ModuleNotFoundError: No module named 'navsim'`

**解决**：
- 确保项目根目录在`sys.path`中
- 检查`NAVSIM_DEVKIT_ROOT`环境变量
- 在Agent文件中添加路径：
```python
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
```

---

## 六、测试和验证

### 1. 单场景测试

在实现完整Agent后，建议先测试单个场景：

```python
# 测试脚本
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.agents.your_model.your_model_agent import YourModelAgent

# 初始化agent
agent = YourModelAgent(
    config_path="configs/your_model_config.py",
    checkpoint_path="checkpoints/your_model.pkl"
)
agent.initialize()

# 加载单个场景
scene_loader = SceneLoader(...)
token = "your_test_token"
agent_input = scene_loader.get_agent_input_from_token(token)

# 计算轨迹
trajectory = agent.compute_trajectory(agent_input)

# 检查输出
print(f"Trajectory shape: {trajectory.poses.shape}")  # 应该是 (8, 3)
print(f"Trajectory poses:\n{trajectory.poses}")
```

### 2. 完整评测运行

```bash
# 使用少量场景测试
bash run/run_navsim_eval_pipeline.sh
# 设置 MAX_SCENARIOS=10 先测试
```

---

## 七、检查清单

在提交评测前，确保以下项目都已完成：

- [ ] **Agent类实现**
  - [ ] 继承自`AbstractAgent`
  - [ ] 实现`name()`方法
  - [ ] 实现`get_sensor_config()`方法
  - [ ] 实现`initialize()`方法
  - [ ] 实现`compute_trajectory()`方法
  - [ ] 输出轨迹格式正确（形状、坐标系、角度单位）

- [ ] **配置文件**
  - [ ] 创建agent YAML配置文件
  - [ ] `_target_`路径正确
  - [ ] 参数配置正确

- [ ] **环境变量**
  - [ ] `OPENSCENE_DATA_ROOT`设置正确
  - [ ] `NUPLAN_MAPS_ROOT`设置正确
  - [ ] `NAVSIM_EXP_ROOT`设置正确
  - [ ] `NAVSIM_DEVKIT_ROOT`设置正确

- [ ] **数据路径**
  - [ ] navsim_logs路径存在
  - [ ] sensor_blobs路径存在
  - [ ] metric_cache路径存在且包含数据

- [ ] **启动脚本**
  - [ ] 创建评测启动脚本
  - [ ] 环境变量配置正确
  - [ ] 命令行参数正确

- [ ] **测试验证**
  - [ ] 单场景测试通过
  - [ ] 小批量场景测试通过
  - [ ] 输出CSV格式正确

---

## 八、参考实现

可以参考以下现有实现：

1. **EponaAgent**：`navsim/agents/epona/epona_agent.py`
   - 使用图像和位姿作为输入
   - 使用VAE编码器
   - 使用DiT模型生成轨迹

2. **ConstantVelocityAgent**：`navsim/agents/constant_velocity_agent.py`
   - 最简单的baseline实现
   - 只使用速度信息

3. **RecogDriveAgent**：`navsim/agents/recogdrive/recogdrive_agent.py`
   - 使用多模态输入（图像、LiDAR等）
   - 使用扩散模型

---

## 九、总结

接入NavSim评测系统的核心步骤：

1. **实现Agent类**：继承`AbstractAgent`，实现必要方法
2. **创建配置文件**：YAML格式的agent配置
3. **配置环境变量**：设置数据路径和输出路径
4. **创建启动脚本**：Shell脚本配置评测参数
5. **测试验证**：先测试单个场景，再运行完整评测

**关键点**：
- 输入：`AgentInput`包含历史帧的传感器数据和ego状态
- 输出：`Trajectory`包含相对坐标系的轨迹点
- 坐标系：所有pose都是相对坐标系，heading是弧度
- 轨迹格式：形状(num_poses, 3)，每行是[x, y, heading]

完成以上步骤后，你的模型就可以在NavSim评测系统上运行了！
