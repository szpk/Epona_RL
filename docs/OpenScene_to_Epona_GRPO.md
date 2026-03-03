# OpenScene 数据格式转换与应用 Epona GRPO 训练指南

## 概述

本文档详细说明如何将 OpenScene 数据集（pkl 格式）转换为 Epona GRPO 训练所需的数据格式。OpenScene 数据是 recogdrive 使用的标准数据格式，通过统一数据源确保训练和评估的一致性。

## 1. OpenScene 数据格式

### 1.1 数据结构

OpenScene 数据以 **pickle (pkl) 格式**存储，主要包含以下部分：

```
openscene-v1.1/
├── meta_datas/
│   ├── trainval/          # 训练/验证场景元数据
│   │   ├── log1.pkl
│   │   ├── log2.pkl
│   │   └── ...
│   └── test/              # 测试场景元数据
│       └── ...
└── sensor_blobs/
    ├── trainval/          # 训练/验证传感器数据（图像、点云等）
    │   ├── log1/
    │   │   ├── CAM_F0/
    │   │   ├── CAM_L0/
    │   │   └── ...
    │   └── ...
    └── test/
        └── ...
```

### 1.2 Pickle 文件内容

每个 `.pkl` 文件包含一个**字典列表**，每个字典代表一个时间步（frame）：

```python
scene_dict_list = [
    {
        "log_name": "2021.05.12.19.36.12_veh-35_00005_00204",
        "token": "1aa44d46e4ab5bc7",  # 唯一标识符（hash）
        "timestamp": 1620844572000000,
        "ego2global_translation": [x, y, z],
        "ego2global_rotation": [w, x, y, z],  # quaternion
        "ego_dynamic_state": [vx, vy, ax, ay],
        "driving_command": [0, 1, 0],  # one-hot
        "cams": {
            "CAM_F0": {
                "data_path": "log_name/CAM_F0/image_001.jpg",
                "sensor2lidar_rotation": [[...], [...], [...]],
                "sensor2lidar_translation": [x, y, z],
                "cam_intrinsic": [[...], [...], [...]],
                "distortion": [...]
            },
            # ... 其他相机
        },
        "lidar_path": "log_name/lidar_001.pcd",
        "roadblock_ids": [...],  # 用于路由过滤
        # ... 其他字段
    },
    # ... 更多帧
]
```

### 1.3 关键字段说明

- **`log_name`**: 日志名称，唯一标识一个驾驶场景
- **`token`**: 帧的唯一标识符（hash），用于 metric cache 查找
- **`cams`**: 相机数据字典，包含图像路径和标定参数
- **`ego2global_*`**: 车辆在全局坐标系中的位姿
- **`ego_dynamic_state`**: 车辆动态状态（速度、加速度）

## 2. 数据加载流程

### 2.1 SceneLoader 初始化

Epona 使用 recogdrive 的 `SceneLoader` 来加载 OpenScene 数据：

```python
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

# 设置路径
data_path = openscene_root / "meta_datas" / "trainval"
sensor_blobs_path = openscene_root / "sensor_blobs" / "trainval"

# 配置场景过滤器
scene_filter = SceneFilter(
    num_history_frames=condition_frames * downsample,  # 历史帧数（原始帧）
    num_future_frames=block_size * downsample,         # 未来帧数（原始帧）
    frame_interval=1,
    has_route=True,
    max_scenes=None,
    log_names=None,
    tokens=None,
)

# 配置传感器（只加载前向相机）
sensor_config = SensorConfig.build_no_sensors()
sensor_config.cameras = ['cam_f0']

# 创建 SceneLoader
scene_loader = SceneLoader(
    data_path=data_path,
    sensor_blobs_path=sensor_blobs_path,
    scene_filter=scene_filter,
    sensor_config=sensor_config,
    load_image_path=False,  # 直接加载图像为 numpy 数组
)
```

### 2.2 SceneLoader 工作原理

1. **扫描 pkl 文件**: 遍历 `data_path` 目录下的所有 `.pkl` 文件
2. **过滤场景**: 根据 `SceneFilter` 配置过滤场景
   - 检查场景长度是否足够
   - 检查是否有路由信息（`has_route=True`）
   - 按 `frame_interval` 采样帧
3. **加载传感器数据**: 
   - 如果 `load_image_path=False`: 直接从 `sensor_blobs_path` 加载图像为 numpy 数组
   - 如果 `load_image_path=True`: 只存储图像路径
4. **生成 tokens**: 为每个场景生成唯一 token: `{log_name}_{initial_token}`

### 2.3 数据转换过程

`SceneLoader` 将 pkl 字典列表转换为 `Scene` 对象：

```python
# Scene 对象结构
scene = Scene(
    frames=[
        Frame(
            token="1aa44d46e4ab5bc7",
            ego_status=EgoStatus(
                ego_pose=[x, y, yaw],  # 局部坐标系
                ego_velocity=[vx, vy],
                ego_acceleration=[ax, ay],
                driving_command=[0, 1, 0],
            ),
            cameras=Cameras(
                cam_f0=Camera(
                    image=np.array(...),  # (H, W, 3) uint8 图像
                    sensor2lidar_rotation=...,
                    sensor2lidar_translation=...,
                    intrinsics=...,
                    distortion=...,
                ),
                # ... 其他相机（空）
            ),
            # ... 其他字段
        ),
        # ... 更多帧
    ],
    scene_metadata=SceneMetadata(
        log_name="2021.05.12.19.36.12_veh-35_00005_00204",
        initial_token="1aa44d46e4ab5bc7",
        num_history_frames=8,
        num_future_frames=1,
    ),
)
```

## 3. Epona Dataset 实现

### 3.1 NuPlanOpenScene 类

`NuPlanOpenScene` 类封装了从 OpenScene 到 Epona 格式的转换：

```python
class NuPlanOpenScene(Dataset):
    def __init__(self, openscene_root, split='train', ...):
        # 初始化 SceneLoader
        self.scene_loader = SceneLoader(...)
        self.tokens = self.scene_loader.tokens  # 获取所有场景 tokens
        
    def __getitem__(self, index):
        # 1. 获取 token
        token = self.tokens[index]
        
        # 2. 从 SceneLoader 加载场景
        scene = self.scene_loader.get_scene_from_token(token)
        
        # 3. 提取图像和位姿
        frames = scene.frames
        imgs = []
        poses = []
        
        # 4. 下采样帧（根据 downsample_fps）
        frame_indices = list(range(0, len(frames), self.downsample))
        frame_indices = frame_indices[:self.condition_frames + self.block_size]
        
        for frame_idx in frame_indices:
            frame = frames[frame_idx]
            
            # 提取图像（已经是 numpy 数组）
            img_array = frame.cameras.cam_f0.image.copy()  # (H, W, 3)
            # 调整大小和格式
            img = Image.fromarray(img_array)
            img = img.resize((self.w, self.h), Image.BICUBIC)
            imgs.append(np.array(img))
            
            # 提取位姿（转换为 4x4 变换矩阵）
            ego_status = frame.ego_status
            pose = self.get_pose_from_ego_status(ego_status)  # (4, 4)
            poses.append(pose)
        
        # 5. 转换为 tensor
        imgs = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in imgs])
        rot_matrix = torch.stack([torch.from_numpy(pose) for pose in poses])
        
        # 6. 归一化图像
        imgs = self.normalize_imgs(imgs)  # 归一化到 [-1, 1]
        
        # 7. 生成 token（用于 metric cache 查找）
        log_name = scene.scene_metadata.log_name
        initial_token = scene.scene_metadata.initial_token
        token_for_cache = f"{log_name}_{initial_token}"
        
        return imgs, rot_matrix, token_for_cache
```

### 3.2 关键转换步骤

#### 步骤 1: 图像处理
```python
# OpenScene 图像格式: (H_orig, W_orig, 3) uint8
img_array = frame.cameras.cam_f0.image  # numpy array

# 转换为 PIL Image 并调整大小
img = Image.fromarray(img_array)
img = img.resize((w, h), Image.BICUBIC)  # (h, w, 3)

# 转换为 tensor 并归一化
img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # (3, h, w)
img_normalized = (img_tensor / 255.0 - 0.5) * 2  # 归一化到 [-1, 1]
```

#### 步骤 2: 位姿转换
```python
# OpenScene 位姿格式: ego_status.ego_pose = [x, y, yaw] (局部坐标系)
def get_pose_from_ego_status(self, ego_status):
    x, y, yaw = ego_status.ego_pose[0], ego_status.ego_pose[1], ego_status.ego_pose[2]
    
    # 构建 4x4 变换矩阵
    pose_matrix = np.eye(4, dtype=np.float32)
    pose_matrix[0, 3] = x
    pose_matrix[1, 3] = y
    pose_matrix[2, 3] = 0.0
    
    # 旋转（绕 z 轴）
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    pose_matrix[0, 0] = cos_yaw
    pose_matrix[0, 1] = -sin_yaw
    pose_matrix[1, 0] = sin_yaw
    pose_matrix[1, 1] = cos_yaw
    
    return pose_matrix  # (4, 4)
```

#### 步骤 3: Token 生成
```python
# Token 格式必须与 metric cache 匹配
log_name = scene.scene_metadata.log_name  # "2021.05.12.19.36.12_veh-35_00005_00204"
initial_token = scene.scene_metadata.initial_token  # "1aa44d46e4ab5bc7"
token_for_cache = f"{log_name}_{initial_token}"
# 结果: "2021.05.12.19.36.12_veh-35_00005_00204_1aa44d46e4ab5bc7"
```

## 4. 数据流完整流程

### 4.1 从 pkl 到训练数据的完整流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. OpenScene pkl 文件                                        │
│    meta_datas/trainval/log1.pkl                             │
│    └─> [dict1, dict2, ..., dictN]                           │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SceneLoader.filter_scenes()                              │
│    - 读取 pkl 文件                                           │
│    - 按 SceneFilter 过滤场景                                 │
│    - 生成 tokens: {log_name}_{initial_token}                │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. SceneLoader.get_scene_from_token(token)                  │
│    - 加载传感器数据（图像、点云等）                           │
│    - 转换为 Scene 对象                                       │
│    - Scene.frames: [Frame1, Frame2, ..., FrameN]            │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. NuPlanOpenScene.__getitem__(index)                       │
│    - 从 Scene 提取图像和位姿                                 │
│    - 下采样帧（根据 downsample_fps）                         │
│    - 转换为 Epona 格式                                      │
│    - 返回: (imgs, rot_matrix, token)                        │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. DataLoader 批处理                                        │
│    - 堆叠多个样本                                            │
│    - imgs: (B, F, C, H, W)                                  │
│    - rot_matrix: (B, F, 4, 4)                               │
│    - tokens_list: [token1, token2, ..., tokenB]             │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. 训练循环                                                  │
│    - 编码图像为 latents                                      │
│    - 提取位姿特征                                            │
│    - 使用 tokens_list 从 metric cache 获取奖励               │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 维度变化示例

假设配置：
- `batch_size = 2`
- `condition_frames = 8` (经过 downsample 后)
- `downsample_fps = 5` (原始 10Hz → 5Hz)
- `image_size = (512, 1024)`

```
单个样本 (Dataset.__getitem__):
  imgs: (8, 3, 512, 1024)      # 8 帧图像
  rot_matrix: (8, 4, 4)        # 8 个位姿矩阵
  token: "log_name_initial_token"

批处理 (DataLoader):
  imgs: (2, 8, 3, 512, 1024)   # 2 个样本，每个 8 帧
  rot_matrix: (2, 8, 4, 4)
  tokens_list: ["token1", "token2"]

编码后 (训练):
  latents: (2, 8, L, C)         # L=latent tokens, C=feature dim
  stt_features: (16, L, C)      # 2*8=16 (B*F)
```

## 5. Token 与 Metric Cache 的关联

### 5.1 Token 格式要求

Token 格式必须与 recogdrive 生成的 metric cache 完全匹配：

```python
# recogdrive 生成 metric cache 时的 token 格式
token = f"{log_name}_{initial_token}"

# Epona 生成 token 时使用相同格式
log_name = scene.scene_metadata.log_name
initial_token = scene.scene_metadata.initial_token
token_for_cache = f"{log_name}_{initial_token}"
```

### 5.2 Metric Cache 查找

在 GRPO 训练中，使用 token 从 metric cache 获取奖励：

```python
# 在 model.py 的 reward_fn 中
for i, token in enumerate(tokens_list):
    # 使用 token 从 metric cache 加载数据
    metric_cache = self.metric_cache_loader.get_from_token(token)
    
    # 计算 PDM 分数
    pdm_result = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=trajectory,
        ...
    )
    rewards.append(pdm_result.score)
```

### 5.3 数据一致性保证

通过使用相同的 OpenScene 数据源和相同的 token 格式，确保：

1. **训练数据**：来自 OpenScene pkl 文件
2. **评估数据**：metric cache 也基于相同的 OpenScene 数据生成
3. **Token 匹配**：训练和评估使用相同的 token 格式

这保证了训练时的奖励计算与评估时的指标计算使用相同的数据，避免了数据不一致导致的训练问题。

## 6. 关键配置参数

### 6.1 SceneFilter 参数

```python
scene_filter = SceneFilter(
    num_history_frames=condition_frames * downsample,  # 注意：原始帧数
    num_future_frames=block_size * downsample,         # 注意：原始帧数
    frame_interval=1,                                   # 帧间隔
    has_route=True,                                     # 必须有路由
    max_scenes=None,                                    # 最大场景数（None=全部）
    log_names=None,                                     # 指定日志名称（None=全部）
    tokens=None,                                        # 指定 tokens（None=全部）
)
```

**重要**: `num_history_frames` 和 `num_future_frames` 需要乘以 `downsample` 因子，因为：
- `condition_frames` 是经过 downsample 后的帧数
- SceneLoader 需要加载原始帧，然后由 Dataset 进行 downsample

### 6.2 图像加载模式

```python
load_image_path=False  # 直接加载为 numpy 数组（推荐）
# 或
load_image_path=True   # 只存储路径，需要手动加载
```

**推荐使用 `False`**，因为：
- 与 recogdrive 训练方式一致
- 避免路径处理错误
- 性能更好（一次性加载）

## 7. 常见问题与解决方案

### 7.1 图像加载失败

**问题**: `Warning: No valid image for frame X`

**原因**: 
- 图像路径不正确
- 图像文件不存在
- 相机数据格式不匹配

**解决**:
- 检查 `sensor_blobs_path` 是否正确
- 确认 `load_image_path=False`（直接加载图像）
- 检查 OpenScene 数据是否完整

### 7.2 Token 不匹配

**问题**: `Warning: PDM scoring failed for token ...`

**原因**:
- Token 格式不匹配
- Metric cache 中不存在该 token

**解决**:
- 确认 token 格式为 `{log_name}_{initial_token}`
- 检查 metric cache 是否包含对应的 token
- 确认训练和评估使用相同的数据源

### 7.3 帧数不匹配

**问题**: `Batch size mismatch: stt_features.shape[0] is not divisible by tokens_list length`

**原因**:
- `stt_features` 形状是 `(B_orig * F, ...)`
- `tokens_list` 长度是 `B_orig`
- 如果 `B_orig * F` 不能被 `B_orig` 整除，说明数据格式错误

**解决**:
- 检查 Dataset 返回的格式是否正确
- 确认 DataLoader 的 collate_fn 是否正确处理 tokens_list
- 检查 `condition_frames` 和 `downsample_fps` 配置

## 8. 总结

将 OpenScene pkl 数据应用到 Epona GRPO 训练的关键步骤：

1. **使用 SceneLoader**: 通过 recogdrive 的 `SceneLoader` 加载 pkl 文件
2. **转换数据格式**: 将 `Scene` 对象转换为 Epona 需要的 tensor 格式
3. **生成匹配的 Token**: 使用 `{log_name}_{initial_token}` 格式
4. **确保数据一致性**: 训练和评估使用相同的数据源和 token 格式

通过这种方式，Epona 的 GRPO 训练可以与 recogdrive 使用完全相同的数据，确保训练和评估的一致性。
