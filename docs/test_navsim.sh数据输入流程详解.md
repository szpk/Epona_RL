# test_navsim.sh 数据输入流程详解

本文档详细说明 `test_navsim.sh` 脚本的数据输入流程，包括数据来源、处理步骤和形状变化。

## 1. 脚本执行流程概览

`test_navsim.sh` 脚本的主要执行流程如下：

```
test_navsim.sh
  └─> run_navsim_eval_pipeline1.py
      └─> test_navsim_traj.py (推理)
          └─> SceneLoader (数据加载)
              └─> EponaAgent (模型推理)
```

## 2. 数据输入路径配置

在 `test_navsim.sh` 中，关键的数据路径配置如下：

```bash
--navsim_log_path /inspire/hdd/project/roboticsystem2/public/OpenScene/openscene-v1.1/navsim_logs/test
--sensor_blobs_path /inspire/hdd/project/roboticsystem2/public/OpenScene/openscene-v1.1/sensor_blobs/test
```

这些路径指向 OpenScene 数据集的测试集：
- `navsim_log_path`: 包含场景元数据的 pkl 文件目录
- `sensor_blobs_path`: 包含传感器数据（图像、点云等）的目录

## 3. 数据加载流程

### 3.1 场景过滤和Token收集

**代码位置**: `scripts/test/run_navsim_eval_pipeline1.py`

在 `run_navsim_eval_pipeline1.py` 的 `main()` 函数中，首先构建场景过滤器并收集 tokens：

```47:67:scripts/test/run_navsim_eval_pipeline1.py
def _build_scene_filter(cfg) -> SceneFilter:
    num_history_frames = cfg.navsim_history_frames or cfg.condition_frames
    required_future = int(round(cfg.output_time_horizon / cfg.output_interval))
    num_future_frames = max(cfg.navsim_future_frames or cfg.traj_len, required_future)
    
    navtest_config_path = ROOT / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    log_names, tokens = None, None
    if navtest_config_path.exists():
        with open(navtest_config_path, 'r') as f:
            navtest_cfg = yaml.safe_load(f)
        log_names = navtest_cfg.get('log_names')
        tokens = navtest_cfg.get('tokens')
    
    return SceneFilter(
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
        frame_interval=cfg.frame_interval,
        max_scenes=cfg.max_scenes,
        log_names=log_names,
        tokens=tokens,
    )
```

```70:80:scripts/test/run_navsim_eval_pipeline1.py
def _collect_tokens(cfg, scene_filter: SceneFilter) -> Tuple[List[str], int]:
    scene_loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    tokens = scene_loader.tokens
    start_id = cfg.start_id
    end_id = min(cfg.end_id, len(tokens))
    return tokens[start_id:end_id], len(tokens)
```

**数据形状变化**:
- 输入: 配置文件中的 `start_id=0`, `end_id=100`
- 输出: `tokens` 列表，长度为 `end_id - start_id`（最多100个token字符串）

### 3.2 SceneLoader 初始化

**代码位置**: `navsim/common/dataloader.py`

`SceneLoader` 在初始化时调用 `filter_scenes()` 函数加载场景数据：

```69:92:navsim/common/dataloader.py
class SceneLoader:
    """Simple data loader of scenes from logs."""

    def __init__(
        self,
        data_path: Path,
        sensor_blobs_path: Path,
        scene_filter: SceneFilter,
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
        load_image_path: bool = False
    ):
        """
        Initializes the scene data loader.
        :param data_path: root directory of log folder
        :param sensor_blobs_path: root directory of sensor data
        :param scene_filter: dataclass for scene filtering specification
        :param sensor_config: dataclass for sensor loading specification, defaults to no sensors
        """

        self.scene_frames_dicts = filter_scenes(data_path, scene_filter)
        self._sensor_blobs_path = sensor_blobs_path
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config
        self.load_image_path = load_image_path
```

`filter_scenes()` 函数从 pkl 文件中加载场景数据：

```14:66:navsim/common/dataloader.py
def filter_scenes(data_path: Path, scene_filter: SceneFilter) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load a set of scenes from dataset, while applying scene filter configuration.
    :param data_path: root directory of log folder
    :param scene_filter: scene filtering configuration class
    :return: dictionary of raw logs format
    """

    def split_list(input_list: List[Any], num_frames: int, frame_interval: int) -> List[List[Any]]:
        """Helper function to split frame list according to sampling specification."""
        return [input_list[i : i + num_frames] for i in range(0, len(input_list), frame_interval)]

    filtered_scenes: Dict[str, Scene] = {}
    stop_loading: bool = False

    # filter logs
    log_files = list(data_path.iterdir())
    if scene_filter.log_names is not None:
        log_files = [log_file for log_file in log_files if log_file.name.replace(".pkl", "") in scene_filter.log_names]

    if scene_filter.tokens is not None:
        filter_tokens = True
        tokens = set(scene_filter.tokens)
    else:
        filter_tokens = False

    for log_pickle_path in tqdm(log_files, desc="Loading logs"):

        scene_dict_list = pickle.load(open(log_pickle_path, "rb"))
        for frame_list in split_list(scene_dict_list, scene_filter.num_frames, scene_filter.frame_interval):
            # Filter scenes which are too short
            if len(frame_list) < scene_filter.num_frames:
                continue

            # Filter scenes with no route
            if scene_filter.has_route and len(frame_list[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0:
                continue

            # Filter by token
            token = frame_list[scene_filter.num_history_frames - 1]["token"]
            if filter_tokens and token not in tokens:
                continue

            filtered_scenes[token] = frame_list

            if (scene_filter.max_scenes is not None) and (len(filtered_scenes) >= scene_filter.max_scenes):
                stop_loading = True
                break

        if stop_loading:
            break

    return filtered_scenes
```

**数据形状变化**:
- 输入: pkl 文件，每个文件包含一个列表，列表元素是字典，每个字典代表一帧数据
- 处理: 根据 `num_frames` 和 `frame_interval` 分割帧列表
- 输出: `Dict[str, List[Dict]]`，键为 token，值为该场景的帧列表
  - 每个场景包含 `num_history_frames + num_future_frames` 帧
  - 默认情况下：`num_history_frames=4`, `num_future_frames=8`（根据配置计算）

### 3.3 AgentInput 构建

**代码位置**: `navsim/common/dataloader.py` 和 `navsim/common/dataclasses.py`

在 `test_navsim_traj.py` 中，通过 `get_agent_input_from_token()` 获取 AgentInput：

```144:147:scripts/test/test_navsim_traj.py
    for idx, token in tqdm(enumerate(tokens), total=len(tokens), desc="NAVSIM inference"):
        agent_input = scene_loader.get_agent_input_from_token(token)
        pred_traj = agent.compute_trajectory(agent_input)
        np.save(os.path.join(save_path, f"{token}.npy"), pred_traj.poses)
```

`get_agent_input_from_token()` 调用 `AgentInput.from_scene_dict_list()`：

```139:152:navsim/common/dataloader.py
    def get_agent_input_from_token(self, token: str) -> AgentInput:
        """
        Loads agent input given a scene identifier string (token).
        :param token: scene identifier string.
        :return: agent input dataclass
        """
        assert token in self.tokens
        return AgentInput.from_scene_dict_list(
            self.scene_frames_dicts[token],
            self._sensor_blobs_path,
            num_history_frames=self._scene_filter.num_history_frames,
            sensor_config=self._sensor_config,
            load_image_path=self.load_image_path
        )
```

`AgentInput.from_scene_dict_list()` 处理每一帧数据：

```161:226:navsim/common/dataclasses.py
    @classmethod
    def from_scene_dict_list(
        cls,
        scene_dict_list: List[Dict],
        sensor_blobs_path: Path,
        num_history_frames: int,
        sensor_config: SensorConfig,
        load_image_path: bool = False 
    ) -> AgentInput:
        """
        Load agent input from scene dictionary.
        :param scene_dict_list: list of scene frames (in logs).
        :param sensor_blobs_path: root directory of sensor data
        :param num_history_frames: number of agent input frames
        :param sensor_config: sensor config dataclass
        :return: agent input dataclass
        """
        assert len(scene_dict_list) > 0, "Scene list is empty!"

        global_ego_poses = []
        for frame_idx in range(num_history_frames):
            ego_translation = scene_dict_list[frame_idx]["ego2global_translation"]
            ego_quaternion = Quaternion(*scene_dict_list[frame_idx]["ego2global_rotation"])
            global_ego_pose = np.array(
                [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
                dtype=np.float64,
            )
            global_ego_poses.append(global_ego_pose)

        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[-1]), np.array(global_ego_poses, dtype=np.float64)
        )

        ego_statuses: List[EgoStatus] = []
        cameras: List[EgoStatus] = []
        lidars: List[Lidar] = []

        for frame_idx in range(num_history_frames):

            ego_dynamic_state = scene_dict_list[frame_idx]["ego_dynamic_state"]
            ego_status = EgoStatus(
                ego_pose=np.array(local_ego_poses[frame_idx], dtype=np.float32),
                ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
                ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
                driving_command=scene_dict_list[frame_idx]["driving_command"],
            )
            ego_statuses.append(ego_status)

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            cameras.append(
                Cameras.from_camera_dict(
                    sensor_blobs_path=sensor_blobs_path,
                    camera_dict=scene_dict_list[frame_idx]["cams"],
                    sensor_names=sensor_names,
                    load_image_path=load_image_path,
                )
            )

            lidars.append(
                Lidar.from_paths(
                    sensor_blobs_path=sensor_blobs_path,
                    lidar_path=Path(scene_dict_list[frame_idx]["lidar_path"]),
                    sensor_names=sensor_names,
                )
            )

        return AgentInput(ego_statuses, cameras, lidars)
```

**数据形状变化**:

1. **输入**: `scene_dict_list` - 列表，长度为 `num_history_frames`（默认4）
   - 每个元素是字典，包含：
     - `ego2global_translation`: `[x, y, z]` (float64)
     - `ego2global_rotation`: `[w, x, y, z]` (四元数)
     - `ego_dynamic_state`: `[vx, vy, ax, ay]` (速度+加速度)
     - `driving_command`: 字符串
     - `cams`: 相机数据字典
     - `lidar_path`: 点云文件路径

2. **全局到局部坐标转换**:
   - `global_ego_poses`: `(num_history_frames, 3)` - `[x, y, yaw]` (float64)
   - `local_ego_poses`: `(num_history_frames, 3)` - 相对于最后一帧的局部坐标 `[x, y, yaw]` (float64)

#### 2.1 `global_ego_poses` 如何构建（与代码逐行对应）

**代码位置**: `navsim/common/dataclasses.py` → `AgentInput.from_scene_dict_list()`

```179:192:navsim/common/dataclasses.py
        global_ego_poses = []
        for frame_idx in range(num_history_frames):
            ego_translation = scene_dict_list[frame_idx]["ego2global_translation"]
            ego_quaternion = Quaternion(*scene_dict_list[frame_idx]["ego2global_rotation"])
            global_ego_pose = np.array(
                [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
                dtype=np.float64,
            )
            global_ego_poses.append(global_ego_pose)

        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[-1]), np.array(global_ego_poses, dtype=np.float64)
        )
```

- **`ego2global_translation` → `x,y`**: 只取 `[0],[1]` 两维，`z` 不参与（这里是 2D 的 SE(2) 位姿）。
- **`ego2global_rotation`(四元数) → `yaw`**: `Quaternion(*[w,x,y,z]).yaw_pitch_roll[0]` 取 yaw（弧度）。
- **`origin` 选择**: `StateSE2(*global_ego_poses[-1])`，也就是**最后一帧（当前帧）**当作局部坐标原点。

#### 2.2 “相对于最后一帧”到底怎么换算（SE(2) 变换细节）

**代码位置**: `navsim/planning/simulation/planner/pdm_planner/utils/pdm_geometry_utils.py` → `convert_absolute_to_relative_se2_array()`

```76:96:navsim/planning/simulation/planner/pdm_planner/utils/pdm_geometry_utils.py
def convert_absolute_to_relative_se2_array(
    origin: StateSE2, state_se2_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    assert len(SE2Index) == state_se2_array.shape[-1]

    theta = -origin.heading
    origin_array = np.array([[origin.x, origin.y, origin.heading]], dtype=np.float64)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    points_rel = state_se2_array - origin_array
    points_rel[..., :2] = points_rel[..., :2] @ R.T
    points_rel[:, 2] = normalize_angle(points_rel[:, 2])

    return points_rel
```

逐步对应：

- **先平移到原点**: `points_rel = state_se2_array - origin_array`  
  得到 \((x - x_0,\; y - y_0,\; yaw - yaw_0)\)
- **再把坐标轴旋回到“以原点朝向为 x 轴”的局部系**: `theta=-yaw0`，对 \((x,y)\) 乘 `R.T`  
  等价于对平移后的点做 \(R(-yaw_0)\) 旋转
- **最后把角度归一化到 [-π, π]**: `normalize_angle = atan2(sin, cos)`

3. **输出**: `AgentInput` 对象，包含：
   - `ego_statuses`: 长度为 `num_history_frames` 的列表
     - 每个 `EgoStatus` 包含：
       - `ego_pose`: `(3,)` - `[x, y, yaw]` (float32)
       - `ego_velocity`: `(2,)` - `[vx, vy]` (float32)
       - `ego_acceleration`: `(2,)` - `[ax, ay]` (float32)
   - `cameras`: 长度为 `num_history_frames` 的列表
     - 每个 `Cameras` 包含 `cam_f0` (前视相机)
       - `cam_f0.image`: `(H, W, 3)` - uint8 图像数组（从 sensor_blobs_path 加载）
   - `lidars`: 长度为 `num_history_frames` 的列表（当前未使用）

#### 3.1 `AgentInput/EgoStatus/Cameras/Lidar` 的组装（与代码一一对应）

**代码位置**: `navsim/common/dataclasses.py` → `AgentInput.from_scene_dict_list()`

```193:226:navsim/common/dataclasses.py
        ego_statuses: List[EgoStatus] = []
        cameras: List[EgoStatus] = []
        lidars: List[Lidar] = []

        for frame_idx in range(num_history_frames):

            ego_dynamic_state = scene_dict_list[frame_idx]["ego_dynamic_state"]
            ego_status = EgoStatus(
                ego_pose=np.array(local_ego_poses[frame_idx], dtype=np.float32),
                ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
                ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
                driving_command=scene_dict_list[frame_idx]["driving_command"],
            )
            ego_statuses.append(ego_status)

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            cameras.append(
                Cameras.from_camera_dict(
                    sensor_blobs_path=sensor_blobs_path,
                    camera_dict=scene_dict_list[frame_idx]["cams"],
                    sensor_names=sensor_names,
                    load_image_path=load_image_path,
                )
            )

            lidars.append(
                Lidar.from_paths(
                    sensor_blobs_path=sensor_blobs_path,
                    lidar_path=Path(scene_dict_list[frame_idx]["lidar_path"]),
                    sensor_names=sensor_names,
                )
            )
```

- **`ego_pose`**: `local_ego_poses[frame_idx]`，并 cast 为 `float32`
- **`ego_velocity/ego_acceleration`**: `ego_dynamic_state[:2]` / `ego_dynamic_state[2:]`，cast 为 `float32`
- **`cameras`**: 由 `sensor_config.get_sensors_at_iteration(frame_idx)` 决定本帧加载哪些相机；`Cameras.from_camera_dict()` 内部会从 `sensor_blobs_path / data_path` 读图（`load_image_path=False` 时返回 `np.array(Image.open(...))`）
- **`lidars`**: 只有当 `sensor_names` 里包含 `lidar_pc` 才会真的读取点云，否则返回空 `Lidar()`

## 4. EponaAgent 数据处理

**代码位置**: `navsim/agents/epona/epona_agent.py`

### 4.1 图像提取和预处理

在 `compute_trajectory()` 方法中，首先提取图像：

```158:194:navsim/agents/epona/epona_agent.py
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        if not self._initialized:
            self.initialize()
        
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._condition_frames is not None
        assert self._traj_len is not None
        assert self._image_size is not None
        assert self._device is not None
        
        history_len = len(agent_input.ego_statuses)
        if history_len == 0:
            logger.warning("Empty agent input. Returning zero trajectory.")
            return self._build_fallback_trajectory()
        
        start_idx = max(0, history_len - self._condition_frames)
        cameras = agent_input.cameras[start_idx:]
        ego_statuses = agent_input.ego_statuses[start_idx:]
        
        if len(cameras) < self._condition_frames:
            pad_count = self._condition_frames - len(cameras)
            cameras = [cameras[0]] * pad_count + cameras
            ego_statuses = [ego_statuses[0]] * pad_count + ego_statuses
        
        images = []
        for camera in cameras:
            image = camera.cam_f0.image
            if image is None:
                # Fall back to a black frame to keep inference running.
                image = np.zeros((self._image_size[0], self._image_size[1], 3), dtype=np.uint8)
            pil = Image.fromarray(image)
            pil = pil.resize((self._image_size[1], self._image_size[0]), Image.BICUBIC)
            images.append(np.array(pil, dtype=np.uint8))
        
        images = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).float()
        images = _normalize_images(images).unsqueeze(0).to(self._device)
```

**数据形状变化**:

1. **输入**: `agent_input.cameras` - 长度为 `num_history_frames` 的列表
   - 每个 `camera.cam_f0.image`: `(H_orig, W_orig, 3)` uint8

2. **裁剪和填充**:
   - 如果 `history_len > condition_frames`，只取最后 `condition_frames` 帧
   - 如果 `history_len < condition_frames`，用第一帧填充到 `condition_frames` 帧
   - 输出: `cameras` 列表，长度为 `condition_frames`（默认4）

3. **图像resize**:
   - 输入: `(H_orig, W_orig, 3)` uint8
   - 输出: `(image_size[0], image_size[1], 3)` uint8
   - 默认 `image_size=(512, 1024)`，所以输出为 `(512, 1024, 3)`

4. **转换为tensor并归一化**:
   - `np.stack(images, axis=0)`: `(condition_frames, 512, 1024, 3)` uint8
   - `.permute(0, 3, 1, 2)`: `(condition_frames, 3, 512, 1024)` float
   - `_normalize_images()`: 归一化到 `[-1, 1]` 范围
   - `.unsqueeze(0)`: `(1, condition_frames, 3, 512, 1024)` float
   - `.to(device)`: 移动到GPU

**最终图像tensor形状**: `(1, condition_frames, 3, 512, 1024)` float32，值域 `[-1, 1]`

### 4.2 位姿处理

```196:203:navsim/agents/epona/epona_agent.py
        pose_mats = []
        for status in ego_statuses:
            x, y, yaw = status.ego_pose
            pose_mats.append(_build_se2_matrix(x, y, yaw))
        
        # Duplicate the last pose to avoid leaking future information.
        pose_mats.append(pose_mats[-1].copy())
        pose_tensor = torch.from_numpy(np.stack(pose_mats, axis=0)).unsqueeze(0).to(self._device)
```

**数据形状变化**:

1. **输入**: `ego_statuses` - 长度为 `condition_frames` 的列表
   - 每个 `ego_status.ego_pose`: `(3,)` - `[x, y, yaw]` (float32)

2. **构建SE2矩阵**:
   - `_build_se2_matrix(x, y, yaw)`: 将 `(x, y, yaw)` 转换为 `(4, 4)` SE2变换矩阵
   - `pose_mats`: 长度为 `condition_frames + 1` 的列表（最后一位重复）
   - 每个元素: `(4, 4)` float32

3. **转换为tensor**:
   - `np.stack(pose_mats, axis=0)`: `(condition_frames + 1, 4, 4)` float32
   - `.unsqueeze(0)`: `(1, condition_frames + 1, 4, 4)` float32
   - `.to(device)`: 移动到GPU

**最终位姿tensor形状**: `(1, condition_frames + 1, 4, 4)` float32

### 4.3 相对位姿计算

```205:225:navsim/agents/epona/epona_agent.py
        with torch.no_grad():
            rel_pose, rel_yaw = get_rel_pose(pose_tensor)
            latents = self._tokenizer.encode_to_z(images)
            use_amp = self.use_amp and self._device.type == "cuda"
            if use_amp:
                with torch.autocast(device_type=self._device.type, dtype=torch.bfloat16):
                    predict_traj, _ = self._model.step_eval(
                        latents,
                        rel_pose[:, : self._condition_frames + 1],
                        rel_yaw[:, : self._condition_frames + 1],
                        self_pred_traj=False,
                        traj_only=True,
                    )
            else:
                predict_traj, _ = self._model.step_eval(
                    latents,
                    rel_pose[:, : self._condition_frames + 1],
                    rel_yaw[:, : self._condition_frames + 1],
                    self_pred_traj=False,
                    traj_only=True,
                )
```

**数据形状变化**:

1. **相对位姿计算**:
   - 输入: `pose_tensor`: `(1, condition_frames + 1, 4, 4)`
   - `get_rel_pose()` 计算相对位姿
   - 输出: 
     - `rel_pose`: `(1, condition_frames + 1, ...)` - 相对位置
     - `rel_yaw`: `(1, condition_frames + 1, ...)` - 相对角度

2. **图像编码**:
   - 输入: `images`: `(1, condition_frames, 3, 512, 1024)`
   - `self._tokenizer.encode_to_z()`: 通过VAE编码器
   - 输出: `latents`: `(1, condition_frames, latent_channels, latent_h, latent_w)`
     - 根据VAE配置，`latent_channels=32`, `latent_h=16`, `latent_w=32`（下采样32倍）
     - 所以 `latents`: `(1, condition_frames, 32, 16, 32)`

3. **模型推理**:
   - 输入:
     - `latents`: `(1, condition_frames, 32, 16, 32)`
     - `rel_pose`: `(1, condition_frames + 1, ...)`
     - `rel_yaw`: `(1, condition_frames + 1, ...)`
   - 输出: `predict_traj`: `(1, traj_len, 3)` - `[x, y, yaw]`（yaw为度数）

### 4.4 轨迹后处理

```227:239:navsim/agents/epona/epona_agent.py
        traj = predict_traj[0].detach().cpu().numpy()
        if traj.shape[-1] > 3:
            # Keep x/y/yaw only if extra channels exist.
            traj = traj[..., :3]
        # Epona outputs yaw in degrees; convert to radians for NAVSIM.
        traj[:, 2] = np.deg2rad(traj[:, 2])
        output_num_poses = int(round(self.output_time_horizon / self.output_interval))
        traj = _resample_trajectory(traj, output_num_poses)
        trajectory_sampling = TrajectorySampling(
            time_horizon=self.output_time_horizon,
            interval_length=self.output_interval,
        )
        return Trajectory(traj.astype(np.float32), trajectory_sampling)
```

**数据形状变化**:

1. **提取轨迹**:
   - 输入: `predict_traj`: `(1, traj_len, 3)` - `[x, y, yaw_degrees]`
   - `predict_traj[0]`: `(traj_len, 3)` - 移除batch维度
   - 默认 `traj_len=8`，所以形状为 `(8, 3)`

2. **角度转换**:
   - `traj[:, 2] = np.deg2rad(traj[:, 2])`: 将yaw从度数转换为弧度
   - 形状保持: `(8, 3)`

3. **重采样**:
   - `output_num_poses = int(round(4.0 / 0.5)) = 8`
   - `_resample_trajectory(traj, 8)`: 如果 `traj_len=8`，则不需要重采样
   - 如果 `traj_len != 8`，则通过线性插值重采样到8个点
   - 输出: `(8, 3)` - `[x, y, yaw_radians]` (float32)

4. **最终输出**:
   - `Trajectory` 对象，包含:
     - `poses`: `(8, 3)` float32 - `[x, y, yaw]`
     - `sampling`: `TrajectorySampling` 对象（`time_horizon=4.0`, `interval_length=0.5`）

## 5. 数据形状变化总结

### 5.1 完整数据流

```
原始数据 (pkl文件)
  └─> scene_dict_list: List[Dict] (长度=num_history_frames+num_future_frames)
      └─> AgentInput
          ├─> ego_statuses: List[EgoStatus] (长度=num_history_frames)
          │   └─> ego_pose: (3,) [x, y, yaw] float32
          ├─> cameras: List[Cameras] (长度=num_history_frames)
          │   └─> cam_f0.image: (H_orig, W_orig, 3) uint8
          └─> lidars: List[Lidar] (长度=num_history_frames)
      
      └─> EponaAgent.compute_trajectory()
          ├─> images: (1, condition_frames, 3, 512, 1024) float32 [-1, 1]
          ├─> pose_tensor: (1, condition_frames+1, 4, 4) float32
          ├─> latents: (1, condition_frames, 32, 16, 32) float32
          ├─> rel_pose, rel_yaw: (1, condition_frames+1, ...) float32
          └─> predict_traj: (1, traj_len, 3) float32
              └─> traj: (8, 3) [x, y, yaw_radians] float32
                  └─> Trajectory.poses: (8, 3) float32
```

### 5.2 关键参数

根据 `test_navsim.sh` 和配置文件 `dit_config_dcae_nuplan.py`:

- `condition_frames = 4`: 历史帧数
- `traj_len = 8`: 预测轨迹长度
- `image_size = (512, 1024)`: 图像尺寸
- `output_time_horizon = 4.0`: 输出时间范围（秒）
- `output_interval = 0.5`: 输出时间间隔（秒）
- `output_num_poses = 8`: 输出轨迹点数（4.0 / 0.5 = 8）

### 5.3 数据形状对照表

| 阶段 | 数据 | 形状 | 数据类型 | 说明 |
|------|------|------|----------|------|
| 原始数据 | pkl文件中的帧 | Dict | - | 包含ego状态、相机、点云路径等 |
| SceneLoader | scene_dict_list | List[Dict] | - | 长度为 num_history_frames+num_future_frames |
| AgentInput | ego_statuses | List[EgoStatus] | - | 长度为 num_history_frames (4) |
| | ego_pose | (3,) | float32 | [x, y, yaw] |
| | cam_f0.image | (H_orig, W_orig, 3) | uint8 | 原始图像 |
| EponaAgent | images (resize后) | (1, 4, 3, 512, 1024) | float32 | 归一化到[-1, 1] |
| | pose_tensor | (1, 5, 4, 4) | float32 | SE2变换矩阵 |
| | latents | (1, 4, 32, 16, 32) | float32 | VAE编码后的特征 |
| | predict_traj | (1, 8, 3) | float32 | 模型预测轨迹 |
| 最终输出 | Trajectory.poses | (8, 3) | float32 | [x, y, yaw_radians] |

## 6. 关键代码文件索引

- **脚本入口**: `run/test_navsim.sh`
- **主执行脚本**: `scripts/test/run_navsim_eval_pipeline1.py`
- **推理脚本**: `scripts/test/test_navsim_traj.py`
- **数据加载器**: `navsim/common/dataloader.py`
- **数据类定义**: `navsim/common/dataclasses.py`
- **Agent实现**: `navsim/agents/epona/epona_agent.py`
- **配置文件**: `configs/dit_config_dcae_nuplan.py`

## 7. 注意事项

1. **坐标系统**: 
   - 原始数据使用全局坐标（ego2global）
   - AgentInput 中转换为局部坐标（相对于最后一帧）
   - 模型输出也是局部坐标

2. **角度单位**:
   - 模型输出yaw为度数
   - 最终输出转换为弧度（NAVSIM标准）

3. **图像处理**:
   - 图像从 `sensor_blobs_path` 动态加载
   - 如果图像加载失败，使用黑色图像作为fallback
   - 图像归一化到 `[-1, 1]` 范围

4. **帧数处理**:
   - 如果历史帧数少于 `condition_frames`，用第一帧填充
   - 如果历史帧数多于 `condition_frames`，只取最后 `condition_frames` 帧

5. **轨迹重采样**:
   - 如果模型输出的轨迹点数与目标点数不同，使用线性插值重采样
   - 重采样保持轨迹的连续性
