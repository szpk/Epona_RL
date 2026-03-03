# epona_agent.yaml参数对test_navsim.sh的影响说明

## 问题

如果在`epona_agent.yaml`配置文件中添加`output_time_horizon: 4.0`和`output_interval: 0.5`，会不会影响`test_navsim.sh`脚本的运行？原本这个脚本运行时这两个参数的值是多少？

---

## 核心答案

### 1. 当前test_navsim.sh使用的参数值

**`test_navsim.sh`当前运行时这两个参数的值**：
- `output_time_horizon = 4.0`（默认值）
- `output_interval = 0.5`（默认值）

**来源**：`run_navsim_eval_pipeline1.py`中的默认值定义（第137-138行）：
```python
parser.add_argument("--output_time_horizon", type=float, default=4.0)
parser.add_argument("--output_interval", type=float, default=0.5)
```

### 2. 在epona_agent.yaml中添加这些参数的影响

**答案：不会影响`test_navsim.sh`的运行**

**原因**：
- `test_navsim.sh`使用的是**Submission模式**（`run_navsim_eval_pipeline1.py`）
- Submission模式**不使用agent配置文件**，而是直接使用命令行参数
- 所以`epona_agent.yaml`中的参数不会影响`test_navsim.sh`

---

## 详细说明

### 当前test_navsim.sh的工作流程

```
test_navsim.sh
  ↓
run_navsim_eval_pipeline1.py (Submission模式)
  ↓
  ├─ test_navsim_traj.py (推理，使用命令行参数)
  ├─ 打包成submission.pkl
  └─ run_pdm_score_from_submission.py (评分)
```

**关键点**：
- `run_navsim_eval_pipeline1.py`使用`argparse`解析命令行参数
- 参数来源：命令行参数 → 配置文件（`Config.fromfile`） → 默认值
- **不使用agent配置文件**

### 参数传递路径

在`run_navsim_eval_pipeline1.py`中：

1. **定义默认值**（第137-138行）：
```python
parser.add_argument("--output_time_horizon", type=float, default=4.0)
parser.add_argument("--output_interval", type=float, default=0.5)
```

2. **传递给test_navsim_traj.py**（第216-219行）：
```python
inference_cmd = [
    sys.executable,
    "scripts/test/test_navsim_traj.py",
    # ...
    "--output_time_horizon",
    str(cfg.output_time_horizon),  # 使用cfg中的值
    "--output_interval",
    str(cfg.output_interval),      # 使用cfg中的值
    # ...
]
```

3. **cfg的值来源**（第159-166行）：
```python
args = parser.parse_args()  # 解析命令行参数
cfg = Config.fromfile(args.config)  # 加载配置文件
overrides = {}
for key, value in vars(args).items():
    if value is None and hasattr(cfg, key):
        continue
    overrides[key] = value
cfg.merge_from_dict(overrides)  # 命令行参数覆盖配置文件
```

**优先级**：
1. 命令行参数（最高优先级）
2. 配置文件（`configs/dit_config_dcae_nuplan.py`）
3. argparse默认值（`default=4.0`和`default=0.5`）

---

## 如果改用Agent接口模式

### 情况1：使用epona_agent_jch.py（当前推荐版本）

**`epona_agent_jch.py`的`__init__`方法**：
```python
def __init__(
    self,
    config_path: str,
    checkpoint_path: Optional[str] = None,
    condition_frames: int = 3,
    use_sde: bool = False,
    # 注意：不接受output_time_horizon和output_interval参数
):
```

**结论**：如果在`epona_agent.yaml`中添加这两个参数，**不会被使用**，因为`epona_agent_jch.py`不接受这些参数。

### 情况2：使用epona_agent.py（旧版本）

**`epona_agent.py`的`__init__`方法**：
```python
def __init__(
    self,
    config_path: str,
    checkpoint_path: str,
    # ...
    output_time_horizon: float = 4.0,  # 接受这个参数
    output_interval: float = 0.5,      # 接受这个参数
    # ...
):
```

**结论**：如果在`epona_agent.yaml`中添加这两个参数，**会被使用**，但需要确保：
1. 使用`epona_agent.py`而不是`epona_agent_jch.py`
2. 使用Agent接口模式（`run_navsim_eval_pipeline.py`）

---

## 总结

### 对test_navsim.sh的影响

| 场景 | 是否影响test_navsim.sh | 说明 |
|------|----------------------|------|
| **当前情况**（Submission模式） | ❌ **不影响** | `test_navsim.sh`不使用agent配置文件 |
| **改用Agent接口模式** + `epona_agent_jch.py` | ❌ **不影响** | `epona_agent_jch.py`不接受这些参数 |
| **改用Agent接口模式** + `epona_agent.py` | ✅ **会影响** | 会使用`epona_agent.yaml`中的参数值 |

### 当前test_navsim.sh的参数值

- `output_time_horizon = 4.0`（来自`run_navsim_eval_pipeline1.py`的默认值）
- `output_interval = 0.5`（来自`run_navsim_eval_pipeline1.py`的默认值）

### 建议

1. **如果使用Submission模式**（当前`test_navsim.sh`）：
   - 在`epona_agent.yaml`中添加这些参数**不会影响**运行
   - 参数值由`run_navsim_eval_pipeline1.py`的默认值或命令行参数决定

2. **如果改用Agent接口模式**：
   - 如果使用`epona_agent_jch.py`：这些参数不会被使用
   - 如果使用`epona_agent.py`：这些参数会被使用，但需要确保`epona_agent.yaml`中的值正确

3. **推荐做法**：
   - 保持`test_navsim.sh`使用Submission模式（当前方式）
   - 如果需要修改参数，在命令行或配置文件中修改，而不是在`epona_agent.yaml`中

---

## 如何修改test_navsim.sh的参数

### 方式1：在命令行添加参数（推荐）

```bash
python scripts/test/run_navsim_eval_pipeline1.py \
  --exp_name "navsim-traj" \
  --output_time_horizon 6.0 \  # 修改为6.0秒
  --output_interval 0.3 \      # 修改为0.3秒
  # ... 其他参数
```

### 方式2：在配置文件中修改

修改`configs/dit_config_dcae_nuplan.py`：
```python
output_time_horizon = 6.0
output_interval = 0.3
```

### 方式3：修改run_navsim_eval_pipeline1.py的默认值

修改`scripts/test/run_navsim_eval_pipeline1.py`第137-138行：
```python
parser.add_argument("--output_time_horizon", type=float, default=6.0)  # 改为6.0
parser.add_argument("--output_interval", type=float, default=0.3)      # 改为0.3
```

---

## 验证方法

### 检查当前使用的参数值

在`test_navsim_traj.py`中添加打印语句：
```python
print(f"output_time_horizon: {args.output_time_horizon}")
print(f"output_interval: {args.output_interval}")
```

或者查看生成的轨迹文件，检查轨迹点数：
- `output_time_horizon = 4.0`, `output_interval = 0.5` → 8个轨迹点
- `output_time_horizon = 6.0`, `output_interval = 0.3` → 20个轨迹点

---

## 参考代码位置

- **默认值定义**：`scripts/test/run_navsim_eval_pipeline1.py` 第137-138行
- **参数传递**：`scripts/test/run_navsim_eval_pipeline1.py` 第216-219行
- **EponaAgent（旧版本）**：`navsim/agents/epona/epona_agent.py` 第75-76行
- **EponaAgent（新版本）**：`navsim/agents/epona/epona_agent_jch.py` 第41-46行
