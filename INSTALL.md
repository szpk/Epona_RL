# Epona_RL 安装指南

## 环境要求

- Python 3.8+
- CUDA 12.8+ (支持 NVIDIA GPU)
- Conda 或 Miniconda

## 快速安装

### 1. 创建 Conda 环境

```bash
conda create -n epona python=3.10 -y
conda activate epona
```

### 2. 安装 PyTorch (CUDA 版本)

```bash
# PyTorch 2.10.0 with CUDA 12.8
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### 3. 安装项目依赖

```bash
cd Epona_RL
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
```

## 数据集准备 (可选)

```bash
# NavSim 数据集
pip install navsim==1.1.0

# NuPlan 数据集
pip install nuplan-devkit==1.2.0
```

## 常见问题

**Q: CUDA 版本不匹配？**
```bash
nvidia-smi  # 检查驱动是否支持 CUDA 12.8+
```

**Q: 依赖冲突？**
```bash
# 使用全新环境
conda remove -n epona --all -y
# 重新执行安装步骤
```

**Q: GPU 内存不足？**
- 减小 batch size (配置文件中修改)
- 使用 DeepSpeed ZeRO-3
- 启用梯度检查点

## 快速开始

```bash
# GRPO 训练
bash run/train_deepspeed_grpo.sh

```
