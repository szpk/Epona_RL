# Random seed | 随机种子
seed=1234

#! Dataset paths | 数据集路径
datasets_paths=dict(
    nuscense_root='',
    nuscense_train_json_path='',
    nuscense_val_json_path='',
    
    nuplan_root= '/inspire/hdd/project/roboticsystem2/public/nuplan/dataset',
    nuplan_json_root= '/inspire/hdd/project/roboticsystem2/public/epona',
    openscene_root= '/inspire/hdd/project/roboticsystem2/public/OpenScene/openscene-v1.1',  # NavSim dataset path (required for GRPO with metric cache) | NavSim数据集路径（GRPO训练需要metric cache）
)
train_data_list=['navsim']  # Use navsim dataset (OpenScene data) | 使用navsim数据集（OpenScene数据）
val_data_list=['navsim']

downsample_fps=2  # video clip is downsampled to * fps. | 视频片段降采样到*帧率
mask_data=0 #1 means all masked, 0 means all gt | 1表示全部遮蔽，0表示全部使用真值
image_size=(512, 1024)
pkeep=0.7 #Percentage for how much latent codes to keep. | 保留潜在编码的百分比
reverse_seq=False
paug=0

# VAE configs | VAE配置
vae_embed_dim=32
downsample_size=32
patch_size=1
vae='DCAE_f32c32'
vae_ckpt='/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/Epona/checkpoints/Epona/dcae_td_20000.pkl' #! VAE checkpoint path | VAE检查点路径
add_encoder_temporal=False
add_decoder_temporal=True
temporal_patch_size=6

# World Model configs | 世界模型配置
condition_frames=4
n_layer=[12, 6, 6]

n_head=16
n_embd=2048
gpt_type='diffgpt_mar'
pose_x_vocab_size=128
pose_y_vocab_size=128
yaw_vocab_size=512

# Logs | 日志配置
outdir="exp/ckpt"
logdir="exp/job_log"
tdir="exp/job_tboard"
validation_dir="exp/validation"

diffusion_model_type="flow"
num_sampling_steps=10  # 增加采样步数以提高SDE轨迹质量 | Increased for better SDE trajectory quality
lambda_yaw_pose=1.0

diff_only=True
forward_iter=3
multifw_perstep=10
block_size=1

n_embd_dit=2048
n_head_dit=16
axes_dim_dit=[16, 56, 56]
return_predict=True

traj_len=8
n_layer_traj=[1, 1]
n_embd_dit_traj=1024
n_head_dit_traj=8
axes_dim_dit_traj=[16, 56, 56]
return_predict_traj=True

# Freeze STT and DiT, only train TrajDiT for GRPO | 冻结STT和DiT，GRPO训练时只训练TrajDiT
fix_stt=True
fix_dit=True
test_video_frames=50
drop_feature=0
no_pose=False
sample_prob=[1.0]
pose_x_bound=50
pose_y_bound=10
yaw_bound=12

# ========== GRPO Configuration | GRPO配置 ==========
# Enable GRPO training | 启用GRPO训练
grpo=True

# GRPO training hyperparameters | GRPO训练超参数
grpo_sample_time=8  # 增加采样数量以获得更好的advantage估计 | Increased for better advantage estimation
grpo_gamma_denoising=0.6  # Discount factor for denoising steps | 去噪步骤的折扣因子
grpo_noise_level=0.3  # 降低SDE噪声水平以提高轨迹质量 | Reduced noise for better trajectory quality (was 0.7 default)
grpo_bc_coeff=0.01  # 降低BC loss权重，让policy loss有更大影响 | Reduced to let policy loss dominate
grpo_use_bc_loss=True  # Whether to use BC loss | 是否使用BC损失

# Advantage clipping | 优势值裁剪
grpo_clip_advantage_lower_quantile=0.0  # Lower quantile for advantage clipping | 优势值裁剪的下分位数
grpo_clip_advantage_upper_quantile=1.0  # Upper quantile for advantage clipping | 优势值裁剪的上分位数

# Log probability computation | 对数概率计算
grpo_min_logprob_denoising_std=0.1  # Minimum standard deviation for logprob computation | 对数概率计算的最小标准差

# Reference policy checkpoint (for BC loss) | 参考策略检查点（用于BC损失）
# Set to None to use the initial model weights as reference | 设为None则使用初始模型权重作为参考
grpo_reference_policy_checkpoint=None  # e.g., '/path/to/reference_policy.ckpt' | 例如：'/path/to/reference_policy.ckpt'

# Metric cache path (for PDM scorer) | Metric缓存路径（用于PDM评分器）
# This should point to the metric cache generated from navsim training data | 应指向从navsim训练数据生成的metric缓存
# 使用Epona完整的metric cache，而不是recogdrive的不完整版本
grpo_metric_cache_path='/inspire/hdd/project/roboticsystem2/jingzhanghui-253108140204/world_model/RL/recogdrive/exp/metric_cache_train'

# PDM Scorer configuration (optional, uses defaults if None) | PDM评分器配置（可选，为None时使用默认值）
grpo_scorer_config=None  # Can be PDMScorerConfig object with custom weights | 可以是带有自定义权重的PDMScorerConfig对象
