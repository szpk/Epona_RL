import argparse
import os
import random
import sys
import yaml
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

root_path = Path(__file__).resolve().parents[2]
print(root_path)
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from utils.config_utils import Config
from utils.testing_utils import plot_trajectory

from navsim.agents.epona.epona_agent import EponaAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader

def _resample_trajectory(poses: np.ndarray, target_len: int) -> np.ndarray:
    if poses.shape[0] == target_len:
        return poses
    if poses.shape[0] < 2:
        return np.repeat(poses[:1], target_len, axis=0)
    src = np.linspace(0.0, 1.0, poses.shape[0])
    dst = np.linspace(0.0, 1.0, target_len)
    x_interp = np.interp(dst, src, poses[:, 0])
    y_interp = np.interp(dst, src, poses[:, 1])
    yaw_unwrapped = np.unwrap(poses[:, 2])
    yaw_interp = np.interp(dst, src, yaw_unwrapped)
    yaw_interp = (yaw_interp + np.pi) % (2.0 * np.pi) - np.pi
    return np.stack([x_interp, y_interp, yaw_interp], axis=-1)


def add_arguments():
    openscene_root = os.environ.get("OPENSCENE_DATA_ROOT", "")
    default_navsim_log_path = (
        os.path.join(openscene_root, "navsim_logs/navtest") if openscene_root else ""
    )
    default_sensor_blobs_path = (
        os.path.join(openscene_root, "sensor_blobs") if openscene_root else ""
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_video_path", type=str, default="test_videos")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--config", default="configs/dit_config_dcae_nuplan.py", type=str)
    parser.add_argument("--resume_path", default=None, type=str, help="pretrained path")
    parser.add_argument("--vae_ckpt", default=None, type=str, help="optional VAE checkpoint override")
    parser.add_argument("--condition_frames", type=int, default=4)
    parser.add_argument("--traj_len", type=int, default=8)

    parser.add_argument("--navsim_log_path", type=str, default=default_navsim_log_path)
    parser.add_argument("--sensor_blobs_path", type=str, default=default_sensor_blobs_path)
    parser.add_argument("--navsim_history_frames", type=int, default=None)
    parser.add_argument("--navsim_future_frames", type=int, default=None)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=100)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--output_time_horizon", type=float, default=4.0)
    parser.add_argument("--output_interval", type=float, default=0.5)
    parser.add_argument("--no_plot", action="store_true")

    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    overrides = {}
    for key, value in vars(args).items():
        if value is None and hasattr(cfg, key):
            continue
        overrides[key] = value
    cfg.merge_from_dict(overrides)
    return cfg


args = add_arguments()
print(args)

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True


def main(cfg):
    if not cfg.navsim_log_path or not cfg.sensor_blobs_path:
        raise ValueError("navsim_log_path and sensor_blobs_path are required.")
    if not cfg.resume_path:
        raise ValueError("resume_path is required for inference.")
    save_path = os.path.join(cfg.save_video_path, cfg.exp_name)
    os.makedirs(save_path, exist_ok=True)

    agent = EponaAgent(
        config_path=cfg.config,
        checkpoint_path=cfg.resume_path,
        vae_checkpoint_path=getattr(cfg, "vae_ckpt", None),
        device=cfg.device,
        use_amp=not cfg.no_amp,
        output_time_horizon=cfg.output_time_horizon,
        output_interval=cfg.output_interval,
        condition_frames=cfg.condition_frames,
        traj_len_override=cfg.traj_len,
        image_size=cfg.image_size,
    )
    agent.initialize()
    num_history_frames = cfg.navsim_history_frames or cfg.condition_frames
    required_future = int(round(cfg.output_time_horizon / cfg.output_interval))
    num_future_frames = max(cfg.navsim_future_frames or cfg.traj_len, required_future)
    navtest_config_path = root_path / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    log_names, tokens = None, None
    if navtest_config_path.exists():
        with open(navtest_config_path, 'r') as f:
            navtest_cfg = yaml.safe_load(f)
        log_names = navtest_cfg.get('log_names')
        tokens = navtest_cfg.get('tokens')
    scene_filter = SceneFilter(
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
        frame_interval=cfg.frame_interval,
        max_scenes=cfg.max_scenes,
        log_names=log_names,
        tokens=tokens,
    )
    scene_loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    tokens = scene_loader.tokens
    start_id = cfg.start_id
    end_id = min(cfg.end_id, len(tokens))
    tokens = tokens[start_id:end_id]
    print(f"Tokens: {len(tokens)} ({start_id}-{end_id})")

    for idx, token in tqdm(enumerate(tokens), total=len(tokens), desc="NAVSIM inference"):
        agent_input = scene_loader.get_agent_input_from_token(token)
        pred_traj = agent.compute_trajectory(agent_input)
        np.save(os.path.join(save_path, f"{token}.npy"), pred_traj.poses)

        if cfg.no_plot:
            continue

        try:
            scene = scene_loader.get_scene_from_token(token)
            gt_traj = scene.get_future_trajectory(num_trajectory_frames=pred_traj.poses.shape[0]).poses
        except Exception:
            continue

        if gt_traj.shape[0] != pred_traj.poses.shape[0]:
            gt_traj = _resample_trajectory(gt_traj, pred_traj.poses.shape[0])

        plot_trajectory(pred_traj.poses, gt_traj, save_path, idx)

if __name__ == "__main__":
    main(args)