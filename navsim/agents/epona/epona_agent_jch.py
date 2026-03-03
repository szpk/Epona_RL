from typing import Optional, List, Dict
import os
import sys
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Add Epona root to path
root_path = Path(__file__).parent.parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig, NAVSIM_INTERVAL_LENGTH
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from utils.config_utils import Config
from models.model import TrainTransformersDiT
from models.modules.tokenizer import VAETokenizer
from utils.preprocess import get_rel_pose


def _build_se2_matrix(x: float, y: float, yaw: float) -> np.ndarray:
    """
    Build a 4x4 SE2 matrix from x, y, yaw.
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.array([[cos_yaw, -sin_yaw, 0.0, x],
                     [sin_yaw, cos_yaw, 0.0, y],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)


class EponaAgent(AbstractAgent):
    """
    Epona agent for navsim evaluation.
    Follows the official test_traj.py implementation.
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str] = None,
        condition_frames: int = 3,
        use_sde: bool = False,
    ):
        """
        Initialize EponaAgent.
        
        :param config_path: Path to model config file
        :param checkpoint_path: Path to model checkpoint
        :param condition_frames: Number of condition frames
        :param use_sde: Whether to use SDE sampling (8 trajectories) or ODE (1 trajectory)
        """
        super().__init__(requires_scene=False)
        
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._condition_frames = condition_frames
        self._use_sde = use_sde

        # Load config
        self._args = Config.fromfile(config_path)
        self._args.condition_frames = condition_frames
        
        # Set default batch_size for evaluation
        if not hasattr(self._args, 'batch_size') or self._args.batch_size is None:
            self._args.batch_size = 1
        
        # Initialize model and tokenizer (same as test_traj.py)
        local_rank = 0
        self._model = TrainTransformersDiT(
            self._args, 
            load_path=checkpoint_path, 
            local_rank=local_rank, 
            condition_frames=condition_frames
        )
        self._tokenizer = VAETokenizer(self._args, local_rank)
        
        # Move to device
        self._model = self._model.cuda()
        self._tokenizer = self._tokenizer.cuda()
        self._model.eval()
        self._tokenizer.eval()

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=True)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Compute trajectory following official test_traj.py implementation.
        
        :param agent_input: Agent input containing cameras and ego statuses
        :return: Predicted trajectory
        """
        self._model.eval()
        
        # 1. Extract and process images (same as test_traj.py)
        # Convert from navsim format to official format: (T, C, H, W)
        images = []
        for i in range(min(self._condition_frames, len(agent_input.cameras))):
            cameras = agent_input.cameras[i]
            if hasattr(cameras, 'cam_f0') and cameras.cam_f0 is not None:
                img = cameras.cam_f0.image  # numpy array (H, W, 3)
                # Normalize and convert to tensor (same as dataset_nuplan.py)
                img_tensor = torch.from_numpy(img.copy()).float() / 255.0
                img_tensor = (img_tensor - 0.5) * 2.0  # Normalize to [-1, 1]
                img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
                # Resize if needed
                if img_tensor.shape[1:] != tuple(self._args.image_size):
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize(self._args.image_size),
                    ])
                    img_tensor = transform(img_tensor)
                images.append(img_tensor)
        
        # Pad if needed
        while len(images) < self._condition_frames:
            if images:
                images.insert(0, images[0])
            else:
                dummy_img = torch.full((3, *self._args.image_size), -1.0, dtype=torch.float32)
                images.append(dummy_img)
        
        # Stack images: (condition_frames, C, H, W)
        # Images are already normalized to [-1, 1] in the loop above
        imgs = torch.stack(images[:self._condition_frames])
        imgs = imgs.cuda()
        
        # 2. Extract poses from agent_input.ego_statuses and build SE2 matrices
        ego_statuses = agent_input.ego_statuses
        pose_mats = []
        for i in range(self._condition_frames + 1):
            if i < len(ego_statuses):
                status = ego_statuses[i]
                x, y, yaw = status.ego_pose
                pose_mats.append(_build_se2_matrix(x, y, yaw))
            else:
                # Use last available pose
                if pose_mats:
                    pose_mats.append(pose_mats[-1].copy())
                else:
                    pose_mats.append(np.eye(4, dtype=np.float32))
        
        # Stack to tensor: (T, 4, 4)
        rot_matrix = torch.from_numpy(np.stack(pose_mats)).float().cuda()
        
        # 3. Encode images to latents (same as test_traj.py)
        start_latents = self._tokenizer.encode_to_z(imgs[:self._condition_frames].unsqueeze(0))
        
        # 4. Extract relative pose and yaw (same as test_traj.py)
        pose, yaw = get_rel_pose(rot_matrix.unsqueeze(0))
        
        # 5. Generate trajectory (same as test_traj.py)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predict_traj, _ = self._model.step_eval(
                    start_latents,
                    pose[:, :self._condition_frames+1, ...], 
                    yaw[:, :self._condition_frames+1, ...], 
                    self_pred_traj=False,
                    traj_only=True,
                    use_sde=self._use_sde
                )
        
        # 6. Handle output (same as test_traj.py)
        if predict_traj.ndim == 4:  # (num_samples, B, H, D)
            predict_traj = predict_traj[0]  # Use first trajectory: (B, H, D)
        
        # Convert to numpy: (H, D)
        predict_traj_np = predict_traj[0].cpu().numpy()
        
        # Extract trajectory poses: (horizon, 3) for x, y, heading
        if predict_traj_np.shape[1] >= 3:
            trajectory_poses = predict_traj_np[:, :3]
        else:
            trajectory_poses = np.zeros((predict_traj_np.shape[0], 3))
            trajectory_poses[:, :predict_traj_np.shape[1]] = predict_traj_np
        
        # Epona outputs yaw in degrees; convert to radians for NAVSIM.
        trajectory_poses[:, 2] = np.deg2rad(trajectory_poses[:, 2])
        
        # Create TrajectorySampling
        num_poses = trajectory_poses.shape[0]
        trajectory_sampling = TrajectorySampling(
            num_poses=num_poses,
            interval_length=NAVSIM_INTERVAL_LENGTH  # 0.5s per pose
        )
        target_num_poses = 8  # NavSim标准：8个点
        if trajectory_poses.shape[0] != target_num_poses:
            print(f"Trajectory shape: {trajectory_poses.shape[0]}, target num poses: {target_num_poses}")
            print(f"Trajectory shape: {trajectory_poses.shape[0]}, target num poses: {target_num_poses}")
            print(f"Trajectory shape: {trajectory_poses.shape[0]}, target num poses: {target_num_poses}")
        print(f"Trajectory shape: {trajectory_poses.shape[0]}, target num poses: {target_num_poses}")
        print(f"Trajectory shape: {trajectory_poses.shape}, target num poses: {target_num_poses}")
        print(f"Trajectory shape: {trajectory_poses.shape}, target num poses: {target_num_poses}")
        print(f"Trajectory shape: {trajectory_poses.shape}, target num poses: {target_num_poses}")
        print(f"Trajectory shape: {trajectory_poses.shape}, target num poses: {target_num_poses}")
        return Trajectory(trajectory_poses, trajectory_sampling)
