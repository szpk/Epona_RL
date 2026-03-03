"""
Dataset class for GRPO training using OpenScene data.
This ensures training and evaluation use the same data source.
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import math
import yaml

# Add recogdrive to path to import navsim modules
recogdrive_path = Path(__file__).parent.parent.parent.parent / "recogdrive"
if str(recogdrive_path) not in sys.path:
    sys.path.insert(0, str(recogdrive_path))

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
# Removed StateSE2 and convert_absolute_to_relative_se2_array imports - using absolute poses like nuplan


def load_navtrain_log_names():
    """
    Load log_names from navtrain.yaml for GRPO training.
    This ensures we only use data that has corresponding metric cache.
    Similar to how run_navsim_eval_pipeline1.py loads navtest.yaml.
    """
    # 使用 Epona 项目中的 navtrain.yaml
    epona_root = Path(__file__).parent.parent
    navtrain_config_path = epona_root / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
    
    if navtrain_config_path.exists():
        print(f"Loading navtrain.yaml from: {navtrain_config_path}")
        with open(navtrain_config_path, 'r') as f:
            navtrain_cfg = yaml.safe_load(f)
        log_names = navtrain_cfg.get('log_names', None)
        if log_names:
            print(f"Loaded {len(log_names)} log_names from navtrain.yaml")
            return log_names
    
    print("Warning: navtrain.yaml not found or no log_names, using all data")
    return None


class NuPlanOpenScene(Dataset):
    """
    Dataset class for GRPO training using OpenScene data.
    This ensures training and evaluation use the same data source as recogdrive.
    """
    
    def __init__(
        self,
        openscene_root,
        split='train',
        condition_frames=3,
        block_size=1,
        downsample_fps=3,
        h=256,
        w=512,
        no_pose=False,
        scene_filter_config=None,
        use_grpo=False,  # GRPO 训练时启用 navtrain 过滤 + 固定 SceneFilter (num_future_frames=10) 以匹配 metric_cache
        grpo_metric_cache_path=None,  # Metric cache path for filtering scenes (only scenes in metric_cache will be used)
    ):
        """
        Initialize OpenScene dataset for GRPO training.
        
        Args:
            openscene_root: Root path to OpenScene data
            split: 'train' or 'val'
            condition_frames: Number of condition frames
            block_size: Block size for frame sampling
            downsample_fps: Downsample FPS
            h: Image height
            w: Image width
            no_pose: Whether to skip pose data
            scene_filter_config: SceneFilter configuration dict
        """
        self.split = split
        self.condition_frames = condition_frames
        self.block_size = block_size
        self.base_condition_frames = 4 if condition_frames > 10 else condition_frames
        self.h = h
        self.w = w
        self.no_pose = no_pose
        
        openscene_root = Path(openscene_root)
        self.openscene_root = openscene_root
        
        # Set up paths (same as recogdrive)
        if split == 'train':
            data_path = openscene_root / "meta_datas" / "trainval"
            sensor_blobs_path = openscene_root / "sensor_blobs" / "trainval"
        else:
            data_path = openscene_root / "meta_datas" / "test"
            sensor_blobs_path = openscene_root / "sensor_blobs" / "test"
        
        # Default scene filter configuration (similar to recogdrive)
        # SceneFilter accepts: num_history_frames, num_future_frames, frame_interval, has_route, max_scenes, log_names, tokens
        # Note: num_frames is a computed property (num_history_frames + num_future_frames), not a parameter
        # IMPORTANT: condition_frames is the number AFTER downsample, but SceneLoader needs original frames
        # So we need to multiply by downsample factor to get enough original frames
        # NavSim/OpenScene original FPS is 2 Hz (0.5s per frame), NOT 10 Hz like nuplan!
        ori_fps = 2  # NavSim original frequency is 2 Hz (0.5s per frame)
        downsample = ori_fps // downsample_fps  # Calculate downsample factor
        
        # Load navtrain log_names if use_grpo is True (for GRPO training)
        # This ensures we only load data that has corresponding metric cache (similar to recogdrive)
        log_names_filter = None
        if use_grpo and split == 'train':
            log_names_filter = load_navtrain_log_names()
        
        if scene_filter_config is None:
            # GRPO 训练时使用固定的 num_history_frames=4, num_future_frames=10
            # 这与 recogdrive 的 navtrain.yaml SceneFilter 配置一致，确保生成的 token 能匹配 metric_cache
            # 实际训练仍然是 4 帧推 8 帧，只是为了 token 匹配
            scene_filter_config = {
                'num_history_frames': 4,  # 与 recogdrive navtrain.yaml 一致
                'num_future_frames': 10,  # 与 recogdrive navtrain.yaml 一致，确保 token 匹配 metric_cache
                'frame_interval': 1,
                'has_route': True,
                'max_scenes': None,
                'log_names': log_names_filter,  # GRPO 训练时使用 navtrain 的 log_names 过滤
                'tokens': None,
            }
            if use_grpo:
                print(f"GRPO mode: Using fixed SceneFilter (num_history_frames=4, num_future_frames=10) to match metric_cache")
        
        scene_filter = SceneFilter(**scene_filter_config)
        
        # Build sensor config - enable front camera
        # SensorConfig uses lowercase attribute names (cam_f0, cam_l0, etc.)
        # This matches the Cameras dataclass which also uses lowercase attributes
        sensor_config = SensorConfig.build_no_sensors()
        sensor_config.cam_f0 = True  # Enable front camera
        
        # Create SceneLoader (same as recogdrive)
        # Note: recogdrive doesn't set load_image_path, so images are loaded as numpy arrays directly
        # Parameter order: data_path, sensor_blobs_path (matches SceneLoader.__init__ signature)
        self.scene_loader = SceneLoader(
            data_path=data_path,
            sensor_blobs_path=sensor_blobs_path,
            scene_filter=scene_filter,
            sensor_config=sensor_config,
            load_image_path=False,  # Load images directly as numpy arrays (same as recogdrive)
        )
        
        self.tokens = self.scene_loader.tokens
        
        # Filter tokens by metric_cache if GRPO training is enabled
        # This ensures we only use scenes that have corresponding metric cache
        if use_grpo and grpo_metric_cache_path is not None:
            from navsim.common.dataloader import MetricCacheLoader
            metric_cache_path = Path(grpo_metric_cache_path)
            if metric_cache_path.exists():
                print(f"Loading metric cache from: {metric_cache_path}")
                metric_cache_loader = MetricCacheLoader(metric_cache_path)
                cache_tokens = set(metric_cache_loader.tokens)
                print(f"Metric cache contains {len(cache_tokens)} scenes")
                
                # Filter tokens to only include those in metric_cache
                original_token_count = len(self.tokens)
                self.tokens = [token for token in self.tokens if token in cache_tokens]
                filtered_token_count = len(self.tokens)
                print(f"Filtered tokens: {original_token_count} -> {filtered_token_count} (kept {filtered_token_count/original_token_count*100:.1f}%)")
                
                # Also filter scene_loader.scene_frames_dicts to match
                self.scene_loader.scene_frames_dicts = {
                    token: scene_dict 
                    for token, scene_dict in self.scene_loader.scene_frames_dicts.items() 
                    if token in cache_tokens
                }
            else:
                print(f"Warning: Metric cache path does not exist: {metric_cache_path}. Not filtering tokens.")
        
        # NavSim/OpenScene original FPS is 2 Hz (0.5s per frame), NOT 10 Hz like nuplan!
        ori_fps = 2  # NavSim original frequency is 2 Hz
        self.downsample = ori_fps // downsample_fps  # Calculate downsample factor
        self.sensor_blobs_path = sensor_blobs_path
        
        # Calculate actual frames needed after downsample
        # condition_frames is the number of frames needed AFTER downsample
        # So we need condition_frames * downsample original frames
        self.actual_condition_frames = condition_frames
        self.actual_block_size = block_size
        
        print(f"Loaded {len(self.tokens)} scenes from OpenScene data")
        print(f"Data path: {data_path}")
        print(f"Sensor blobs path: {sensor_blobs_path}")
        print(f"Condition frames (after downsample): {self.actual_condition_frames}, Block size: {self.actual_block_size}, Downsample: {self.downsample}")
    
    def __len__(self):
        return len(self.tokens)
    
    def normalize_imgs(self, imgs):
        """Normalize images to [-1, 1] range."""
        imgs = imgs / 255.0
        imgs = (imgs - 0.5) * 2
        return imgs
    
    def get_pose_from_ego_status_local(self, x, y, yaw):
        """Convert [x, y, yaw] to 4x4 transformation matrix."""
        # Create 4x4 transformation matrix
        pose_matrix = np.eye(4, dtype=np.float32)
        pose_matrix[0, 3] = x
        pose_matrix[1, 3] = y
        pose_matrix[2, 3] = 0.0  # z is 0 for 2D pose
        
        # Rotation around z-axis (yaw)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        pose_matrix[0, 0] = cos_yaw
        pose_matrix[0, 1] = -sin_yaw
        pose_matrix[1, 0] = sin_yaw
        pose_matrix[1, 1] = cos_yaw
        
        return pose_matrix
    
    def __getitem__(self, index):
        """
        Get item from dataset.
        Returns: (imgs, rot_matrix, token)
        - imgs: (F, C, H, W) normalized images
        - rot_matrix: (F, 4, 4) pose matrices
        - token: string token for metric cache lookup
        """
        token = self.tokens[index]
        
        # Load scene from token (same as recogdrive)
        scene = self.scene_loader.get_scene_from_token(token)
        
        # Extract images and poses from frames
        imgs = []
        poses = []
        
        # Get frames from scene
        frames = scene.frames
        num_frames = len(frames)
        
        # Downsample frames
        # Take frames with downsample interval, up to the number we need
        frame_indices = list(range(0, num_frames, self.downsample))
        # Limit to the number of frames we actually need (condition_frames + block_size)
        frame_indices = frame_indices[:self.actual_condition_frames + self.actual_block_size]
        
        # Collect global ego poses first (Scene uses global coordinates)
        # IMPORTANT: Use absolute poses directly like nuplan dataset does, not relative poses
        # The model expects absolute poses in UTM coordinate system
        global_ego_poses = []
        for frame_idx in frame_indices:
            if frame_idx >= num_frames:
                break
            frame = frames[frame_idx]
            if frame.ego_status is not None:
                # Scene.frames use global coordinates (in_global_frame=True)
                global_ego_poses.append(frame.ego_status.ego_pose.copy())
            else:
                # Use identity pose if ego_status is missing
                global_ego_poses.append(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        
        # Now process frames with absolute coordinates (same as nuplan)
        for idx, frame_idx in enumerate(frame_indices):
            if frame_idx >= num_frames:
                break
            
            frame = frames[frame_idx]
            
            # Get image from front camera (cam_f0)
            # When load_image_path=False, cam_f0.image is already a numpy array (same as recogdrive)
            # If image loading fails, SceneLoader will raise an exception (same as recogdrive)
            if frame.cameras is None:
                raise ValueError(
                    f"Frame {frame_idx} has no cameras. "
                    f"Token: {token}, Scene metadata: {scene.scene_metadata.log_name}"
                )
            
            if not hasattr(frame.cameras, 'cam_f0'):
                raise ValueError(
                    f"Frame {frame_idx} cameras object has no cam_f0 attribute. "
                    f"Token: {token}, Available cameras: {dir(frame.cameras)}"
                )
            
            cam_f0 = frame.cameras.cam_f0
            if cam_f0 is None:
                raise ValueError(
                    f"Frame {frame_idx} cam_f0 is None. "
                    f"Token: {token}, This should not happen if sensor_config includes 'cam_f0'"
                )
            
            if cam_f0.image is None:
                raise ValueError(
                    f"Frame {frame_idx} cam_f0.image is None. "
                    f"Token: {token}, This indicates image loading failed in SceneLoader. "
                    f"Check sensor_blobs_path: {self.sensor_blobs_path}"
                )
            
            # Image is already loaded as numpy array (load_image_path=False)
            # Same as recogdrive: image should be a valid numpy array at this point
            img_array = cam_f0.image.copy()
            
            # Validate image array
            if not isinstance(img_array, np.ndarray):
                raise TypeError(
                    f"Frame {frame_idx} cam_f0.image is not a numpy array, got {type(img_array)}. "
                    f"Token: {token}, Expected numpy array (load_image_path=False)"
                )
            
            if img_array.size == 0:
                raise ValueError(
                    f"Frame {frame_idx} cam_f0.image is empty. "
                    f"Token: {token}, Image shape: {img_array.shape}"
                )
            
            # Ensure uint8 format
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            
            # Ensure 3 channels
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                img_array = np.repeat(img_array, 3, axis=2)
            elif len(img_array.shape) != 3 or img_array.shape[2] != 3:
                raise ValueError(
                    f"Frame {frame_idx} invalid image shape: {img_array.shape}. "
                    f"Token: {token}, Expected (H, W, 3) or (H, W)"
                )
            
            # Resize image
            img = Image.fromarray(img_array)
            img = img.resize((self.w, self.h), Image.BICUBIC)
            img_array = np.array(img)
            imgs.append(img_array)
            
            # Get pose from absolute coordinates (same as nuplan dataset)
            # Use absolute UTM coordinates directly, not relative coordinates
            if idx < len(global_ego_poses):
                x, y, yaw = global_ego_poses[idx]
                pose = self.get_pose_from_ego_status_local(x, y, yaw)
                poses.append(pose)
            else:
                print(f"Warning: No ego_pose found for frame {frame_idx}, using identity pose")
                pose = np.eye(4, dtype=np.float32)
                poses.append(pose)
        
        # Convert to tensors
        imgs_tensor = []
        poses_tensor = []
        for img, pose in zip(imgs, poses):
            imgs_tensor.append(torch.from_numpy(img.copy()).permute(2, 0, 1))
            poses_tensor.append(torch.from_numpy(pose.copy()))
        
        imgs = self.normalize_imgs(torch.stack(imgs_tensor, 0))  # (F, C, H, W)
        rot_matrix = torch.stack(poses_tensor, 0).float()  # (F, 4, 4)
        
        # Generate token for metric cache lookup
        # Token format: just initial_token (hash), matching MetricCacheLoader's key extraction
        # MetricCacheLoader uses cache_path.split("/")[-2] which gets only the hash part
        initial_token = scene.scene_metadata.initial_token
        token_for_cache = initial_token
        
        if self.no_pose:
            return imgs, torch.tensor(0.0), token_for_cache
        else:
            return imgs, rot_matrix, token_for_cache
