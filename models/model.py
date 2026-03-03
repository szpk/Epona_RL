import os
import time
import torch
import torch.nn as nn
import torch.nn.functional
import random
import copy
import sys
import numpy as np
from pathlib import Path
from dataclasses import asdict
from einops import rearrange
from utils.preprocess import get_rel_pose, get_rel_traj
from models.stt import SpatialTemporalTransformer
from models.flux_dit import FluxParams, FluxDiT
from models.traj_dit import TrajDiT, TrajParams
from models.modules.tokenizer import poses_to_indices, yaws_to_indices
from utils.fft_utils import freq_mix, ideal_low_pass_filter
from models.modules.sampling import prepare_ids, get_schedule

# Add recogdrive to path for PDM scorer imports
# Current file: RL/Epona/models/model.py
# Target: RL/recogdrive/
# So we need: .parent (models) -> .parent (Epona) -> .parent (RL) -> /recogdrive
recogdrive_path = Path(__file__).parent.parent.parent / "recogdrive"
if str(recogdrive_path) not in sys.path:
    sys.path.insert(0, str(recogdrive_path))

try:
    from navsim.common.dataclasses import Trajectory
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer,
        PDMScorerConfig,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
        PDMSimulator,
    )
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )
    PDM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PDM scorer imports failed: {e}. PDM scoring will not be available.")
    PDM_AVAILABLE = False

class TrainTransformersDiT(nn.Module):
    def __init__(
        self,
        args,
        local_rank=-1, 
        load_path=None, 
        condition_frames=3,
    ):
        super().__init__()
        self.local_rank = local_rank
        self.args = args
        self.condition_frames = condition_frames
        self.vae_emb_dim = self.args.vae_embed_dim * self.args.patch_size ** 2
        self.image_size = self.args.image_size
        self.traj_len = self.args.traj_len
        self.h, self.w = (self.image_size[0]//(self.args.downsample_size*self.args.patch_size),  self.image_size[1]//(self.args.downsample_size*self.args.patch_size))
        self.pkeep = args.pkeep

        self.img_token_size = self.h * self.w
        self.pose_x_vocab_size = self.args.pose_x_vocab_size
        self.pose_y_vocab_size = self.args.pose_y_vocab_size
        self.yaw_vocab_size = self.args.yaw_vocab_size
        self.pose_x_bound = self.args.pose_x_bound
        self.pose_y_bound = self.args.pose_y_bound
        self.yaw_bound = self.args.yaw_bound

        self.pose_token_size = 2 * self.args.block_size
        self.yaw_token_size = 1 * self.args.block_size
        self.traj_token_size = self.pose_token_size + self.yaw_token_size
        self.total_token_size = self.img_token_size + self.pose_token_size + self.yaw_token_size
        self.token_size_dict = {
            'img_tokens_size': self.img_token_size,
            'pose_tokens_size': self.pose_token_size,
            'yaw_token_size': self.yaw_token_size,
            'total_tokens_size': self.total_token_size
        }
        
        self.model = SpatialTemporalTransformer(
            block_size=condition_frames*(self.total_token_size),
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            pose_x_vocab_size=self.pose_x_vocab_size,
            pose_y_vocab_size=self.pose_y_vocab_size,
            yaw_vocab_size=self.yaw_vocab_size,
            latent_size=(self.h, self.w), 
            # L=self.img_token_size, 
            local_rank=local_rank, 
            condition_frames=self.condition_frames, 
            token_size_dict=self.token_size_dict,
            vae_emb_dim = self.vae_emb_dim,
            temporal_block=self.args.block_size
        )
        self.model.cuda()
        
        self.dit = FluxDiT(FluxParams(
            in_channels=self.vae_emb_dim,        # origin: 64
            out_channels=self.vae_emb_dim,
            vec_in_dim=args.n_embd*(self.total_token_size-self.img_token_size),              # origin: 768
            context_in_dim=args.n_embd,          # origin: 4096
            hidden_size=args.n_embd_dit,         # origin: 3072
            mlp_ratio=4.0,
            num_heads=args.n_head_dit,           # origin: 24
            depth=args.n_layer[1],               # origin: 19
            depth_single_blocks=args.n_layer[2], # origin: 38
            axes_dim=args.axes_dim_dit,
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,                # origin: True
        ))
        self.dit.cuda()
        
        self.traj_dit = TrajDiT(TrajParams(
            in_channels=self.traj_token_size,
            out_channels=self.traj_token_size,
            context_in_dim=args.n_embd,
            hidden_size=args.n_embd_dit_traj,
            mlp_ratio=4.0,
            num_heads=args.n_head_dit_traj,
            depth=args.n_layer_traj[0],
            depth_single_blocks=args.n_layer_traj[1],
            axes_dim=args.axes_dim_dit_traj,
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ))
        self.traj_dit.cuda()
        
        bs = args.batch_size * condition_frames
        self.img_ids, self.cond_ids, self.traj_ids = prepare_ids(bs, self.h, self.w, self.total_token_size, self.traj_len)                
        self.lambda_yaw_pose = self.args.lambda_yaw_pose

        # NOTE: GRPO initialization is deferred to after checkpoint loading
        # and DDP wrapping. Call model.init_grpo_after_load() from the training
        # script after loading pretrained weights. This ensures:
        # 1. old_policy gets the pretrained weights (not random init)
        # 2. old_policy is not registered as a DDP-tracked submodule (avoids OOM)
        self._grpo_initialized = False

        if load_path is not None:
            # load_model_path = os.path.join(load_path, 'tvar'+'_%d.pkl'%(resume_step))
            state_dict = torch.load(load_path, map_location='cpu')["model_state_dict"]
            model_state_dict = self.model.state_dict()
            for k in model_state_dict.keys():
                model_state_dict[k] = state_dict['module.model.'+k]
            self.model.load_state_dict(model_state_dict)
            traj_dit_state_dict = self.traj_dit.state_dict()
            if any(k.startswith('module.traj_dit.') for k in state_dict.keys()):
                for k in traj_dit_state_dict.keys():
                    traj_dit_state_dict[k] = state_dict['module.traj_dit.'+k]
                self.traj_dit.load_state_dict(traj_dit_state_dict)
            dit_state_dict = self.dit.state_dict()
            for k in dit_state_dict.keys():
                dit_state_dict[k] = state_dict['module.dit.'+k]
            self.dit.load_state_dict(dit_state_dict)
            print(f"Successfully load model from {load_path}")

    def normalize_traj(self, traj_targets):
        traj_targets[..., 0:1] = 2 * traj_targets[..., 0:1] / self.pose_x_bound - 1
        traj_targets[..., 1:2] /= self.pose_y_bound
        traj_targets[..., 2:3] /= self.yaw_bound
        return traj_targets
    
    def denormalize_traj(self, traj_targets):
        traj_targets[..., 0:1] = (traj_targets[..., 0:1] + 1) * self.pose_x_bound / 2
        traj_targets[..., 1:2] *= self.pose_y_bound
        traj_targets[..., 2:3] *= self.yaw_bound
        return traj_targets
        
    def model_forward(self, feature_total, rot_matrix, targets, rel_pose_cond=None, rel_yaw_cond=None, step=0):
        if (rel_pose_cond is not None) and (rel_yaw_cond is not None):
            with torch.cuda.amp.autocast(enabled=False):
                rel_pose_gt, rel_yaw_gt = get_rel_pose(rot_matrix[:, (self.condition_frames-1)*self.args.block_size:(self.condition_frames+1)*self.args.block_size])
            rel_pose_total = torch.cat([rel_pose_cond, rel_pose_gt[:, -1:]], dim=1)
            rel_yaw_total = torch.cat([rel_yaw_cond, rel_yaw_gt[:, -1:]], dim=1)
        else:
            with torch.cuda.amp.autocast(enabled=False):
                rel_pose_total, rel_yaw_total = get_rel_pose(rot_matrix[:, :(self.condition_frames+1)*self.args.block_size])

        pose_indices_total = poses_to_indices(rel_pose_total, self.pose_x_vocab_size, self.pose_y_vocab_size)  # (b, t+n, 2)
        yaw_indices_total = yaws_to_indices(rel_yaw_total, self.yaw_vocab_size)  # (b, t+n, 1)
        logits = self.model(feature_total, pose_indices_total, yaw_indices_total, drop_feature=self.args.drop_feature) # 输入 b F L c 进去
        stt_features = logits['logits']
        pose_emb = logits['pose_emb']
        
        with torch.cuda.amp.autocast(enabled=False):
            traj_poses, traj_yaws = get_rel_traj(rot_matrix, self.condition_frames, self.traj_len)
        traj_targets = torch.cat([traj_poses, traj_yaws], dim=-1)   # (B, F, N, 3)
        traj_targets = traj_targets.reshape(-1, *traj_targets.shape[2:])
        traj_targets = self.normalize_traj(traj_targets)
        traj_targets = traj_targets.to(dtype=torch.bfloat16)
        yaw_pose_loss_terms = self.traj_dit.training_losses(
                        traj=traj_targets,
                        traj_ids=self.traj_ids,
                        cond=stt_features,
                        cond_ids=self.cond_ids,
                        t=torch.rand((traj_targets.shape[0], 1, 1), device=traj_targets.device),
                        return_predict=self.args.return_predict_traj
                    )
        yaw_pose_loss = self.lambda_yaw_pose * yaw_pose_loss_terms['loss']
        traj_predict = yaw_pose_loss_terms['predict']
        
        loss_terms = self.dit.training_losses(
                        img=targets,
                        img_ids=self.img_ids,
                        cond=stt_features,
                        cond_ids=self.cond_ids,
                        t=torch.rand((targets.shape[0], 1, 1), device=targets.device),
                        y=pose_emb,
                        return_predict=self.args.return_predict
                    )
        diff_loss = loss_terms['loss']
        predict = loss_terms['predict']
                    
        loss_all = diff_loss + yaw_pose_loss
        loss = {
            "loss_all": loss_all,
            "loss_diff": diff_loss,
            "loss_yaw_pose": yaw_pose_loss,
            "predict": None if not self.args.return_predict else predict,
            "predict_traj": None if not self.args.return_predict_traj else traj_predict,
        }
        return loss

    def step_train(self, latents, rot_matrix, latents_gt, rel_pose_cond=None, rel_yaw_cond=None, latents_aug=None, step=0, tokens_list=None):
        self.model.train()
        
        # Check if GRPO training is enabled
        use_grpo = hasattr(self.args, 'grpo') and self.args.grpo
        
        if use_grpo:
            # GRPO training path
            if latents_aug is None:
                latents_total = torch.cat([latents, latents_gt], dim=1)
            else:
                latents_total = latents_aug

            pro = random.random()
            if pro < self.args.mask_data:
                mask = torch.bernoulli(random.uniform(0.7, 1) * torch.ones_like(latents_total))
                mask = mask.round().to(dtype=torch.int64)
                noise = torch.randn_like(latents_total)
                
                if random.random() < 0.5:
                    LPF = ideal_low_pass_filter(latents_total.shape, d_s=random.uniform(0.5, 1), dims=(-1,)).cuda()
                    latents_total = freq_mix(latents_total, noise, LPF, dims=(-1,))
                else:
                    latents_total = mask * latents_total + (1 - mask) * noise
            
            targets = torch.cat([latents, latents_gt], dim=1)[:, 1:]
            targets = rearrange(targets, 'B F L C -> (B F) L C')
            loss = self.forward_grpo(latents_total, rot_matrix, targets, rel_pose_cond=rel_pose_cond, rel_yaw_cond=rel_yaw_cond, tokens_list=tokens_list, step=step)
            return loss
        else:
            # Standard training path
            if latents_aug is None:
                latents_total = torch.cat([latents, latents_gt], dim=1)
            else:
                latents_total = latents_aug

            pro = random.random()
            if  pro < self.args.mask_data:
                mask = torch.bernoulli(random.uniform(0.7, 1) * torch.ones_like(latents_total))
                mask = mask.round().to(dtype=torch.int64)
                noise = torch.randn_like(latents_total)
                
                if random.random() < 0.5:
                    LPF = ideal_low_pass_filter(latents_total.shape, d_s=random.uniform(0.5, 1), dims=(-1,)).cuda()
                    latents_total = freq_mix(latents_total, noise, LPF, dims=(-1,))
                else:
                    latents_total = mask * latents_total + (1 - mask) * noise
                    
            targets = torch.cat([latents, latents_gt], dim=1)[:, 1:]
            targets = rearrange(targets, 'B F L C -> (B F) L C')
            loss = self.model_forward(latents_total, rot_matrix, targets, rel_pose_cond=rel_pose_cond, rel_yaw_cond=rel_yaw_cond, step=step)
            return loss

    def forward(self, latents, rot_matrix, latents_gt, rel_pose_cond=None, rel_yaw_cond=None, latents_aug=None, sample_last=True, step=0, tokens_list=None, **kwargs):
        if self.training:
            return self.step_train(latents, rot_matrix, latents_gt, rel_pose_cond, rel_yaw_cond, latents_aug, step, tokens_list)
        else:
            return self.step_eval(latents, rot_matrix, sample_last=sample_last, **kwargs)
    
    @torch.no_grad()
    def step_eval(self, latents, rel_pose, rel_yaw, sample_last=True, self_pred_traj=True, traj_only=False, use_sde=False):
        self.model.eval()
        start_time = time.time()
        pose_total = poses_to_indices(rel_pose, self.pose_x_vocab_size, self.pose_y_vocab_size)  # (b, t+1, 2)
        yaw_total = yaws_to_indices(rel_yaw, self.yaw_vocab_size)  # (b, t+1, 1)
        
        stt_features, pose_emb = self.model.evaluate(latents, pose_total, yaw_total, sample_last=sample_last) # (b F) L c
        interval = time.time() - start_time
        print("MST time:{:.2f}", interval)
        bsz = stt_features.shape[0]
        img_ids, cond_ids, traj_ids = prepare_ids(bsz, self.h, self.w, self.total_token_size, self.traj_len)
        
        start_time = time.time()
        self.traj_dit.eval()
        noise_traj = torch.randn(bsz, self.traj_len, self.traj_token_size).to(stt_features)
        timesteps_traj = get_schedule(int(self.args.num_sampling_steps), self.traj_len)
        # Use SDE with 8 trajectories if use_sde=True, otherwise use ODE with 1 trajectory
        if use_sde:
            predict_traj = self.traj_dit.sample(noise_traj, traj_ids, stt_features, cond_ids, timesteps_traj, deterministic=False, num_samples=8)
        else:
            predict_traj = self.traj_dit.sample(noise_traj, traj_ids, stt_features, cond_ids, timesteps_traj, deterministic=True, num_samples=1)
        
        # Handle multiple trajectories: (num_samples, B, H, D) -> denormalize all
        if predict_traj.ndim == 4:  # Multiple trajectories: (num_samples, B, H, D)
            predict_traj = self.denormalize_traj(predict_traj)  # Denormalize all trajectories
        else:  # Single trajectory: (B, H, D)
            predict_traj = self.denormalize_traj(predict_traj)
        interval = time.time() - start_time
        print("TrajDiT time:{:.2f}", interval)
        
        if traj_only:
            predict_latents = None
        else:
            # For self_pred_traj, use the first trajectory (index 0) if multiple trajectories exist
            if predict_traj.ndim == 4:
                traj_for_pose = predict_traj[0]  # Use first trajectory: (B, H, D)
            else:
                traj_for_pose = predict_traj  # Single trajectory: (B, H, D)
            
            if self_pred_traj:
                predict_pose, predict_yaw = traj_for_pose[:, 0:1, 0:2], traj_for_pose[:, 0:1, 2:3]
                predict_pose = poses_to_indices(predict_pose, self.pose_x_vocab_size, self.pose_y_vocab_size)  # (b, 1, 2)
                predict_yaw = yaws_to_indices(predict_yaw, self.yaw_vocab_size)  # (b, 1, 1)
                pose_emb = self.model.get_pose_emb(predict_pose, predict_yaw)
            
            start_time = time.time()
            self.dit.eval()
            noise = torch.randn(bsz, self.img_token_size, self.vae_emb_dim).to(stt_features)
            timesteps = get_schedule(int(self.args.num_sampling_steps), self.img_token_size)
            predict_latents = self.dit.sample(noise, img_ids, stt_features, cond_ids, pose_emb, timesteps)
            predict_latents = rearrange(predict_latents, 'b (h w) c -> b h w c', h=self.h, w=self.w)
            interval = time.time() - start_time
            print("VisDiT time:{:.2f}", interval)
            
        return predict_traj, predict_latents

    @torch.no_grad()
    def generate_gt_pose_gt_yaw(self, latents, rel_pose, rel_yaw, sample_last=True):
        self.model.eval()
        pose_total = poses_to_indices(rel_pose, self.pose_x_vocab_size, self.pose_y_vocab_size)  # (b, t+1, 2)
        yaw_total = yaws_to_indices(rel_yaw, self.yaw_vocab_size)  # (b, t+1, 1)
        
        stt_features, pose_emb = self.model.evaluate(latents, pose_total[:, :(self.condition_frames+1)*self.args.block_size], yaw_total[:, :(self.condition_frames+1)*self.args.block_size], sample_last=sample_last) # (b F) L c

        self.dit.eval()
        bsz = stt_features.shape[0]
        noise = torch.randn(bsz, self.img_token_size, self.vae_emb_dim).to(stt_features)
        timesteps = get_schedule(int(self.args.num_sampling_steps), self.img_token_size)
        img_ids, cond_ids, traj_ids = prepare_ids(bsz, self.h, self.w, self.total_token_size, self.traj_len)
        predict_latents = self.dit.sample(noise, img_ids, stt_features, cond_ids, pose_emb, timesteps)
        predict_latents = rearrange(predict_latents, 'b (h w) c -> b h w c', h=self.h, w=self.w)
        return predict_latents
     
    def save_model(self, path, epoch, rank=0):
        if rank == 0:
            torch.save(self.model.state_dict(),'{}/tvar_{}.pkl'.format(path, str(epoch)))
    
    def init_grpo_after_load(self, args=None):
        """
        Public method to initialize GRPO after checkpoint loading and DDP wrapping.
        Must be called from the training script AFTER:
        1. Checkpoint loading (so old_policy gets pretrained weights)
        2. DDP wrapping (so old_policy is not tracked by DDP)
        """
        if args is None:
            args = self.args
        if hasattr(args, 'grpo') and args.grpo and not self._grpo_initialized:
            self._init_grpo(args)
            self._grpo_initialized = True
            print("GRPO initialized after checkpoint loading (old_policy has pretrained weights)")

    def _init_grpo(self, args):
        """
        Initializes components and hyperparameters for GRPO training.
        
        Args:
            args: Configuration arguments containing GRPO settings.
        """
        # GRPO hyperparameters
        self.grpo_sample_time = getattr(args, 'grpo_sample_time', 8)
        self.grpo_gamma_denoising = getattr(args, 'grpo_gamma_denoising', 0.6)
        self.grpo_bc_coeff = getattr(args, 'grpo_bc_coeff', 0.1)
        self.grpo_use_bc_loss = getattr(args, 'grpo_use_bc_loss', True)
        self.grpo_clip_advantage_lower_quantile = getattr(args, 'grpo_clip_advantage_lower_quantile', 0.0)
        self.grpo_clip_advantage_upper_quantile = getattr(args, 'grpo_clip_advantage_upper_quantile', 1.0)
        self.grpo_min_logprob_denoising_std = getattr(args, 'grpo_min_logprob_denoising_std', 0.1)
        
        # Load reference policy (old_policy) for BC loss
        reference_policy_checkpoint = getattr(args, 'grpo_reference_policy_checkpoint', None)
        if reference_policy_checkpoint is not None:
            try:
                state_dict = torch.load(reference_policy_checkpoint, map_location='cpu')
                # Handle different checkpoint formats
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                
                # Filter and load traj_dit weights
                traj_dit_state_dict = self.traj_dit.state_dict()
                filtered_ckpt = {}
                for k, v in state_dict.items():
                    # Handle different key prefixes
                    if k.startswith('traj_dit.'):
                        k2 = k[len('traj_dit.'):]
                    elif k.startswith('module.traj_dit.'):
                        k2 = k[len('module.traj_dit.'):]
                    else:
                        k2 = k
                    
                    if k2 in traj_dit_state_dict and v.shape == traj_dit_state_dict[k2].shape:
                        filtered_ckpt[k2] = v
                    else:
                        print(f"Skip loading '{k}' → '{k2}' (checkpoint shape {tuple(v.shape)} vs model shape {tuple(traj_dit_state_dict.get(k2, v).shape)})")
                
                self.traj_dit.load_state_dict(filtered_ckpt, strict=False)
                print(f"Successfully loaded reference policy from {reference_policy_checkpoint}")
            except FileNotFoundError:
                print(f"Warning: GRPO reference policy checkpoint not found at {reference_policy_checkpoint}. Skipping loading.")
            except Exception as e:
                print(f"Warning: Failed to load GRPO reference policy checkpoint: {e}")
        
        # Create old_policy (frozen copy for BC loss)
        # IMPORTANT: Store in a list to avoid nn.Module registration!
        # If registered as self.old_policy = Module(), DDP would track it,
        # allocate gradient buffers, and cause OOM during DDP init.
        old_policy = copy.deepcopy(self.traj_dit)
        old_policy.eval()
        for param in old_policy.parameters():
            param.requires_grad = False
        self._old_policy_container = [old_policy]  # Hidden from DDP
        
        # Set min_logprob_denoising_std for logprob computation
        min_logprob_std = getattr(args, 'grpo_min_logprob_denoising_std', 0.1)
        self.traj_dit.min_logprob_denoising_std = min_logprob_std
        self._old_policy_container[0].min_logprob_denoising_std = min_logprob_std
        
        # Initialize PDM scorer components if available
        self.grpo_metric_cache_path = getattr(args, 'grpo_metric_cache_path', None)
        self.use_pdm_scorer = PDM_AVAILABLE and self.grpo_metric_cache_path is not None
        
        if self.use_pdm_scorer:
            # Initialize metric cache loader
            metric_cache_path = Path(self.grpo_metric_cache_path)
            if not metric_cache_path.exists():
                print(f"Warning: Metric cache path does not exist: {metric_cache_path}. PDM scoring disabled.")
                self.use_pdm_scorer = False
            else:
                self.metric_cache_loader = MetricCacheLoader(metric_cache_path)
                
                # Initialize PDM simulator and scorer
                # NavSim data is at 2Hz (0.5s interval), model outputs 8 waypoints for 4s
                proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
                self.simulator = PDMSimulator(proposal_sampling)
                
                # Use default scorer config or from args
                scorer_config = getattr(args, 'grpo_scorer_config', None)
                if scorer_config is None:
                    scorer_config = PDMScorerConfig(
                        progress_weight=10.0,
                        ttc_weight=5.0,
                        comfortable_weight=2.0
                    )
                self.train_scorer = PDMScorer(proposal_sampling, scorer_config)
                print(f"PDM scorer initialized with metric cache from: {metric_cache_path}")
        else:
            self.metric_cache_loader = None
            self.simulator = None
            self.train_scorer = None
            if not PDM_AVAILABLE:
                print("Warning: PDM scorer not available. Using placeholder reward function.")
            elif self.grpo_metric_cache_path is None:
                print("Warning: grpo_metric_cache_path not set. Using placeholder reward function.")
        
        print("GRPO initialized successfully")
    
    def reward_fn(
        self,
        pred_traj: torch.Tensor,
        tokens_list=None,
        cache_dict=None,
    ) -> torch.Tensor:
        """
        Calculates rewards for a batch of predicted trajectories using PDM scorer.
        
        Args:
            pred_traj (torch.Tensor): Predicted trajectories of shape (B, H, D).
                D should be 3: (x, y, heading) in ego frame.
            tokens_list: Optional list of tokens for metric cache lookup.
            cache_dict: Optional dictionary of metric caches (pre-loaded).
        
        Returns:
            torch.Tensor: Rewards of shape (B,).
        """
        B = pred_traj.shape[0]
        device = pred_traj.device
        
        # Use PDM scorer if available
        if self.use_pdm_scorer and tokens_list is not None:
            # Convert trajectories to numpy and create Trajectory objects
            pred_np = pred_traj.detach().cpu().numpy()  # (B, H, D)
            
            rewards = []
            for i, token in enumerate(tokens_list):
                try:
                    # Load metric cache if not in cache_dict
                    if cache_dict is None or token not in cache_dict:
                        if hasattr(self, 'metric_cache_loader'):
                            metric_cache = self.metric_cache_loader.get_from_token(token)
                        else:
                            # Fallback: use placeholder reward
                            rewards.append(0.0)
                            continue
                    else:
                        metric_cache = cache_dict[token]
                    
                    # Convert trajectory to Trajectory dataclass
                    # pred_np[i] shape: (H, 3) -> (num_poses, 3) with (x, y, yaw_degrees)
                    # NavSim data is at 2Hz (0.5s interval), model outputs traj_len waypoints
                    num_poses = pred_np[i].shape[0]
                    time_horizon = num_poses * 0.5  # 2Hz = 0.5s per waypoint
                    trajectory_sampling = TrajectorySampling(
                        time_horizon=time_horizon,
                        interval_length=0.5  # NavSim is 2Hz
                    )
                    
                    # IMPORTANT: Convert yaw from degrees to radians!
                    # denormalize_traj outputs yaw in degrees (from radians_to_degrees),
                    # but PDM scorer expects radians (same conversion as EponaAgent)
                    pred_poses = pred_np[i].copy()
                    pred_poses[:, 2] = np.deg2rad(pred_poses[:, 2])
                    
                    trajectory = Trajectory(
                        poses=pred_poses.astype(np.float32),
                        trajectory_sampling=trajectory_sampling
                    )
                    
                    # Compute PDM score
                    pdm_result = pdm_score(
                        metric_cache=metric_cache,
                        model_trajectory=trajectory,
                        future_sampling=self.simulator.proposal_sampling,
                        simulator=self.simulator,
                        scorer=self.train_scorer,
                    )
                    
                    # Extract score from PDMResults
                    rewards.append(asdict(pdm_result)["score"])
                    
                except Exception as e:
                    # Fallback: use zero reward if scoring fails
                    print(f"Warning: PDM scoring failed for token {token}: {e}")
                    rewards.append(0.0)
            
            return torch.tensor(rewards, device=device, dtype=pred_traj.dtype).detach()
        else:
            # Fallback: return zero rewards
            if not self.use_pdm_scorer:
                print("Warning: PDM scorer not available. Returning zero rewards.")
            elif tokens_list is None:
                print("Warning: tokens_list is None. Returning zero rewards.")
            return torch.zeros(B, device=device, dtype=pred_traj.dtype)
    
    def forward_grpo(
        self,
        feature_total,
        rot_matrix,
        targets,
        rel_pose_cond=None,
        rel_yaw_cond=None,
        tokens_list=None,
        step=0,
    ):
        """
        Computes the Diffusion-GRPO loss.
        
        Args:
            feature_total: Input features.
            rot_matrix: Rotation matrices.
            targets: Target images.
            rel_pose_cond: Optional relative pose conditioning.
            rel_yaw_cond: Optional relative yaw conditioning.
            tokens_list: Optional list of tokens for reward computation.
            step: Training step.
        
        Returns:
            dict: Dictionary containing loss components.
        """
        # Set frozen modules to eval mode during GRPO training
        if hasattr(self, '_old_policy_container') and self._old_policy_container:
            self._old_policy_container[0].eval()
        
        # Get conditioning features (similar to model_forward)
        if (rel_pose_cond is not None) and (rel_yaw_cond is not None):
            with torch.cuda.amp.autocast(enabled=False):
                rel_pose_gt, rel_yaw_gt = get_rel_pose(rot_matrix[:, (self.condition_frames-1)*self.args.block_size:(self.condition_frames+1)*self.args.block_size])
            rel_pose_total = torch.cat([rel_pose_cond, rel_pose_gt[:, -1:]], dim=1)
            rel_yaw_total = torch.cat([rel_yaw_cond, rel_yaw_gt[:, -1:]], dim=1)
        else:
            with torch.cuda.amp.autocast(enabled=False):
                rel_pose_total, rel_yaw_total = get_rel_pose(rot_matrix[:, :(self.condition_frames+1)*self.args.block_size])

        pose_indices_total = poses_to_indices(rel_pose_total, self.pose_x_vocab_size, self.pose_y_vocab_size)
        yaw_indices_total = yaws_to_indices(rel_yaw_total, self.yaw_vocab_size)
        logits = self.model(feature_total, pose_indices_total, yaw_indices_total, drop_feature=self.args.drop_feature)
        stt_features = logits['logits']
        
        # stt_features shape: (B_orig * F, L, C) where:
        # - B_orig: original batch size
        # - F: number of frames per sample
        # - L: sequence length
        # - C: feature dimension
        B_total = stt_features.shape[0]  # B_orig * F
        G = self.grpo_sample_time
        
        # Compute rewards
        tokens_rep = None
        cache_dict = None
        if tokens_list is not None:
            # tokens_list length is B_orig (original batch size from DataLoader)
            # B_total = B_orig * F (where F is the number of frames per sample)
            # We need tokens_rep length to be B_total * G = B_orig * F * G
            # So we need to repeat each token F * G times
            B_orig = len(tokens_list)
            if B_total % B_orig != 0:
                raise ValueError(
                    f"Batch size mismatch: stt_features.shape[0]={B_total} is not divisible by "
                    f"tokens_list length={B_orig}. This suggests a data format error."
                )
            F = B_total // B_orig  # Number of frames per sample
            tokens_rep = [tok for tok in tokens_list for _ in range(F * G)]
            
            # Pre-load metric caches for efficiency (optional optimization)
            if self.use_pdm_scorer and hasattr(self, 'metric_cache_loader'):
                unique_tokens = set(tokens_list)
                cache_dict = {}
                for token in unique_tokens:
                    try:
                        cache_dict[token] = self.metric_cache_loader.get_from_token(token)
                    except Exception as e:
                        print(f"Warning: Failed to load metric cache for token {token}: {e}")
        
        # Prepare IDs for repeated sampling
        stt_features_rep = stt_features.repeat_interleave(G, 0)  # (B_orig * F * G, L, C)
        bsz_rep = stt_features_rep.shape[0]
        img_ids_rep, cond_ids_rep, traj_ids_rep = prepare_ids(bsz_rep, self.h, self.w, self.total_token_size, self.traj_len, device=stt_features.device)
        
        # Sample multiple trajectories using SDE (following flow_grpo)
        self.traj_dit.train()
        noise_traj = torch.randn(bsz_rep, self.traj_len, self.traj_token_size, device=stt_features.device, dtype=stt_features.dtype)
        timesteps_traj = get_schedule(int(self.args.num_sampling_steps), self.traj_len)
        noise_level = getattr(self.args, 'grpo_noise_level', 0.7)
        
        # Phase 1: Sample trajectories with SDE and collect log probs (no grad)
        with torch.no_grad():
            all_latents, old_log_probs, trajs_normalized, sde_timesteps = self.traj_dit.sample_chain(
                noise_traj, traj_ids_rep, stt_features_rep, cond_ids_rep, timesteps_traj,
                deterministic=False, noise_level=noise_level
            )
        
        # Denormalize trajectories for reward computation
        trajs = self.denormalize_traj(trajs_normalized)
        
        # Compute rewards
        rewards = self.reward_fn(trajs, tokens_rep, cache_dict)
        
        # Compute advantages
        # rewards shape: (B_orig * F * G,)
        # Reshape to (B_orig * F, G) for advantage computation
        rewards_matrix = rewards.view(B_total, G)
        mean_r = rewards_matrix.mean(dim=1, keepdim=True)
        std_r = rewards_matrix.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_matrix - mean_r) / std_r).view(-1).detach()
        
        # Clip advantages
        adv_min = torch.quantile(advantages, self.grpo_clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, self.grpo_clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=adv_min, max=adv_max)
        
        # Phase 2: Re-compute log probs under current policy (with grad)
        # Following flow_grpo's generate_image_learn pattern
        new_log_probs = self.traj_dit.get_logprobs(
            traj_ids_rep, stt_features_rep, cond_ids_rep, all_latents, timesteps_traj,
            noise_level=noise_level
        )  # list of K scalar log_probs
        
        num_denoising_steps = len(new_log_probs)
        
        # Compute discounted advantages for each denoising step
        denoising_indices = torch.arange(num_denoising_steps, device=advantages.device)
        discount = (self.grpo_gamma_denoising ** (num_denoising_steps - denoising_indices - 1))
        
        # Compute policy loss step by step (per-sample GRPO)
        # After fixing _sde_step_with_logprob, log_probs are now per-sample: shape (B_total * G,)
        policy_loss = torch.tensor(0.0, device=stt_features.device, requires_grad=True)
        clipfrac_list = []
        
        clip_range_lt = getattr(self.args, 'grpo_clip_range_lt', 0.2)
        clip_range_gt = getattr(self.args, 'grpo_clip_range_gt', 0.28)
        
        for k in range(num_denoising_steps):
            # Per-sample importance ratio: exp(new_logprob - old_logprob)
            # old_lp, new_lp shape: (B_total * G,) — per-sample log probs
            old_lp = old_log_probs[k].detach()
            new_lp = new_log_probs[k]
            ratio = torch.exp(new_lp - old_lp)  # (B_total * G,)
            
            # Per-sample advantage, weighted by denoising discount
            # advantages shape: (B_total * G,), already normalized per group
            step_advantage = advantages * discount[k]  # (B_total * G,)
            
            # Clipped surrogate loss (per-sample, then mean over batch)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_range_lt, 1.0 + clip_range_gt)
            step_loss = -torch.min(ratio * step_advantage, clipped_ratio * step_advantage).mean()
            
            policy_loss = policy_loss + step_loss
            clipfrac_list.append(
                ((ratio - 1.0).abs() > max(clip_range_lt, clip_range_gt)).float().mean().item()
            )
        
        policy_loss = policy_loss / num_denoising_steps
        total_loss = policy_loss
        
        # BC loss (behavioral cloning loss)
        # Use ODE (deterministic) sampling from old_policy to avoid distributional shift!
        # Previously used SDE sampling which trains the model to match noisy trajectories,
        # causing slight performance degradation when evaluated with ODE.
        bc_loss = 0.0
        if self.grpo_use_bc_loss and hasattr(self, '_old_policy_container') and self._old_policy_container:
            old_policy = self._old_policy_container[0]
            with torch.no_grad():
                # Sample from old policy using ODE (deterministic) for clean teacher trajectories
                noise_traj_old = torch.randn(B_total, self.traj_len, self.traj_token_size, device=stt_features.device, dtype=stt_features.dtype)
                img_ids, cond_ids, traj_ids = prepare_ids(B_total, self.h, self.w, self.total_token_size, self.traj_len, device=stt_features.device)
                teacher_latents, _, _, _ = old_policy.sample_chain(
                    noise_traj_old, traj_ids, stt_features, cond_ids, timesteps_traj,
                    deterministic=True, noise_level=noise_level  # ODE: deterministic=True
                )
            
            # Compute log probabilities under current policy
            # Still use SDE logprobs for the current policy (needed for gradient computation)
            bc_log_probs = self.traj_dit.get_logprobs(
                traj_ids, stt_features, cond_ids, teacher_latents, timesteps_traj,
                noise_level=noise_level
            )
            # bc_log_probs are now per-sample after fix: each element shape (B_total,)
            # Stack and mean: first mean over denoising steps, then mean over batch
            bc_loss = -torch.stack(bc_log_probs).mean()
            total_loss = total_loss + self.grpo_bc_coeff * bc_loss
        
        return {
            "loss_all": total_loss,
            "loss_policy": policy_loss,
            "loss_bc": bc_loss,
            "reward": rewards.mean(),
            "predict": None,
            "predict_traj": None,
        }
