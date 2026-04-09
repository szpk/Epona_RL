import torch
from torch import Tensor, nn
from dataclasses import dataclass
import math

from models.modules.dit_modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))    

@dataclass
class TrajParams:
    in_channels: int
    out_channels: int
    # vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class TrajDiT(nn.Module):
    def __init__(self, params: TrajParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.traj_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        # self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.cond_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        traj: Tensor,
        traj_ids: Tensor,
        cond: Tensor,
        cond_ids: Tensor,
        timesteps: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if traj.ndim != 3 or cond.ndim != 3:
            raise ValueError("Input traj and cond tensors must have 3 dimensions.")

        # running on sequences traj
        traj = self.traj_in(traj)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        # vec = vec + self.vector_in(y)
        cond = self.cond_in(cond)

        ids = torch.cat((cond_ids, traj_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            traj, cond = block(traj, cond=cond, vec=vec, pe=pe)

        traj = torch.cat((cond, traj), 1)
        for block in self.single_blocks:
            traj = block(traj, vec=vec, pe=pe)
        traj = traj[:, cond.shape[1] :, ...]

        traj = self.final_layer(traj, vec)  # (N, T, patch_size ** 2 * out_channels)
        return traj
    
    def training_losses(self, 
                        traj: Tensor,     # (B, L, C)
                        traj_ids: Tensor,
                        cond: Tensor,
                        cond_ids: Tensor,
                        t: Tensor,
                        guidance: Tensor | None = None,
                        noise: Tensor | None = None,
                        return_predict=False
                    ) -> Tensor:
        if noise is None:
            noise = torch.randn_like(traj)
        terms = {}
        
        x_t = t * traj + (1. - t) * noise
        target = traj - noise
        pred = self(traj=x_t, traj_ids=traj_ids, cond=cond, cond_ids=cond_ids, timesteps=t.reshape(-1), guidance=guidance)
        assert pred.shape == target.shape == traj.shape
        predict = x_t + pred * (1. - t)
        terms["mse"] = mean_flat((target - pred) ** 2)
        
        terms["loss"] = terms["mse"].mean()
        if return_predict:
            terms["predict"] = predict
        else:
            terms["predict"] = None
        return terms
        
    def sample(self,
                traj: Tensor,
                traj_ids: Tensor,
                cond: Tensor,
                cond_ids: Tensor,
                timesteps: list[float],
                deterministic: bool = False,
                num_samples: int = 8,
                noise_level: float = 0.7,
            ):
        """
        Sample trajectories using SDE flow matching.
        
        Args:
            traj: Initial trajectory (noise) of shape (B, H, D)
            traj_ids: Trajectory position IDs
            cond: Conditioning features
            cond_ids: Conditioning position IDs
            timesteps: List of timesteps for sampling
            deterministic: If True, use ODE (deterministic), else use SDE (stochastic)
            num_samples: Number of trajectories to sample (default: 8)
            noise_level: Noise level for SDE sampling (default: 0.7)
        
        Returns:
            If num_samples > 1: Tensor of shape (num_samples, B, H, D) containing all sampled trajectories
            If num_samples == 1: Tensor of shape (B, H, D) containing single trajectory
        """
        all_trajectories = []
        
        # Sample multiple trajectories
        for sample_idx in range(num_samples):
            traj_sample = traj.clone()  # Start from the same initial noise
            
            # Set sigma_max (use second timestep as default, following flow_grpo)
            sigma_max = timesteps[1] if len(timesteps) > 1 else timesteps[0]
            
            for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
                t_vec = torch.full((traj_sample.shape[0],), t_curr, dtype=traj_sample.dtype, device=traj_sample.device)
                d_timestep = t_prev - t_curr  # Negative value (going from t=1 to t=0)
                
                # Get model prediction (velocity field)
                pred = self(
                    traj=traj_sample,
                    traj_ids=traj_ids,
                    cond=cond,
                    cond_ids=cond_ids,
                    timesteps=t_vec,
                )
                
                # # Debug: check if pred is all zeros or NaN
                # if sample_idx == 0 and t_curr == timesteps[0]:  # Only print for first sample, first timestep
                #     print(f"[DEBUG] t_curr={t_curr:.4f}, pred stats: min={pred.min().item():.4f}, max={pred.max().item():.4f}, mean={pred.mean().item():.4f}, has_nan={torch.isnan(pred).any().item()}")
                #     print(f"[DEBUG] traj_sample stats: min={traj_sample.min().item():.4f}, max={traj_sample.max().item():.4f}, mean={traj_sample.mean().item():.4f}")
                
                if deterministic:
                    # ODE: deterministic update
                    traj_sample = traj_sample + d_timestep * pred
                else:
                    # SDE: stochastic update (following flow_grpo's _sde_step_with_logprob)
                    # Convert to float32 for numerical stability
                    traj_sample_fp32 = traj_sample.float()
                    pred_fp32 = pred.float()
                    t_curr_fp32 = torch.tensor(t_curr, dtype=torch.float32, device=traj_sample.device)
                    d_timestep_fp32 = torch.tensor(d_timestep, dtype=torch.float32, device=traj_sample.device)
                    sigma_max_fp32 = torch.tensor(sigma_max, dtype=torch.float32, device=traj_sample.device)
                    
                    # Compute std_dev_t following flow_grpo's SDE formula
                    # std_dev_t = sqrt(timestep / (1 - timestep)) * noise_level
                    # Handle timestep == 1 case with sigma_max, and timestep == 0 case to avoid division by zero
                    t_curr_safe = torch.where(t_curr_fp32 == 0.0, torch.tensor(1e-8, dtype=torch.float32, device=t_curr_fp32.device), t_curr_fp32)
                    std_dev_t = torch.sqrt(t_curr_safe / (1 - torch.where(t_curr_safe == 1.0, sigma_max_fp32, t_curr_safe))) * noise_level
                    
                    # Compute mean using flow_grpo's SDE formula (exactly as in flow_grpo)
                    # prev_sample_mean = sample*(1+std_dev_t**2/(2*timestep)*d_timestep) + 
                    #                    model_output*(1+std_dev_t**2*(1-timestep)/(2*timestep))*d_timestep
                    # std_dev_t is a scalar, PyTorch will broadcast automatically
                    # Use t_curr_safe to avoid division by zero when t_curr is 0
                    term1 = traj_sample_fp32 * (1 + std_dev_t**2 / (2 * t_curr_safe) * d_timestep_fp32)
                    term2 = pred_fp32 * (1 + std_dev_t**2 * (1 - t_curr_safe) / (2 * t_curr_safe)) * d_timestep_fp32
                    prev_sample_mean = term1 + term2
                    
                    # Sample noise for SDE step
                    variance_noise = torch.randn_like(traj_sample_fp32)
                    
                    # Compute prev_sample using SDE formula (exactly as in flow_grpo)
                    # prev_sample = prev_sample_mean + std_dev_t * sqrt(-d_timestep) * variance_noise
                    # Note: d_timestep is negative, so -d_timestep is positive
                    prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-d_timestep_fp32) * variance_noise
                    
                    # Convert back to original dtype
                    traj_sample = prev_sample.to(dtype=traj_sample.dtype)
                    
                    # Debug: check final traj_sample
                    if sample_idx == 0 and t_prev == timesteps[-1]:  # Only print for first sample, last timestep
                        print(f"[DEBUG] Final traj_sample stats: min={traj_sample.min().item():.4f}, max={traj_sample.max().item():.4f}, mean={traj_sample.mean().item():.4f}")
            
            all_trajectories.append(traj_sample)
        
        # Stack all trajectories: (num_samples, B, H, D)
        if num_samples > 1:
            return torch.stack(all_trajectories, dim=0)
        else:
            return all_trajectories[0]
    
    def _sde_step_with_logprob(
        self,
        model_output: Tensor,
        timestep: Tensor,
        d_timestep: Tensor,
        sample: Tensor,
        prev_sample: Tensor | None = None,
        sigma_max: Tensor | None = None,
        noise_level: float = 0.7,
        randn_clip_value: float = 5.0,
        log_prob_clamp: tuple | None = None,
    ):
        """
        Single SDE denoising step with log probability computation.
        Directly ported from flow_grpo's _sde_step_with_logprob.
        
        Args:
            model_output: Predicted velocity field (B, H, D), float32.
            timestep: Current timestep scalar or (B,) tensor, float32.
            d_timestep: Time difference (negative), float32.
            sample: Current noisy sample (B, H, D), float32.
            prev_sample: If provided, use this as the next state instead of sampling.
            sigma_max: Maximum sigma for timestep==1 handling, float32 tensor.
            noise_level: Noise level for SDE.
        
        Returns:
            prev_sample: Next state (B, H, D).
            log_prob: Log probability, shape (B,).
            prev_sample_mean: Predicted mean (B, H, D).
            std_dev_t: Standard deviation at this step.
        """
        # Exactly following flow_grpo's _sde_step_with_logprob (sde_type="sde")
        # bf16 can overflow here, all inputs must already be float32
        std_dev_t = torch.sqrt(
            timestep / (1 - torch.where(timestep == 1, sigma_max, timestep))
        ) * noise_level
        # Apply minimum std floor for log_prob computation (matching ReCogDrive's min_logprob_denoising_std)
        # Without this, std_dev_t at small timesteps → ~0.01, causing log_probs to explode to -1000+,
        # which get clamped to -5 killing gradients at the most important denoising steps
        min_logprob_std = getattr(self, 'min_logprob_denoising_std', 0.1)
        std_dev_t = std_dev_t.clamp(min=min_logprob_std)

        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * timestep) * d_timestep) +
            model_output * (1 + std_dev_t**2 * (1 - timestep) / (2 * timestep)) * d_timestep
        )

        if prev_sample is None:
            variance_noise = torch.randn_like(sample)
            # Clamp noise to prevent extreme outliers (following recogdrive randn_clip_value=5.0)
            if randn_clip_value is not None and randn_clip_value > 0:
                variance_noise = variance_noise.clamp(-randn_clip_value, randn_clip_value)
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * d_timestep) * variance_noise

        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2) / (
            2 * ((std_dev_t * torch.sqrt(-1 * d_timestep)) ** 2)
        )

        # Per-element clamp before mean reduction (following recogdrive's clamp(-5, 2))
        # This prevents extreme log_prob values from small σ steps dominating bc_loss
        if log_prob_clamp is not None:
            log_prob = log_prob.clamp(min=log_prob_clamp[0], max=log_prob_clamp[1])

        # Mean along trajectory dimensions only, KEEP batch dimension!
        # log_prob shape: (B, H, D) -> (B,) by averaging over H and D
        # This preserves per-sample log probabilities needed for GRPO's
        # per-sample ratio * advantage computation.
        if log_prob.dim() >= 2:
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.dim())))  # (B,)
        # else: scalar, keep as is (shouldn't happen in practice)

        return prev_sample, log_prob, prev_sample_mean, std_dev_t

    def sample_chain(
        self,
        traj: Tensor,
        traj_ids: Tensor,
        cond: Tensor,
        cond_ids: Tensor,
        timesteps: list[float],
        deterministic: bool = False,
        noise_level: float = 0.7,
        randn_clip_value: float = 5.0,
        clip_x: float | None = 1.5,
        clip_y: float | None = 1.5,
        clip_yaw: float | None = 1.5,
    ):
        """
        Generates the full denoising chain using SDE sampling (following flow_grpo).
        Computes log probabilities on-the-fly during sampling.

        The SDE formula is ported from flow_grpo which uses t=1 (noise) → t=0 (data)
        convention (descending). However, Epona's model uses get_schedule which returns
        ascending timesteps after applying `1-t`, so t=0 (noise) → t=1 (data).

        To bridge the two conventions:
        1. We flip the schedule to descending for the SDE formula (flow_grpo convention).
        2. When calling the model, we remap t_sde → t_model = 1 - t_sde.
        3. We negate the model output: v_flow = -v_epona (opposite velocity directions).

        Args:
            traj (Tensor): Initial trajectory (noise) of shape (B, H, D), where D=3 (x, y, yaw).
            traj_ids (Tensor): Trajectory position IDs.
            cond (Tensor): Conditioning features.
            cond_ids (Tensor): Conditioning position IDs.
            timesteps (list[float]): List of timesteps from get_schedule (ascending, Epona convention).
            deterministic (bool): If True, use ODE (no noise). If False, use SDE.
            noise_level (float): Noise level for SDE sampling.
            randn_clip_value (float): Clamp noise to [-val, val] during SDE steps (like recogdrive).
            clip_x (float | None): Clamp x dimension to [-val, val] in normalized space. Default 1.5.
            clip_y (float | None): Clamp y dimension to [-val, val] in normalized space. Default 1.5.
            clip_yaw (float | None): Clamp yaw dimension to [-val, val] in normalized space. Default 1.5.

        Returns:
            Tuple of (all_latents, all_log_probs, final_traj, all_timesteps).
        """
        # Build timestep arrays for the SDE formula (flow_grpo convention: descending).
        # Epona's get_schedule returns ascending [0.0, t1, ..., 1.0] (t=0 noise, t=1 data).
        # flow_grpo's SDE formula expects descending [1.0, ..., t1, 0.0] (t=1 noise, t=0 data).
        # The mapping is: t_sde = 1 - t_epona, so flipping gives the descending SDE schedule.
        ts = torch.tensor(timesteps, device=traj.device, dtype=torch.float32)
        ts_desc = ts.flip(0)  # [1.0, ..., t1, 0.0]  descending (flow_grpo convention)
        dts = ts_desc[1:] - ts_desc[:-1]  # negative values
        ts_desc = ts_desc[:-1]  # remove last (0.0), so ts_desc never contains 0
        # Now ts_desc = [1.0, ..., t2, t1] (all > 0), len = K
        # dts has length K

        sigma_max = ts_desc[1] if len(ts_desc) > 1 else ts_desc[0]
        sigma_max_t = torch.tensor(sigma_max.item(), device=traj.device, dtype=torch.float32)

        x_t = traj.clone()
        all_latents = [x_t.clone()]
        all_log_probs = []
        all_timesteps_list = []

        for i in range(len(ts_desc)):
            t_curr = ts_desc[i]  # SDE convention: descending from 1.0
            dt = dts[i]

            # Remap to Epona's model convention: t_model = 1 - t_sde
            t_model = 1.0 - t_curr.item()
            t_vec = torch.full((x_t.shape[0],), t_model, dtype=x_t.dtype, device=x_t.device)

            # Get model prediction (velocity field in Epona's convention: noise→data)
            pred = self(
                traj=x_t,
                traj_ids=traj_ids,
                cond=cond,
                cond_ids=cond_ids,
                timesteps=t_vec,
            )

            # Negate velocity: flow_grpo expects v pointing data→noise,
            # but Epona outputs v pointing noise→data
            pred = -pred

            if deterministic:
                # ODE step: no noise, no logprob
                x_t = x_t + dt * pred
                all_log_probs.append(torch.tensor(0.0, device=x_t.device))
            else:
                # SDE step with logprob (following flow_grpo exactly)
                # Convert to float32 for numerical stability
                x_t_fp32 = x_t.float()
                pred_fp32 = pred.float()
                t_scalar = t_curr.float()
                dt_scalar = dt.float()

                x_t_fp32, log_prob, _, _ = self._sde_step_with_logprob(
                    model_output=pred_fp32,
                    timestep=t_scalar,
                    d_timestep=dt_scalar,
                    sample=x_t_fp32,
                    prev_sample=None,  # Sample new noise
                    sigma_max=sigma_max_t,
                    noise_level=noise_level,
                    randn_clip_value=randn_clip_value,
                )
                x_t = x_t_fp32.to(dtype=traj.dtype)
                all_log_probs.append(log_prob)

            all_latents.append(x_t.clone())
            all_timesteps_list.append(t_curr.item())

        # Clamp final trajectory dimensions separately before denormalization
        # Different dimensions have different physical meanings and require different clip values
        if clip_x is not None and clip_x > 0:
            x_t[..., 0:1] = x_t[..., 0:1].clamp(-clip_x, clip_x)
        if clip_y is not None and clip_y > 0:
            x_t[..., 1:2] = x_t[..., 1:2].clamp(-clip_y, clip_y)
        if clip_yaw is not None and clip_yaw > 0:
            x_t[..., 2:3] = x_t[..., 2:3].clamp(-clip_yaw, clip_yaw)

        # Update the last element in all_latents to match clipped values
        all_latents[-1] = x_t.clone()

        all_timesteps_tensor = torch.tensor(all_timesteps_list, device=traj.device, dtype=torch.float32)
        return all_latents, all_log_probs, x_t, all_timesteps_tensor

    def get_logprobs(
        self,
        traj_ids: Tensor,
        cond: Tensor,
        cond_ids: Tensor,
        all_latents: list,
        timesteps: list[float],
        noise_level: float = 0.7,
        deterministic: bool = False,
        log_prob_clamp: tuple | None = None,
    ) -> list:
        """
        Re-computes log probabilities for a previously sampled chain.
        Uses the stored latents from sample_chain and re-runs the model
        to get updated log probabilities under the current policy.

        Following flow_grpo's generate_image_learn: for each step, we run the
        model on the stored x_t, compute the SDE mean, and evaluate the logprob
        of the stored x_{t+1} under that distribution.

        Uses the same timestep convention remapping as sample_chain:
        - SDE formula operates in flow_grpo convention (descending, t=1 noise)
        - Model is called with remapped timestep: t_model = 1 - t_sde
        - Model output is negated: v_flow = -v_epona

        Args:
            traj_ids (Tensor): Trajectory position IDs (B, H, 3).
            cond (Tensor): Conditioning features (B, L, C).
            cond_ids (Tensor): Conditioning position IDs (B, L, 3).
            all_latents (list): List of latent states from sample_chain (length K+1).
            timesteps (list[float]): Original timesteps from get_schedule (ascending).
            noise_level (float): Noise level for SDE.
            deterministic (bool): Not used, kept for API compatibility.

        Returns:
            list: List of log probability tensors, one per denoising step.
        """
        # Reconstruct timestep arrays (same as in sample_chain, flow_grpo convention)
        ts = torch.tensor(timesteps, device=cond.device, dtype=torch.float32)
        ts_desc = ts.flip(0)
        dts = ts_desc[1:] - ts_desc[:-1]
        ts_desc = ts_desc[:-1]

        sigma_max = ts_desc[1] if len(ts_desc) > 1 else ts_desc[0]
        sigma_max_t = torch.tensor(sigma_max.item(), device=cond.device, dtype=torch.float32)

        num_steps = len(ts_desc)
        log_probs = []

        for i in range(num_steps):
            t_curr = ts_desc[i]
            dt = dts[i]
            x_t = all_latents[i]      # current state
            x_next = all_latents[i+1]  # next state (the one we want logprob for)

            # Remap to Epona's model convention: t_model = 1 - t_sde
            t_model = 1.0 - t_curr.item()
            t_vec = torch.full((x_t.shape[0],), t_model, dtype=x_t.dtype, device=x_t.device)

            # Get model prediction under current policy
            pred = self(
                traj=x_t,
                traj_ids=traj_ids,
                cond=cond,
                cond_ids=cond_ids,
                timesteps=t_vec,
            )

            # Negate velocity: flow_grpo convention vs Epona convention
            pred = -pred

            # Compute logprob of x_next under current policy's SDE distribution
            x_t_fp32 = x_t.float()
            pred_fp32 = pred.float()
            x_next_fp32 = x_next.float()
            t_scalar = t_curr.float()
            dt_scalar = dt.float()

            _, log_prob, _, _ = self._sde_step_with_logprob(
                model_output=pred_fp32,
                timestep=t_scalar,
                d_timestep=dt_scalar,
                sample=x_t_fp32,
                prev_sample=x_next_fp32,  # Evaluate logprob at this point
                sigma_max=sigma_max_t,
                noise_level=noise_level,
                log_prob_clamp=log_prob_clamp,
            )
            log_probs.append(log_prob)

        return log_probs