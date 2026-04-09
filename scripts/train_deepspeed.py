import os
import sys
import math
import time
import torch
import random
import logging
import argparse
from einops import rearrange
import numpy as np
import torch.distributed as dist
import torch.utils
from torch.utils.data import DataLoader, Subset
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from utils.merge_dataset import MixedBatchSampler
import torch.multiprocessing as mp
from dataset.create_dataset import create_dataset
from utils.config_utils import Config
import deepspeed
from utils.deepspeed_utils import get_deepspeed_config

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_path)
sys.path.append(root_path)

from utils.utils import *
from models.model import TrainTransformersDiT
from models.modules.tokenizer import VAETokenizer
from utils.comm import _init_dist_envi
from utils.running import init_lr_schedule, save_ckpt, load_parameters, add_weight_decay, save_ckpt_deepspeed, load_from_deepspeed_ckpt
from torch.nn.parallel import DistributedDataParallel as DDP

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', default=60000000, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help='minibatch size')
    parser.add_argument('--config', default='configs/mar/demo_config.py', type=str)
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('--resume_path', default=None, type=str, help='pretrained path')
    parser.add_argument('--resume_step', default=0, type=int, help='continue to train, step')
    parser.add_argument('--load_stt_path', default=None, type=str, help='pretrained path')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--launcher', type=str, default='pytorch')
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=2000)
    parser.add_argument('--load_from_deepspeed', default=None, type=str, help='pretrained path')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.__dict__)
    return cfg

logger = logging.getLogger('base')

def init_logs(global_rank, args):
    print('#### Initial logs.')
    log_path = os.path.join(args.logdir, args.exp_name)
    save_model_path = os.path.join(args.outdir, args.exp_name)
    tdir_path = os.path.join(args.tdir, args.exp_name)
    validation_path = os.path.join(args.validation_dir, args.exp_name)

    if global_rank == 0:
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        
        if not os.path.exists(validation_path):
            os.makedirs(validation_path)

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        if not os.path.exists(tdir_path):
                os.makedirs(tdir_path)
        setup_logger('base', log_path, 'train', level=logging.INFO, screen=True, to_file=True)
        writer = SummaryWriter(tdir_path + '/train')
        writer_val = SummaryWriter(tdir_path + '/validate')

        args.writer = writer
        args.writer_val = writer_val
    else:
        args.writer = None
        args.writer_val = None
        
    args.log_path = log_path
    args.save_model_path = save_model_path
    args.tdir_path = tdir_path
    args.validation_path = validation_path

def init_environment(args):
    _init_dist_envi(args)
    
    # set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # set backends
    torch.backends.cudnn.benchmark = True

def main(args):
    init_environment(args)
    
    if not args.distributed:
        start_training(0, args)
    else:
        # distributed training
        if args.launcher == 'pytorch':
            print('pytorch launcher.')
            local_rank = int(os.environ["LOCAL_RANK"])
            start_training(local_rank, args)
        elif args.launcher == 'slurm': 
            # this is for debug
            num_gpus_per_nodes = torch.cuda.device_count()
            mp.spawn(start_training, nprocs=num_gpus_per_nodes, args=(args, ))
        else:
            raise RuntimeError(f'{args.launcher} is not supported.')
        
def start_training(local_rank, args):
    torch.cuda.set_device(local_rank)

    if 'RANK' not in os.environ:
        node_rank  = 0  # when debugging, only has a single node
        global_rank = node_rank * torch.cuda.device_count() + local_rank
        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        
    init_logs(int(os.environ["RANK"]), args)

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])

    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")  
    train(local_rank, args)


def train(local_rank, args):
    print(args)
    writer = args.writer
    rank = int(os.environ['RANK'])
    save_model_path = args.save_model_path

    step = args.resume_step

    model = TrainTransformersDiT(args, local_rank=local_rank, condition_frames=args.condition_frames // args.block_size)
    stt_params, dit_params, traj_params = count_parameters(model.model), count_parameters(model.dit), count_parameters(model.traj_dit)
    print(f"Total Parameters: {format_number(stt_params + dit_params)}, SST Parameters: {format_number(stt_params)}, DiT Parameters: {format_number(dit_params)}, TrajDiT Parameters: {format_number(traj_params)}")

    use_grpo = getattr(args, 'grpo', False)
    if not use_grpo:
        model = DDP(model, device_ids=[local_rank, ], output_device=local_rank, find_unused_parameters=True)
    tokenizer = VAETokenizer(args, local_rank)
    eff_batch_size = args.batch_size * args.condition_frames // args.block_size * dist.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)
    lr = args.lr

    # For GRPO (no DDP): model is TrainTransformersDiT directly
    # For non-GRPO (with DDP): model.module is TrainTransformersDiT
    raw_model = model if use_grpo else model.module

    # Load checkpoints BEFORE freezing and optimizer creation
    skip_key = None
    if args.load_stt_path is not None:
        checkpoint = torch.load(args.load_stt_path, map_location="cpu")
        print(f"Load stt: {args.load_stt_path}")
        skip_key="causal_time_space_blocks"
        raw_model = load_parameters(raw_model, checkpoint)
        del checkpoint
    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path, map_location="cpu")
        print(f"Load model: {args.resume_path}")
        raw_model = load_parameters(raw_model, checkpoint, skip_key=skip_key)
        del checkpoint

    # Freeze params BEFORE optimizer creation.
    # CRITICAL for GRPO: add_weight_decay skips requires_grad=False params,
    # so freezing first ensures the optimizer only contains trainable params (TrajDiT).
    # Without this, the optimizer contains ALL 2.53B params but only 59M get gradients,
    # causing DeepSpeed ZeRO-2's flat buffer gradient-parameter mapping to fail silently.
    if args.fix_stt:
        for name, param in raw_model.named_parameters():
            if "causal_time_space_blocks" in name:
                param.requires_grad = False
                print(f"Frozen: {name}")
    if getattr(args, 'fix_dit', False):
        for name, param in raw_model.named_parameters():
            if "dit." in name and "traj_dit" not in name:
                param.requires_grad = False
                print(f"Frozen: {name}")

    # Initialize GRPO AFTER checkpoint loading and freezing
    # This ensures: (1) old_policy has pretrained weights, not random init
    #               (2) old_policy is not tracked by DeepSpeed (avoids OOM/NCCL errors)
    if use_grpo:
        raw_model.init_grpo_after_load(args)

    # Create optimizer AFTER freezing — now add_weight_decay only collects trainable params
    param_groups = add_weight_decay(raw_model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # Log optimizer param count for verification
    total_opt_params = sum(p.numel() for group in param_groups for p in group['params'])
    trainable_count = sum(1 for _ in filter(lambda p: p.requires_grad, raw_model.parameters()))
    print(f"[GRPO_OPT] Optimizer params: {total_opt_params:,} ({total_opt_params/1e6:.1f}M), "
          f"trainable param tensors: {trainable_count}")
    print(optimizer)
    if use_grpo:
        lr_schedule = init_lr_schedule(optimizer, milstones=[10000, 20000, 30000], gamma=0.5)
    else:
        lr_schedule = init_lr_schedule(optimizer, milstones=[100000, 150000, 200000], gamma=0.5)

    train_dataset, train_datalist = create_dataset(args)
    if args.overfit:
        train_dataset = Subset(train_dataset, list(range(4096-100, 4096+100))+list(range(0, 200)))
    
    mix_data_sampler = MixedBatchSampler(
        src_dataset_ls=train_datalist,
        batch_size=args.batch_size, 
        rank=rank, 
        seed=args.seed, 
        num_replicas=int(os.environ["WORLD_SIZE"]), 
        drop_last=True, 
        shuffle=True, 
        prob=args.sample_prob, 
        generator=torch.Generator().manual_seed(0),
    )
    train_data = DataLoader(
        train_dataset,
        num_workers=32,
        batch_sampler=mix_data_sampler
    )
    
    # sampler = DistributedSampler(train_dataset)
    # train_data = DataLoader(
    #     train_dataset, 
    #     batch_size=args.batch_size, 
    #     num_workers=32, 
    #     pin_memory=True, 
    #     drop_last=True, 
    #     sampler=sampler
    # )        
    
    print('Length of train_data', len(train_data))
    epoch = step // len(train_data) + 1
    deepspeed_cfg = get_deepspeed_config(args)
    model, optimizer, _, _ = deepspeed.initialize(
        config_params=deepspeed_cfg,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedule,
    )
    load_from_deepspeed_ckpt(args, model)

    # Register gradient hooks AFTER DeepSpeed init to trace if backward() reaches TrajDiT
    _grpo_grad_hook_log = {}
    if use_grpo and rank == 0:
        for name, param in model.module.named_parameters():
            if "traj_dit" in name and "old_policy" not in name and param.requires_grad:
                def _make_hook(n):
                    def _hook(grad):
                        _grpo_grad_hook_log[n] = grad.float().norm().item()
                        return grad
                    return _hook
                param.register_hook(_make_hook(name))
                print(f"[HOOK_REGISTERED] {name}")

    torch.set_float32_matmul_precision('high')

    print('training...')
    torch.cuda.synchronize()
    time_stamp = time.time()
    while step < args.iter:
        # lr = adjust_learning_rate(optimizer, epoch, args)
        for i, batch_data in enumerate(train_data):
            # Handle both old format (img, rot_matrix) and new format (img, rot_matrix, tokens)
            if len(batch_data) == 2:
                img, rot_matrix = batch_data
                tokens_list = None
            elif len(batch_data) == 3:
                img, rot_matrix, tokens_list = batch_data
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
            
            # print(f'into training {i}: {img.shape}, {rot_matrix.shape}')
            model.train()
            img = img.cuda()
            latents = tokenizer.encode_to_z(img[:, :-args.traj_len+args.forward_iter])
            rot_matrix = rot_matrix.cuda()# .to(torch.bfloat16)
                        
            data_time_interval = time.time() - time_stamp
            torch.cuda.synchronize()
            time_stamp = time.time()
            cf = args.condition_frames // args.block_size
            latents_cond = latents[:, :cf, ...] # B F L(H*W) C!
            rel_pose_cond, rel_yaw_cond = None, None
            fw_iter = 1
            if step % args.multifw_perstep == 0:
                fw_iter = args.forward_iter
            for j in range(fw_iter):
                # optimizer.zero_grad()
                rot_matrix_cond = rot_matrix[:, j*args.block_size:j*args.block_size+args.condition_frames+args.traj_len, ...]
                latents_gt = latents[:, j+cf:j+cf+1, ...]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss_final = model(
                        latents_cond, 
                        rot_matrix_cond,
                        latents_gt,
                        rel_pose_cond=rel_pose_cond,
                        rel_yaw_cond=rel_yaw_cond,
                        step=step,
                        tokens_list=tokens_list
                    )
                loss_value = loss_final["loss_all"]

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)
                model.backward(loss_value)

                # Check if gradient hooks fired during backward
                if use_grpo and step <= 5 and rank == 0:
                    if _grpo_grad_hook_log:
                        hook_norms = [f"{k.split('.')[-2]+'.'+k.split('.')[-1]}={v:.4e}" for k, v in list(_grpo_grad_hook_log.items())[:5]]
                        print(f"[HOOK_FIRED] step={step} n_params={len(_grpo_grad_hook_log)} "
                              f"samples: {', '.join(hook_norms)}")
                    else:
                        print(f"[HOOK_FIRED] step={step} NO HOOKS FIRED — backward did NOT reach TrajDiT!")
                    _grpo_grad_hook_log.clear()

                # Gradient diagnostics for GRPO debugging
                if getattr(args, 'grpo', False) and step % 10 == 0 and rank == 0:
                    traj_dit_grad_norm = 0.0
                    traj_dit_param_norm = 0.0
                    num_grad_params = 0
                    num_zero_grad = 0
                    # Snapshot parameter values BEFORE optimizer step
                    param_snapshot = None
                    for name, param in model.module.named_parameters():
                        if "traj_dit" in name and "old_policy" not in name and param.requires_grad:
                            traj_dit_param_norm += param.data.float().norm().item() ** 2
                            if param_snapshot is None:
                                param_snapshot = (name, param.data[:2, :3].clone().float())
                            if param.grad is not None:
                                traj_dit_grad_norm += param.grad.float().norm().item() ** 2
                                num_grad_params += 1
                            else:
                                num_zero_grad += 1
                    traj_dit_grad_norm = traj_dit_grad_norm ** 0.5
                    traj_dit_param_norm = traj_dit_param_norm ** 0.5
                    print(f"[GRAD] step={step} traj_dit_grad_norm={traj_dit_grad_norm:.6e} "
                          f"param_norm={traj_dit_param_norm:.6e} "
                          f"params_with_grad={num_grad_params} zero_grad={num_zero_grad}")

                model.step()

                # Check if parameters actually changed after optimizer step
                if getattr(args, 'grpo', False) and step % 10 == 0 and rank == 0:
                    if param_snapshot is not None:
                        snap_name, snap_before = param_snapshot
                        for name, param in model.module.named_parameters():
                            if name == snap_name:
                                snap_after = param.data[:2, :3].clone().float()
                                diff = (snap_after - snap_before).abs().max().item()
                                print(f"[PARAM_UPDATE] step={step} param={snap_name} "
                                      f"bf16_diff={diff:.6e} "
                                      f"before={snap_before[0,:3].tolist()} "
                                      f"after={snap_after[0,:3].tolist()}")
                                break

                    # Check DeepSpeed fp32 master weights
                    try:
                        ds_optimizer = model.optimizer
                        # BF16_Optimizer stores fp32 master in bit16_groups_flat / fp32_groups_flat
                        fp32_master = None
                        if hasattr(ds_optimizer, 'fp32_groups_flat'):
                            fp32_master = ds_optimizer.fp32_groups_flat
                        elif hasattr(ds_optimizer, 'single_partition_of_fp32_groups'):
                            fp32_master = ds_optimizer.single_partition_of_fp32_groups

                        if fp32_master is not None:
                            # Print first few values of fp32 master to see if they change
                            for gi, group in enumerate(fp32_master):
                                print(f"[FP32_MASTER] step={step} group={gi} "
                                      f"shape={group.shape} first5={group.data.flatten()[:5].tolist()} "
                                      f"norm={group.data.float().norm().item():.4f}")
                        else:
                            # Try to find any fp32 related attributes
                            fp32_attrs = [a for a in dir(ds_optimizer) if 'fp32' in a.lower() or 'master' in a.lower() or 'bit16' in a.lower()]
                            print(f"[FP32_MASTER] step={step} could not find fp32 master. "
                                  f"Optimizer type: {type(ds_optimizer).__name__}, "
                                  f"fp32-related attrs: {fp32_attrs[:10]}")
                    except Exception as e:
                        print(f"[FP32_MASTER] step={step} error: {e}")
                
                # if args.return_predict and rank == 0 and step % args.eval_steps == 0:
                #     predict_latents = loss_final["predict"].detach()
                #     # validation_step_path = os.path.join(args.validation_path, 'val_'+str(step))
                #     os.makedirs(args.validation_path, exist_ok=True)
                #     gt = ((img[0, 0].permute(0, 2, 3, 1).cpu().numpy() / 2 + 0.5) * 255).astype('uint8')
                #     latents_pred = rearrange(predict_latents, 'b (h w) c -> b h w c', h=args.image_size[0]//(args.downsample_size*args.patch_size), w=args.image_size[1]//(args.downsample_size*args.patch_size))
                #     imgs_pred = tokenizer.z_to_image(latents_pred.float())
                #     pred = (imgs_pred[0].cpu().numpy() * 255).astype('uint8')
                #     imgs = np.concatenate((gt, pred), axis=2)
                #     # imgs = np.concatenate(imgs, axis=0)
                #     cv2.imwrite(os.path.join(args.validation_path, str(step)+'.jpg'), imgs[:,:,::-1])
                
                if j < fw_iter - 1:
                    # GRPO mode does not produce predict/predict_traj, skip autoregressive rollout
                    use_grpo = getattr(args, 'grpo', False)
                    if use_grpo:
                        break
                    if args.return_predict:
                        predict_latents = loss_final["predict"].detach()
                        predict_traj = loss_final["predict_traj"].detach()
                    else:
                        model.eval()
                        predict_traj, predict_latents = model(
                            latents_cond, 
                            rot_matrix_cond,
                            latents_gt,
                            sample_last=False
                        )
                        model.train()
                    latents_cond = rearrange(predict_latents, '(b t) l c -> b t l c', b=args.batch_size, t=args.condition_frames // args.block_size)
                    rel_traj_cond = rearrange(predict_traj, '(b t) l c -> b t l c', b=args.batch_size, t=args.condition_frames // args.block_size)[:, :, 0, :]
                    rel_pose_cond, rel_yaw_cond = rel_traj_cond[..., 0:2], rel_traj_cond[..., 2:3]
            step += 1
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            dist.barrier()
            torch.cuda.synchronize()
            if step % 100 == 1 and rank == 0:
                writer.add_scalar('learning_rate/lr', lr, step)
                writer.add_scalar('loss/loss_all', loss_final["loss_all"].to(torch.float32), step)
                # Check if GRPO training is enabled
                use_grpo = getattr(args, 'grpo', False)
                if use_grpo:
                    # GRPO specific logging
                    if "loss_policy" in loss_final:
                        writer.add_scalar('loss/loss_policy', loss_final["loss_policy"].to(torch.float32), step)
                    if "loss_bc" in loss_final and loss_final["loss_bc"] != 0.0:
                        if isinstance(loss_final["loss_bc"], torch.Tensor):
                            writer.add_scalar('loss/loss_bc', loss_final["loss_bc"].to(torch.float32), step)
                        else:
                            writer.add_scalar('loss/loss_bc', loss_final["loss_bc"], step)
                    if "reward" in loss_final:
                        writer.add_scalar('reward/mean_reward', loss_final["reward"].to(torch.float32), step)
                else:
                    # Standard training logging
                    writer.add_scalar('loss/loss_diff', loss_final["loss_diff"].to(torch.float32), step)
                    writer.add_scalar('loss/loss_yaw_pose', loss_final["loss_yaw_pose"].to(torch.float32), step)
                writer.flush()
            if rank == 0:
                use_grpo = getattr(args, 'grpo', False)
                if use_grpo:
                    # GRPO specific logging
                    policy_loss = loss_final.get("loss_policy", torch.tensor(0.0)).to(torch.float32)
                    bc_loss = loss_final.get("loss_bc", 0.0)
                    if isinstance(bc_loss, torch.Tensor):
                        bc_loss = bc_loss.to(torch.float32)
                    reward = loss_final.get("reward", torch.tensor(0.0)).to(torch.float32)
                    logger.info('step:{} time:{:.2f}+{:.2f} lr:{:.4e} loss_all:{:.4e} policy_loss:{:.4e} bc_loss:{:.4e} reward:{:.4f}'.format(
                        step, data_time_interval, train_time_interval, optimizer.param_groups[0]['lr'],
                        loss_final["loss_all"].to(torch.float32), policy_loss, bc_loss, reward))
                else:
                    # Standard training logging
                    logger.info('step:{} time:{:.2f}+{:.2f} lr:{:.4e} loss_avg:{:.4e} diff_loss:{:.4e} pose_loss:{:.4e} '.format( \
                        step, data_time_interval, train_time_interval, optimizer.param_groups[0]['lr'],  loss_final["loss_all"].to(torch.float32), loss_final["loss_diff"].to(torch.float32), loss_final["loss_yaw_pose"].to(torch.float32)))
            if step % args.eval_steps == 0: # or (step == 1): 
                dist.barrier()
                torch.cuda.synchronize()
                save_ckpt_deepspeed(args, save_model_path, model, optimizer, lr_schedule, step)
                dist.barrier()
                if rank == 0:
                    # For GRPO mode (no DDP), model.module is TrainTransformersDiT directly.
                    # Add 'module.' prefix to match the checkpoint format expected by load_parameters(),
                    # which always strips the first dotted prefix (originally from DDP wrapping).
                    save_model_obj = model.module
                    if use_grpo:
                        prefixed_state_dict = {'module.' + k: v for k, v in save_model_obj.state_dict().items()}
                        ckpt = dict(model_state_dict=prefixed_state_dict)
                        ckpt_path = '{}/tvar_{}.pkl'.format(save_model_path, str(step))
                        torch.save(ckpt, ckpt_path)
                        print(f'#### Save model: {ckpt_path}')
                    else:
                        save_ckpt(args, save_model_path, save_model_obj, optimizer, lr_schedule, step)
                torch.cuda.synchronize()
                dist.barrier()
        epoch += 1
        dist.barrier()
        
if __name__ == "__main__":
    os.chdir(root_path)
    args = add_arguments()
    main(args)