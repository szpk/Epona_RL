from dataset.dataset import TrainDataset, TrainImgDataset
from torch.utils.data import ConcatDataset
from dataset.dataset_nuplan import NuPlan

def create_dataset(args, split='train'):
    data_list = args.train_data_list
    dataset_list = []
    
    # Check if GRPO is enabled
    use_grpo = getattr(args, 'grpo', False)
    openscene_root = args.datasets_paths.get('openscene_root', None)
    
    for data_name in data_list:
        if data_name == 'navsim':
            # Use OpenScene/NavSim data (can be used with or without GRPO)
            if openscene_root is None:
                raise ValueError("openscene_root must be set in datasets_paths for navsim dataset")
            from dataset.dataset_openscene import NuPlanOpenScene
            
            # GRPO 训练时使用 navtrain.yaml 过滤 + 固定 SceneFilter (num_future_frames=10) 以匹配 metric_cache
            grpo_metric_cache_path = getattr(args, 'grpo_metric_cache_path', None) if use_grpo else None
            dataset = NuPlanOpenScene(
                openscene_root=openscene_root,
                split=split,
                condition_frames=args.condition_frames+(args.forward_iter+args.traj_len-1)*args.block_size,
                block_size=args.block_size,
                downsample_fps=args.downsample_fps,
                h=args.image_size[0],
                w=args.image_size[1],
                no_pose=args.no_pose,
                use_grpo=use_grpo,  # GRPO 训练时启用 navtrain 过滤 + 固定 SceneFilter
                grpo_metric_cache_path=grpo_metric_cache_path,  # 用于过滤 metric_cache 中不存在的 scenes
            )
            print(f"NavSim OpenScene data length: {len(dataset)}")
        elif data_name == 'nuplan':
            # For GRPO training, use OpenScene data (same as recogdrive)
            # For non-GRPO training, use Epona data (original format)
            if use_grpo and openscene_root is not None:
                # Import OpenScene dataset
                from dataset.dataset_openscene import NuPlanOpenScene
                
                grpo_metric_cache_path = getattr(args, 'grpo_metric_cache_path', None) if use_grpo else None
                dataset = NuPlanOpenScene(
                    openscene_root=openscene_root,
                    split=split,
                    condition_frames=args.condition_frames+(args.forward_iter+args.traj_len-1)*args.block_size,
                    block_size=args.block_size,
                    downsample_fps=args.downsample_fps,
                    h=args.image_size[0],
                    w=args.image_size[1],
                    no_pose=args.no_pose,
                    use_grpo=use_grpo,  # GRPO 训练时启用 navtrain 过滤 + 固定 SceneFilter
                    grpo_metric_cache_path=grpo_metric_cache_path,  # 用于过滤 metric_cache 中不存在的 scenes
                )
                print(f"Nuplan OpenScene data length (GRPO): {len(dataset)}")
            else:
                # Use original Epona dataset (non-GRPO)
                dataset = NuPlan(
                    args.datasets_paths['nuplan_root'], 
                    args.datasets_paths['nuplan_json_root'], 
                    split=split, 
                    condition_frames=args.condition_frames+(args.forward_iter+args.traj_len-1)*args.block_size, 
                    block_size=args.block_size,
                    downsample_fps=args.downsample_fps,
                    h=args.image_size[0],
                    w=args.image_size[1],
                    no_pose=args.no_pose
                )
                print(f"Nuplan Epona data length (non-GRPO): {len(dataset)}")
        elif data_name == 'nuscense':
            dataset = TrainDataset(
                args.datasets_paths['nuscense_root'],
                args.datasets_paths['nuscense_train_json_path'], 
                condition_frames=args.condition_frames+(args.forward_iter+args.traj_len-1)*args.block_size, 
                downsample_fps=args.downsample_fps,
                h=args.image_size[0],
                w=args.image_size[1])
        elif data_name == 'nuscense_img':
            dataset = TrainImgDataset(
                args.datasets_paths['nuscense_root'], # args.train_nuscenes_path, 
                args.datasets_paths['nuscense_train_json_path'], 
                condition_frames=args.condition_frames, 
                downsample_fps=args.downsample_fps,
                reverse_seq=args.reverse_seq)
        dataset_list.append(dataset)
    
    data_array = ConcatDataset(dataset_list)
    return data_array, dataset_list