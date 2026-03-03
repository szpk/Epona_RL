import os
import sys
from pathlib import Path

# Add project root to path before any navsim imports
root_path = Path(__file__).parent.parent.parent.resolve()  # Use absolute path
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from typing import Any, Dict, List, Union
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import uuid

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map, chunk_list
from nuplan.planning.utils.multithreading.worker_pool import Task

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.epona.epona_agent_jch import EponaAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from utils.config_utils import Config
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache

logger = logging.getLogger(__name__)

# Use absolute path for config
CONFIG_PATH = os.path.join(root_path, "navsim/planning/script/config/pdm_scoring")
CONFIG_NAME = "default_run_pdm_score"


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation with Epona agent.
    
    数据流：
    1. 输入: args -> List[Dict]，每个Dict包含log_file, tokens, cfg
    2. SceneLoader加载数据 -> AgentInput
       - AgentInput.ego_statuses: List[EgoStatus]，每个EgoStatus.ego_pose形状为(3,) [x, y, heading]
       - AgentInput.cameras: List[Cameras]，每个Cameras.cam_f0.image形状为(H, W, 3)，numpy array
    3. agent.compute_trajectory() -> Trajectory
       - Trajectory.poses: numpy array，形状为(num_poses, 3)，其中3表示(x, y, heading)
    4. pdm_score() -> PDMResults (dataclass)
    5. 输出: List[Dict[str, Any]]，每个Dict包含token, valid, 以及PDM评分指标
    
    :param args: input arguments，每个元素包含:
        - "log_file": str，日志文件名
        - "tokens": List[str]，场景token列表
        - "cfg": DictConfig，配置对象
    :return: List[Dict[str, Any]]，每个Dict包含评估结果
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    # 提取log文件名和tokens
    log_names = [a["log_file"] for a in args]  # List[str]
    tokens = [t for a in args for t in a["tokens"]]  # List[str]，扁平化的token列表
    cfg: DictConfig = args[0]["cfg"]

    # 初始化PDM模拟器和评分器
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    
    # Initialize Epona agent
    # Since we're using +agent.xxx parameters, we need to create EponaAgent directly
    # Check if epona-specific parameters exist (config_path is unique to EponaAgent)
    if hasattr(cfg.agent, "config_path"):
        # Read condition_frames from config file if not provided via command line
        # Priority: command line > config file > default (3)
        condition_frames = cfg.agent.get("condition_frames", None)
        if condition_frames is None:
            # Read from config file
            config_args = Config.fromfile(cfg.agent.config_path)
            condition_frames = getattr(config_args, "condition_frames", 3)
        
        # Create EponaAgent with provided parameters
        agent: EponaAgent = EponaAgent(
            config_path=cfg.agent.config_path,
            checkpoint_path=cfg.agent.get("checkpoint_path", None),
            condition_frames=condition_frames,
            use_sde=cfg.agent.get("use_sde", False),
        )
    else:
        # Fallback to instantiate from config (for other agents like constant_velocity_agent)
        agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    # 加载指标缓存和场景数据
    # 注意：data_points中的tokens已经在main函数中过滤过，确保都在metric_cache中存在
    # 这里只需要加载metric_cache_loader来获取缓存文件路径，不需要再次计算交集
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=False  # Epona 需要实际图像数据，不是路径
    )

    # 使用传入的tokens（已经在main函数中过滤过，确保都在metric_cache中存在）
    # 只需要验证并过滤出实际可用的tokens
    tokens_to_evaluate = [token for token in tokens if token in scene_loader.tokens and token in metric_cache_loader.tokens]
    pdm_results: List[Dict[str, Any]] = []
    
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            # 加载指标缓存
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            # 加载AgentInput
            # AgentInput包含:
            #   - ego_statuses: List[EgoStatus]，长度=num_history_frames
            #     - 每个EgoStatus.ego_pose: numpy array，形状(3,) [x, y, heading]
            #   - cameras: List[Cameras]，长度=num_history_frames
            #     - 每个Cameras.cam_f0.image: numpy array，形状(H, W, 3)，RGB图像
            agent_input = scene_loader.get_agent_input_from_token(token)
            
            # 计算轨迹
            # 在EponaAgent内部:
            #   - 图像处理: (H, W, 3) -> (C, H, W) -> stack -> (1, condition_frames, C, H, W)
            #   - 姿态处理: 从agent_input.ego_statuses获取pose信息
            #   - 模型输出: predict_traj形状为(B, H, D)或(num_samples, B, H, D)
            #   - 最终输出: Trajectory.poses形状为(num_poses, 3)，其中num_poses=trajectory_sampling.num_poses
            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input)
            else:
                trajectory = agent.compute_trajectory(agent_input)
            
            # trajectory.poses: numpy array，形状(num_poses, 3)
            # 其中num_poses由trajectory.trajectory_sampling.num_poses决定
            # 3表示(x, y, heading)在ego坐标系中的相对位置

            # PDM评分
            # pdm_score内部会:
            #   1. 将trajectory.poses (num_poses, 3)转换为InterpolatedTrajectory
            #   2. 插值得到pdm_states和pred_states，形状为(num_interpolated_states, state_dim)
            #   3. 模拟得到simulated_states
            #   4. 计算评分得到PDMResults
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,  # Trajectory.poses: (num_poses, 3)
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            # pdm_result: PDMResults dataclass，包含各种评分指标
            # 转换为字典并更新到score_row
            score_row.update(asdict(pdm_result))
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    
    # 返回: List[Dict[str, Any]]，每个Dict包含:
    #   - "token": str
    #   - "valid": bool
    #   - PDM评分指标（从PDMResults转换而来）
    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig, experiment_name: str = "epona_eval", output_dir: str = "./exp/navsim_eval") -> None:
    """
    Main entrypoint for running PDMS evaluation with Epona agent.
    
    数据流概览：
    1. 加载场景数据和指标缓存，计算需要评估的tokens
    2. 按log文件分组，构建data_points
    3. 并行执行run_pdm_score，处理每个data_point
    4. 汇总结果到DataFrame，计算平均值
    5. 保存为CSV文件
    
    :param cfg: omegaconf dictionary，包含所有配置参数
    :param experiment_name: Name of the experiment（未使用，保留用于兼容性）
    :param output_dir: Output directory for results
    """
    # Override output_dir if provided
    if output_dir:
        cfg.output_dir = output_dir

    build_logger(cfg)
    worker = build_worker(cfg)

    # 初始化SceneLoader（用于获取token列表，不加载实际传感器数据）
    # 注意：这里使用build_no_sensors()，因为只需要token列表，不需要实际图像数据
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    # 加载指标缓存
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    # 计算需要评估的tokens（场景数据和指标缓存的交集）
    # tokens_to_evaluate: List[str]，场景token列表
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    
    # Limit number of scenarios if max_scenarios is specified
    original_count = len(tokens_to_evaluate)
    if hasattr(cfg, "max_scenarios") and cfg.max_scenarios is not None:
        max_scenarios = int(cfg.max_scenarios)
        if max_scenarios > 0 and max_scenarios < len(tokens_to_evaluate):
            tokens_to_evaluate = tokens_to_evaluate[:max_scenarios]
            logger.info(f"Limiting evaluation to {max_scenarios} scenarios (out of {original_count} total)")
    
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    
    # Create a set of tokens to evaluate for fast lookup
    tokens_to_evaluate_set = set(tokens_to_evaluate)
    
    # Build data_points, filtering tokens to only include those in tokens_to_evaluate
    # data_points: List[Dict]，每个Dict包含:
    #   - "cfg": DictConfig，配置对象
    #   - "log_file": str，日志文件名
    #   - "tokens": List[str]，该log文件中的token列表（已过滤）
    data_points = []
    for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items():
        # Filter tokens to only include those in tokens_to_evaluate
        filtered_tokens = [token for token in tokens_list if token in tokens_to_evaluate_set]
        if filtered_tokens:  # Only add if there are tokens to process
            data_points.append({
                "cfg": cfg,
                "log_file": log_file,
                "tokens": filtered_tokens,
            })
    # Use GPU-enabled task for agent that requires GPU (e.g., EponaAgent)
    # Check if agent config has GPU requirements (config_path indicates EponaAgent)
    use_gpu = hasattr(cfg.agent, "config_path") if hasattr(cfg, "agent") else False
    if use_gpu:
        # Create task with GPU allocation
        num_gpus_per_task = 1.0  # Allocate 1 GPU per task
        
        # Limit parallelism to available GPUs if num_gpus is specified
        # This ensures we don't create more workers than available GPUs
        if hasattr(cfg, "num_gpus") and cfg.num_gpus is not None:
            num_gpus = int(cfg.num_gpus)
            # Limit worker threads to GPU count to avoid GPU contention
            max_workers = min(worker.number_of_threads, num_gpus) if worker.number_of_threads > 0 else num_gpus
            logger.info(f"Using {num_gpus} GPUs, limiting parallelism to {max_workers} workers")
        else:
            max_workers = worker.number_of_threads

        if max_workers == 0:
            # 串行执行
            # run_pdm_score返回: List[Dict[str, Any]]
            score_rows = run_pdm_score(data_points)
        else:
            # 并行执行
            # object_chunks: List[List[Dict]]，将data_points分成max_workers个chunk
            object_chunks = chunk_list(data_points, max_workers)
            task = Task(fn=run_pdm_score, num_gpus=num_gpus_per_task)
            # scattered_objects: List[List[Dict[str, Any]]]，每个元素是run_pdm_score的返回值
            scattered_objects = worker.map(task, object_chunks)
            # 扁平化: List[Dict[str, Any]]，所有场景的评估结果
            score_rows = [result for results in scattered_objects for result in results]
    else:
        # Use default worker_map for agents that don't need GPU
        # worker_map返回List[Dict[str, Any]]，不是Tuple
        score_rows: List[Dict[str, Any]] = worker_map(worker, run_pdm_score, data_points)

    # score_rows: List[Dict[str, Any]]，每个Dict包含token, valid, 以及PDM评分指标
    # 转换为DataFrame，每行是一个场景的评估结果
    # pdm_score_df形状: (num_scenarios, num_columns)
    # columns包括: token, valid, no_at_fault_collisions, drivable_area_compliance, 
    #              ego_progress, time_to_collision_within_bound, comfort, 
    #              driving_direction_compliance, score
    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    
    # 计算平均值行
    # average_row: Series，包含所有数值列的平均值
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()  # 只有当所有场景都valid时才为True
    # 添加平均值行到DataFrame
    pdm_score_df.loc[len(pdm_score_df)] = average_row
    # 现在pdm_score_df形状: (num_scenarios + 1, num_columns)，最后一行是平均值

    # 保存结果到CSV文件
    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )


if __name__ == "__main__":
    main()

