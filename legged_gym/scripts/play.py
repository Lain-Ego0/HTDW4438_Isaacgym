from legged_gym import LEGGED_GYM_ROOT_DIR
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

# --- 自定义导出函数：直接保存到你指定的目录 ---
def local_export_policy_as_onnx(actor_critic, path):
    """
    将策略导出为 ONNX 格式
    """
    # 获取 actor 模型
    if hasattr(actor_critic, 'actor'):
        model = actor_critic.actor
    else:
        model = actor_critic

    # 创建虚拟输入用于追踪模型结构
    # 输入维度适配 observation 
    dummy_input = torch.zeros(1, model[0].in_features, device=model[0].weight.device)
    
    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(path), exist_ok=True)

    print(f"正在导出 ONNX 到: {path}")
    
    # 导出
    torch.onnx.export(
        model, 
        dummy_input, 
        path, 
        opset_version=11,
        verbose=False,
        input_names=['obs'],
        output_names=['actions']
    )
    print("导出成功！")

def _safe_filename_part(value: str) -> str:
    """Make a filesystem-friendly filename part (no slashes/spaces)."""
    value = str(value).strip()
    for ch in ("/", "\\", " ", "\t", "\n", ":", ";"):
        value = value.replace(ch, "_")
    return value

def build_onnx_filename(project_name: str, checkpoint) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_title = f"model_{checkpoint}" if checkpoint is not None else "model_auto"
    return f"{_safe_filename_part(project_name)}_{timestamp}_{_safe_filename_part(ckpt_title)}.onnx"

def play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # --- 环境设置 ---
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.commands.heading_command = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 设置速度
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel

    obs = env.get_observations()
    
    # --- 加载模型 ---
    train_cfg.runner.resume = True
    load_run = args.load_run
    checkpoint = args.checkpoint
    
    # 手动构建模型路径
    if load_run and checkpoint:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, load_run, f"model_{checkpoint}.pt")
        print(f"--> 正在加载模型路径: {path}")
        # 显式传入 train_path
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, train_path=path)
    else:
        # 如果没传参数，让它尝试自动寻找
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # --- 导出到指定的目录 ---
    if EXPORT_POLICY:
        # 导出到项目根目录的 onnx/，文件名: 项目名 + 时间 + checkpoint标题
        target_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "onnx")
        project_name = getattr(train_cfg.runner, "experiment_name", None) or args.task
        onnx_filename = build_onnx_filename(project_name, checkpoint)
        onnx_full_path = os.path.join(target_dir, onnx_filename)
        
        # 调用本地函数导出
        local_export_policy_as_onnx(ppo_runner.alg.actor_critic, onnx_full_path)

    logger = Logger(env.dt)
    robot_index = 0 
    joint_index = 1 
    stop_state_log = 100 
    stop_rew_log = env.max_episode_length + 1 
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        
        # 强制覆盖指令
        env.commands[:, 0] = x_vel
        env.commands[:, 1] = y_vel
        env.commands[:, 2] = yaw_vel
        
        obs, _, rews, dones, infos = env.step(actions.detach())

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            current_target = actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item()
            logger.log_states(
                {
                    'dof_pos_target': current_target,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True # 
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, x_vel=0.6, y_vel=0.6, yaw_vel=0.0)
