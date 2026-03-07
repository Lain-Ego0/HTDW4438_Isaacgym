from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Htdw4438Cfg(LeggedRobotCfg):
    # ==========================
    # 1. 环境 (Environment)
    # ==========================
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        # 角速度(3) + 重力向量(3) + 指令(3) + 关节位置(12) + 关节速度(12) + 上一帧动作(12) = 45
        num_observations = 45 # 减去线速度感知和地形高度（235-187-3）
        num_privileged_obs = None # 标准 PPO 通常不需要显式的 privileged_obs (除非使用非对称 Actor-Critic)，如果为 None，则 privileged_obs = observations
        
        num_actions = 12
        episode_length_s = 20 

    # ==========================
    # 2. 地形 (Terrain)
    # ==========================
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'       # 训练初期用plane，后期改为 'trimesh' 进行崎岖地形训练
        # measure_heights = True    # 开启高度测量，以便填入观测向量的后 187 维
        measure_heights = False
        



        # 地形参数继承自父类，如需调整可在此覆盖
        # static_friction = 1.0
        # dynamic_friction = 1.0

    # ==========================
    # 3. 初始状态 (Init State)
    # ==========================
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.15] # 初始高度
        default_joint_angles = { 
            'fl_hip_joint': 0, 'fl_thigh_joint': 0.9, 'fl_calf_joint': -1.5, # 前左
            'fr_hip_joint': 0, 'fr_thigh_joint': 0.9, 'fr_calf_joint': -1.5, # 前右
            'rl_hip_joint': 0,  'rl_thigh_joint': 0.9, 'rl_calf_joint': -1.5, # 后左
            'rr_hip_joint': 0, 'rr_thigh_joint': 0.9, 'rr_calf_joint': -1.5, # 后右
        }

    # ==========================
    # 4. 控制 (Control)
    # ==========================
    class control(LeggedRobotCfg.control):
        control_type = 'P'          # 位置控制 (PD)
        stiffness = {'joint': 10.0} 
        damping = {'joint': 0.3}     
        action_scale = 0.25          
        decimation = 2              # 200Hz Sim / 2 = 100Hz Policy

    # ==========================
    # 5. 指令 (Commands)
    # ==========================
    class commands(LeggedRobotCfg.commands):
        curriculum = True 
        max_curriculum = 1.0
        num_commands = 4 
        resampling_time = 10. 
        heading_command = True 
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-4.0, 4.0] # min max [m/s]
            lin_vel_y = [-2.0, 2.0]   # min max [m/s]
            ang_vel_yaw = [-1.56, 1.56]    # min max [rad/s]
            heading = [-3.14, 3.14]

    # ==========================
    # 6. 资产 (Asset)
    # ==========================
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/htdw_4438/urdf/htdw_4438.urdf'
        name = "htdw_4438"
        foot_name = "foot"
        penalize_contacts_on = ["hip", "thigh", "calf", "base"] # 碰撞惩罚：除了脚以外的所有部分碰撞都给予惩罚
        terminate_after_contacts_on = ["base"] # 终止条件：如果机身触地（摔倒），重置回合
        self_collisions = 1 
        flip_visual_attachments = False 
        
        density = 0.001
        angular_damping = 0. # 移除默认阻尼，完全靠 PD 控制
        linear_damping = 0.
        max_angular_velocity = 9.5
        max_linear_velocity = 20.
        armature = 0.005 

    # ==========================
    # 7. 域随机化 (Domain Randomization)
    # ==========================
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25] # 地面摩擦力随机范围

        randomize_base_mass = True
        # base_mass_range = [0.8, 1.2] 
        added_mass_range = [-0.1, 0.3] 
        
        push_robots = True  
        push_interval_s = 15
        max_push_vel_xy = 0.5
        
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1] # 模拟电机力矩输出误差 (±10%)

    # ==========================
    # 8. 奖励函数 (Rewards)
    # ==========================
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.185
        clearance_height_target = -0.135  # 目标脚部离地高度（身体坐标系） 身体高度+clearance_height_target=抬高

    
        class scales(LeggedRobotCfg.rewards.scales):
            termination = -0.0              # 终止条件惩罚
            tracking_lin_vel = 1.0          # 追踪线速度奖励
            tracking_ang_vel = 0.5          # 追踪角速度奖励
            lin_vel_z = -2.0                # 垂直速度惩罚（防止机器人向上跳）
            ang_vel_xy = -0.05              # 水平角速度惩罚（保持姿态稳定）
            orientation = -1.3             # 机身方向惩罚（保持机身水平）
            dof_acc = -2.5e-7               # 关节加速度惩罚（平滑运动）
            joint_power = -2e-5             # 关节功率惩罚（节省能量）

            base_height = -1.0
            # base_height_linear = -1.0        # 线性高度惩罚（保持目标高度）
            default_pos_linear = -0.02        # 默认位置线性惩罚（回归初始姿态）
            diagonal_sync = -0.1             # 对角线腿部同步惩罚（协调性）
            hip_mirror_symmetry = -0.1      # 髋关节镜像对称惩罚

            foot_clearance = -0.0          # 脚部高度惩罚（防止拖脚）
            action_rate = -0.01             # 动作变化率惩罚（平滑控制）
            smoothness = -0.01              # 平滑度惩罚（流畅运动）
            feet_air_time = 0.05             # 脚离地时间奖励（鼓励摆动腿抬起）
            feet_stumble = -0.0             # 脚绊倒惩罚（暂不使用）
            stand_still = -1.0               # 静止状态惩罚
            torques = -0.0                  # 扭矩惩罚（暂不使用）
            dof_vel = -0.0                  # 关节速度惩罚（暂不使用）
            dof_pos_limits = -0.0           # 关节位置限制惩罚（暂不使用）
            dof_vel_limits = -0.0           # 关节速度限制惩罚（暂不使用）
            torque_limits = -0.0            # 扭矩限制惩罚（暂不使用）
            collision = -1.0                # 碰撞惩罚（非脚部分碰撞给予惩罚）           

    # ==========================
    # 9. 归一化 (Normalization)
    # ==========================
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    # ==========================
    # 10. 噪声 (Noise)
    # ==========================
    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # ==========================
    # 11. 物理引擎 (Sim)
    # ==========================
    class sim(LeggedRobotCfg.sim):
        dt = 0.005 
        substeps = 1 
        gravity = [0., 0., -9.81] 
        up_axis = 1  # 0 is y, 1 is z
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1 
            num_position_iterations = 4 
            num_velocity_iterations = 1 
            contact_offset = 0.01 
            rest_offset = 0.0 
            bounce_threshold_velocity = 0.5 
            max_depenetration_velocity = 1.0 
            default_buffer_size_multiplier = 5
            max_gpu_contact_pairs = 2**23 

# ==========================
# 训练参数 (Standard PPO Runner)
# ==========================
class Htdw4438CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01 # 恢复标准值，原 0.005 可能偏小
        learning_rate = 1e-3 # 恢复标准 LeggedGym 学习率
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'htdw_4438_standard'
        max_iterations = 300000 
        save_interval = 200 
        policy_class_name = 'ActorCritic' 
        algorithm_class_name = 'PPO'    
        num_steps_per_env = 24 
