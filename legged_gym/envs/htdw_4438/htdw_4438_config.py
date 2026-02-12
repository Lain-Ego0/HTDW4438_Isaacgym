from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

#机器人参数：电机减速比30 最高效率点速度95rpm，大腿长0.12m，小腿长0.12m 正常站立高度0.13-0.14m

#注意注意：
# 单步原始观测：45 维（角速度 3 + 重力 3 + 指令 3 + 关节位 12 + 关节速 12 + 上一动作 12）
# 历史观测：6 帧 ×45 维 = 270 维（输入估计器）
# Estimator (将 270 维历史观测压缩为 19 维潜变量)
# Actor (接收 45 维当前观测 + 19 维潜变量)

class Htdw4438Cfg(LeggedRobotCfg):
    # ==========================
    # 1. 环境与地形 (适配 HimLoco)
    # ==========================
    class env(LeggedRobotCfg.env):
        num_envs = 4096

        # 角速度(3) + 重力向量(3) + 指令(3) + 关节位置(12) + 关节速度(12) + 上一帧动作(12) = 45
        num_one_step_observations = 45
        
        # [历史信息] 输入过去 6 帧的观测，用于让网络推断环境参数和自身状态 也就是45*6=270
        num_observations = num_one_step_observations * 6 
        
        # 特权观测 (用于 Teacher Policy)
        # 包含：单步观测 + 线速度(3) + 外力(3) + 地形点(187) 
        num_one_step_privileged_obs = 45 + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs * 1
        
        episode_length_s = 20 # 每个回合 20 秒，超时重置

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'      
        measure_heights = True   

    # ==========================
    # 2. 初始状态
    # ==========================
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.15] 
        default_joint_angles = { 
            'fl_hip_joint': 0.1, 'fl_thigh_joint': 0.8, 'fl_calf_joint': -1.5, # 前左
            'fr_hip_joint': -0.1, 'fr_thigh_joint': 0.8, 'fr_calf_joint': -1.5, # 前右
            'rl_hip_joint': 0.1,  'rl_thigh_joint': 0.8, 'rl_calf_joint': -1.5, # 后左
            'rr_hip_joint': -0.1, 'rr_thigh_joint': 0.8, 'rr_calf_joint': -1.5, # 后右
        }

    # ==========================
    # 3. 控制参数 (适配 HTDW-4438)
    # ==========================
    class control(LeggedRobotCfg.control):
        control_type = 'P'          # 位置控制 (PD Controller)
        stiffness = {'joint': 30.0}   # PD参数保留
        damping = {'joint': 0.7}     
        action_scale = 0.05          # 动作缩放：网络输出通常在 [-1, 1]，乘上 0.05 后变为目标关节弧度
        decimation = 2              # 控制频率设置 物理引擎 dt = 0.005 (200Hz) -> 控制频率 = 200 / 2 = 100Hz

    # ==========================
    # 4. 指令范围 (低速限制)   
    # ==========================
    class commands(LeggedRobotCfg.commands):
        curriculum = True # 课程学习：从简单指令开始，慢慢增加难度
        max_curriculum = 1.0
        num_commands = 4  # x vel, y vel, yaw vel, heading
        resampling_time = 10. # 每 10 秒重新采样一次指令
        heading_command = True # 是否使用朝向指令
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.6, 0.6] 
            lin_vel_y = [-0.4, 0.4] 
            ang_vel_yaw = [-0.2, 0.2] 
            heading = [-3.14, 3.14]

    # ==========================
    # 5. 资产配置
    # ==========================
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/htdw_4438/urdf/htdw_4438.urdf'
        name = "htdw_4438"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"] # 碰撞惩罚：如果大腿、小腿或机身触地，给予惩罚
        terminate_after_contacts_on = ["base"] # 终止条件：如果机身触地（摔倒），重置回合
        self_collisions = 1 # 自碰撞
        flip_visual_attachments = False 
        
        density = 0.001
        angular_damping = 0. # 移除默认阻尼，完全靠 PD 控制
        linear_damping = 0.
        max_angular_velocity = 9.5
        max_linear_velocity = 20.
        armature = 0.005 # 电机转子惯量 (Armature)

    # ==========================
    # 6. 域随机化 
    # ==========================
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 2.75] # 地面摩擦力随机范围

        randomize_base_mass = True
        base_mass_range = [0.8, 1.2] # 机身
        added_mass_range = [-0.1, 0.3] # 负载范围
        
        push_robots = False # 随机推力，训练抗干扰
        push_interval_s = 15
        max_push_vel_xy = 0.5
        
        # randomize_motor_offset = True
        # motor_offset_range = [-0.05, 0.05] # 模拟电机零点偏移误差 (±5度)
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2] # 模拟电机力矩输出误差 (±10%)
        
        # 关节 PD 参数随机化
        randomize_kp = True
        kp_range = [0.8, 1.2]
        randomize_kd = True
        kd_range = [0.8, 1.2]
        
        # 外力干扰 (Disturbance)
        disturbance = True
        disturbance_range = [-2.0, 2.0] 
        disturbance_interval = 8
        
        # 延迟随机化 (模拟通信/计算延迟)
        delay = True
        delay_range = [0.0, 0.06]

    # ==========================
    # 7. 奖励函数
    # ==========================
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.13
    
        class scales(LeggedRobotCfg.rewards.scales):

            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_acc = -2.5e-7
            joint_power = -2e-5

            # base_height = -1.0
            base_height_linear = -0.9 # 新的高度惩罚（已修改）
            # default_pos_linear = -0.1 # 新的线性默认位置惩罚
            # diagonal_sync = -0.5      # 对角线非同步惩罚

            foot_clearance = -0.1
            action_rate = -0.01
            smoothness = -0.01

            feet_air_time =  0.0 
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -0.
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0

    # ==========================
    # 8. 归一化
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
    # 9. 噪声
    # ==========================
    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01    # 关节编码器噪声
            dof_vel = 0.1     # 速度计算噪声 (通常比较大)
            lin_vel = 0.1     # IMU/状态估计噪声
            ang_vel = 0.2     # 陀螺仪噪声
            gravity = 0.05    # 重力感应噪声
            height_measurements = 0.1 # 高度图噪声
    # ==========================
    # 10. 物理引擎
    # ==========================_reward_soft_dof_pos_limit
    class sim(LeggedRobotCfg.sim):
        dt = 0.005 # 物理仿真步长 5ms 关系到上位机频率
        substeps = 1 # 在每个 dt 内部，物理引擎还要细分几步来算
        gravity = [0., 0., -9.81] # 重力加速度
        up_axis = 1 # Z轴为上方向
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
# 训练参数 (PPO Runner)
# ==========================
            solver_type = 1 # 0: PGS, 1: TGS
            num_position_iterations = 4 # 次数越高，关节连接越紧密，不容易“脱臼”。
            num_velocity_iterations = 1 # 求解器计算“速度误差”的次数。主要解决“碰撞后的反弹速度”和“摩擦力”
            contact_offset = 0.01 # 碰撞检测距离
            rest_offset = 0.0 # 物体静止时的平衡距离
            bounce_threshold_velocity = 0.5 # 反弹阈值。如果撞击速度小于 0.5m/s，就不发生反弹（完全非弹性碰撞）
            max_depenetration_velocity = 1.0 # 最大去穿模速度。如果物体不幸穿模了（嵌进地里），引擎会把它推出来。这个参数限制推出来的最大速度。
            default_buffer_size_multiplier = 5
            max_gpu_contact_pairs = 2**23 # 预分配显存缓冲区的大小

# ==========================
# 训练参数 (PPO Runner)
# ==========================
class Htdw4438CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005
        learning_rate = 3e-4 # 降低一点学习率

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'htdw_4438_himloco_v1.1'
        experiment_name = 'flat_htdw_4438'
        max_iterations = 2000 # 最大训练迭代次数
        save_interval = 200 # 每 200 次迭代保存一次模型

        # HimLoco核心配置
        policy_class_name = 'HIMActorCritic' 
        algorithm_class_name = 'HIMPPO'    
        num_steps_per_env = 48



# tensorboard --logdir .
