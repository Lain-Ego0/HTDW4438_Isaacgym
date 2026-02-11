from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Htdw4438Cfg(LeggedRobotCfg):
    # ==========================
    # 1. 初始状态 (保留 4438 特性)
    # ==========================
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.15] # 初始高度
        default_joint_angles = { 
            'fl_hip_joint': 0.1, 'fl_thigh_joint': 0.8, 'fl_calf_joint': -1.5, # 前左
            'fr_hip_joint': -0.1, 'fr_thigh_joint': 0.8, 'fr_calf_joint': -1.5, # 前右
            'rl_hip_joint': 0.1,  'rl_thigh_joint': 0.8, 'rl_calf_joint': -1.5, # 后左
            'rr_hip_joint': -0.1, 'rr_thigh_joint': 0.8, 'rr_calf_joint': -1.5, # 后右
        }

    # ==========================
    # 2. 环境设置 (回归标准 LeggedGym)
    # ==========================
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 45 # 3（角速度）+3（重力投影）+3（命令控制）+12（关节位置）+12（关节速度）+12（上次网络输出）
        # 移除 HimLoco 的特权观测和历史堆叠
        num_privileged_obs = None 
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # 也可以改为 'trimesh' 用于崎岖地形训练
        measure_heights = False # 如果是平地训练，通常设为 False；崎岖地形设为 True

    # ==========================
    # 3. 控制参数 (保留 4438 硬件参数)
    # ==========================
    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 30.0}   # [N*m/rad] 
        damping = {'joint': 0.7}      # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.05           # 4438 设置较小，保留原设置
        decimation = 4                # A1 默认为4, 4438 原设为2。
                                      # 如果 sim.dt=0.005, decimation=4 -> 50Hz 控制频率
                                      # 如果需要 100Hz 控制，请改回 2

    # ==========================
    # 4. 指令范围 (保留 4438 速度限制)
    # ==========================
    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4 
        resampling_time = 10. 
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
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 disable, 0 enable
        flip_visual_attachments = False
        
        # 物理特性保留
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 9.5
        max_linear_velocity = 20.
        armature = 0.005

    # ==========================
    # 6. 域随机化 (标准 LeggedGym 配置)
    # ==========================
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

    # ==========================
    # 7. 奖励函数 (清理 HimLoco 特有项)
    # ==========================
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.13 # 4438 的目标高度
        
        class scales(LeggedRobotCfg.rewards.scales):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2 # 稍微降低惩罚权重，参考标准
            torques = -0.00001 # 需根据电机扭矩量级调整
            dof_vel = -0.0
            dof_acc = -2.5e-7
            base_height = -1.0 # 恢复标准名称
            feet_air_time =  1.0
            collision = -1.0
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.

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
    # 9. 物理引擎
    # ==========================
    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            default_buffer_size_multiplier = 5
            max_gpu_contact_pairs = 2**23

class Htdw4438CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1e-3 # 标准 LeggedGym 通常用稍大一点的学习率，或者保持 3e-4 也可以

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'htdw_4438_standard' # 修改实验名称
        max_iterations = 2000 
        save_interval = 200

'''

    How to run:
    export PYTHONPATH=$PWD
    python legged_gym/scripts/train.py --task=htdw_4438 --headless
    
'''

# export PYTHONPATH=. && python legged_gym/scripts/play.py --task=htdw_4438 --load_run Feb11_10-32-53_ --checkpoint 1400

# tensorboard --logdir .
