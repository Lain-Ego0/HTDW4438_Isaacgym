# import time
# import os
# import numpy as np
# import mujoco
# import mujoco.viewer
# import onnxruntime as ort
# import yaml
# from collections import deque
# from pynput import keyboard

# # ================= 1. 绝对路径配置 =================
# # 根目录 
# PROJECT_ROOT = "/home/sunteng/Documents/GitHub/Lain_isaacgym"
# # 配置文件路径 
# YAML_PATH = os.path.join(PROJECT_ROOT, "deploy/deploy_mujoco/configs/htdw_4438.yaml")
# # 模型文件路径
# XML_PATH = os.path.join(PROJECT_ROOT, "resources/robots/htdw_4438/xml/scene.xml")
# ONNX_PATH = os.path.join(PROJECT_ROOT, "onnx/HTDW_4438.onnx") # 输出路径

# # 打印路径信息，便于调试路径是否正确
# print(f"YAML: {YAML_PATH}")
# print(f"XML : {XML_PATH}")
# print(f"ONNX: {ONNX_PATH}")

# # ================= 2. 全局变量 =================
# # [vx, vy, omega]
# cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32) 
# paused = False
# default_dof_pos = None # 将在 main 中加载

# # ================= 3. 辅助函数 =================

# def quat_rotate_inverse(q, v):
#     """
#     参考 deploy_lanbu.py: 计算向量 v 在四元数 q 表示的坐标系下的逆旋转
#     用于将重力向量转到机身坐标系
#     q: [w, x, y, z] (MuJoCo 格式)
#     v: [x, y, z]
#     """
#     q_w = q[0]
#     q_vec = q[1:4]
    
#     a = v * (2.0 * q_w**2 - 1.0)
#     b = np.cross(q_vec, v) * q_w * 2.0
#     c = q_vec * np.dot(q_vec, v) * 2.0
#     return a - b + c

# def pd_control(target_q, q, kp, target_dq, dq, kd):
#     return (target_q - q) * kp + (target_dq - dq) * kd

# # def pd_control(target_q, q, kp, kd, qvel, kds_val):
# #     # 修改：确保传入并使用了微分增益 kds_val
# #     return kp * (target_q - q) - kds_val * qvel

# # ================= 4. 键盘控制 =================
# def on_press(key):
#     global cmd
#     try:
#         if key == keyboard.Key.up:
#             cmd[0] = 0.6  # 前进
#         elif key == keyboard.Key.down:
#             cmd[0] = -0.4 # 后退
#         elif key == keyboard.Key.left:
#             cmd[2] = 0.8  # 左转
#         elif key == keyboard.Key.right:
#             cmd[2] = -0.8 # 右转
#     except AttributeError:
#         pass

# def on_release(key):
#     global cmd
#     try:
#         if key == keyboard.Key.up or key == keyboard.Key.down:
#             cmd[0] = 0.0
#         elif key == keyboard.Key.left or key == keyboard.Key.right:
#             cmd[2] = 0.0
#     except AttributeError:
#         pass

# def key_callback(keycode):
#     global paused
#     if chr(keycode) == ' ':
#         paused = not paused
#         print(f"Paused: {paused}")

# # ================= 5. 主程序 =================
# def run_simulation():
#     global cmd, default_dof_pos
    
#     # --- 加载 YAML 配置 ---
#     if not os.path.exists(YAML_PATH):
#         print(f"错误: 找不到配置文件 {YAML_PATH}")
#         return

#     with open(YAML_PATH, "r") as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
    
#     # 提取参数
#     sim_dt = 0.005 # config['simulation_dt']
#     control_decimation = 2 # config['control_decimation'] 强制为 2 (100Hz)
    
#     kps = np.array(config['kps'], dtype=np.float32)
#     kds = np.array(config['kds'], dtype=np.float32)
#     default_dof_pos = np.array(config['default_angles'], dtype=np.float32)
    
#     # 缩放因子
#     lin_vel_scale = config['lin_vel_scale']
#     ang_vel_scale = config['ang_vel_scale']
#     dof_pos_scale = config['dof_pos_scale']
#     dof_vel_scale = config['dof_vel_scale']
#     action_scale = config['action_scale']
#     cmd_scale = np.array(config['cmd_scale'], dtype=np.float32)

#     # --- 加载 MuJoCo & ONNX ---
#     if not os.path.exists(XML_PATH):
#         print(f"错误: 找不到模型文件 {XML_PATH}")
#         return
        
#     print("正在加载 MuJoCo 模型...")
#     model = mujoco.MjModel.from_xml_path(XML_PATH)
#     data = mujoco.MjData(model)
#     model.opt.timestep = sim_dt

#     # print(f"正在加载 ONNX: {ONNX_PATH}")
#     # ort_session = ort.InferenceSession(ONNX_PATH)
#     # input_name = ort_session.get_inputs()[0].name

#     print(f"正在加载 ONNX: {ONNX_PATH}")
#     ort_session = ort.InferenceSession(ONNX_PATH)
#     input_name = ort_session.get_inputs()[0].name
#     # 检查一下输入维度，确保是 270
#     input_shape = ort_session.get_inputs()[0].shape
#     print(f"ONNX Input Shape: {input_shape}")

#     # --- 初始化状态 ---
#     data.qpos[7:] = default_dof_pos
#     data.qpos[2] = 0.5 # 起始高度
#     mujoco.mj_forward(model, data)
    
#     target_dof_pos = default_dof_pos.copy()
#     action = np.zeros(12, dtype=np.float32)
    
#     # 键盘监听
#     listener = keyboard.Listener(on_press=on_press, on_release=on_release)
#     listener.start()
#     print("仿真开始！使用方向键控制移动，空格键暂停。")

#     # === 新增：历史观测队列 ===
#     # 长度为 6，对应 config 中的 num_observations / num_one_step_observations
#     history_len = 6
#     obs_dim = 45
#     # 初始化全为 0 的队列
#     obs_history_buffer = deque([np.zeros(obs_dim, dtype=np.float32) for _ in range(history_len)], maxlen=history_len)

#     # --- 仿真循环 ---
#     with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
#         step_counter = 0
#         while viewer.is_running():
#             step_start = time.time()
            
#             if not paused:
#                 # ================= 策略控制 (100Hz) =================
#                 if step_counter % control_decimation == 0:
#                     # 1. 获取传感器数据
#                     qj = data.qpos[7:]
#                     dqj = data.qvel[6:]
#                     quat = data.qpos[3:7] # [w, x, y, z]
#                     omega = data.qvel[3:6] # 机身角速度
                    
#                     # 2. 数据转换与归一化
#                     gravity_vec = np.array([0., 0., -1.], dtype=np.float32)
#                     proj_gravity = quat_rotate_inverse(quat, gravity_vec)
                    
#                     qj_norm = (qj - default_dof_pos) * dof_pos_scale
#                     dqj_norm = dqj * dof_vel_scale
#                     omega_norm = omega * ang_vel_scale
#                     cmd_norm = cmd * cmd_scale
                    
#                     # 3. 构建观测向量 (45维)
#                     # 注意：顺序必须严格遵守 htdw_4438_config.py
#                     # 训练配置: AngVel(3) + Gravity(3) + Cmd(3) + DofPos(12) + DofVel(12) + LastAction(12)
#                     obs_list = []
#                     obs_list.extend(omega_norm)      # 0-2
#                     obs_list.extend(proj_gravity)    # 3-5
#                     obs_list.extend(cmd_norm)        # 6-8
#                     obs_list.extend(qj_norm)         # 9-20
#                     obs_list.extend(dqj_norm)        # 21-32
#                     obs_list.extend(action)          # 33-44
                    
#                     obs_raw = np.array(obs_list, dtype=np.float32)
                    
#                     # # === 核心修改：更新历史队列并构建输入 ===
#                     # # 将最新观测加入队列左侧（如果你的训练代码是把最新帧放在最前面）
#                     # # 或者 append (如果最新帧在最后)。
#                     # # ⚠️ 这里的顺序非常关键！
#                     # # LeggedGym 通常的做法：obs = torch.cat((current_obs, obs_history_frames...), dim=1)
#                     # # 也就是说：[当前帧, 历史1, 历史2, 历史3, 历史4, 历史5]
                    
#                     # obs_history_buffer.appendleft(obs_raw) 
                    
#                     # # 展平为一维数组 [270]
#                     # # buffer[0] 是最新帧, buffer[5] 是最旧帧
#                     # full_history_input = np.concatenate(obs_history_buffer)
                    
#                     # # 4. 适配 ONNX 输入 (270维)
#                     # # 此时全输入即为历史数据，Latent 会在 ONNX 内部计算

#                     # 4. 适配 ONNX 输入 (64维)
#                     # 你的模型输入 = 45 (Obs) + 19 (Latent)
#                     full_input = np.zeros(64, dtype=np.float32)
#                     full_input[:45] = obs_raw
#                     # full_input[45:] = 0.0 # Latent 留空
                    
#                     # 5. 推理
#                     outputs = ort_session.run(None, {input_name: full_input.reshape(1, -1)})
#                     raw_action = outputs[0][0]
                    
#                     # 6. 处理输出
#                     raw_action = np.clip(raw_action, -10, 10)
#                     action = raw_action # 更新 LastAction
#                     target_dof_pos = raw_action * action_scale + default_dof_pos

#                 # ================= 物理执行 (500Hz) =================
#                 tau = pd_control(target_dof_pos, data.qpos[7:], kps, 
#                                  np.zeros_like(kds), data.qvel[6:], kds)
#                 tau = np.clip(tau, -40, 40) 
#                 data.ctrl[:] = tau
                
#                 mujoco.mj_step(model, data)
#                 step_counter += 1

#             viewer.sync()
            
#             # 帧率同步
#             time_until_next_step = model.opt.timestep - (time.time() - step_start)
#             if time_until_next_step > 0:
#                 time.sleep(time_until_next_step)

# if __name__ == "__main__":
#     run_simulation()

import time
import os
import yaml
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort

# ===================== 1. 配置 (Configuration) =====================
class Cfg:
    # --- 1.1 路径配置 (使用相对路径，提高移植性) ---
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 假设文件结构保持原样：
    # PROJECT_ROOT/deploy/deploy_mujoco/deploy_4438.py (本文件)
    # PROJECT_ROOT/deploy/deploy_mujoco/configs/htdw_4438.yaml
    # PROJECT_ROOT/resources/robots/htdw_4438/xml/scene.xml
    # PROJECT_ROOT/onnx/HTDW_4438.onnx
    
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../")) 
    YAML_PATH = os.path.join(PROJECT_ROOT, "deploy/deploy_mujoco/configs/htdw_4438.yaml")
    XML_PATH = os.path.join(PROJECT_ROOT, "resources/robots/htdw_4438/xml/scene.xml")
    ONNX_PATH = os.path.join(PROJECT_ROOT, "onnx/HTDW_4438.onnx")

    # --- 1.2 仿真与控制参数 ---
    sim_dt = 0.005              # 物理步长
    decimation = 2              # 控制频率分频 (100Hz Policy / 200Hz Sim)
    
    # 动作与观测限制
    action_clip = 10.0
    tau_limit = 40.0
    
    # --- 1.3 运行时变量 (将在 load_config 中填充) ---
    kps = None
    kds = None
    default_dof_pos = None
    
    # 缩放因子
    lin_vel_scale = 1.0
    ang_vel_scale = 1.0
    dof_pos_scale = 1.0
    dof_vel_scale = 1.0
    action_scale = 1.0
    cmd_scale = np.array([1.0, 1.0, 1.0])

    @classmethod
    def load_yaml(cls):
        """加载 YAML 配置文件并更新类属性"""
        if not os.path.exists(cls.YAML_PATH):
            raise FileNotFoundError(f"Config not found: {cls.YAML_PATH}")
            
        with open(cls.YAML_PATH, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        cls.kps = np.array(config['kps'], dtype=np.float32)
        cls.kds = np.array(config['kds'], dtype=np.float32)
        cls.default_dof_pos = np.array(config['default_angles'], dtype=np.float32)
        
        cls.lin_vel_scale = config['lin_vel_scale']
        cls.ang_vel_scale = config['ang_vel_scale']
        cls.dof_pos_scale = config['dof_pos_scale']
        cls.dof_vel_scale = config['dof_vel_scale']
        cls.action_scale = config['action_scale']
        cls.cmd_scale = np.array(config['cmd_scale'], dtype=np.float32)
        
        print(f"✅ Config Loaded from: {cls.YAML_PATH}")

# ===================== 2. 工具函数 (Utils) =====================
def quat_rotate_inverse(q, v):
    """计算向量 v 在四元数 q 表示的坐标系下的逆旋转 (World frame to Body frame)"""
    q_w = q[0]
    q_vec = q[1:4]
    
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

class CommandHandler:
    """处理键盘输入，替代 pynput，使用 MuJoCo 原生回调"""
    def __init__(self):
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32) # [vx, vy, omega]
        self.paused = False
        # 速度增量
        self.vel_inc_x = 0.2
        self.vel_inc_w = 0.4

    def key_callback(self, keycode):
        # 简单的状态机或按键映射
        # keycode 对应 ASCII 码
        char_key = chr(keycode) if keycode <= 255 else None
        
        if keycode == 265: # Up Arrow
            self.cmd[0] += self.vel_inc_x
        elif keycode == 264: # Down Arrow
            self.cmd[0] -= self.vel_inc_x
        elif keycode == 263: # Left Arrow
            self.cmd[2] += self.vel_inc_w
        elif keycode == 262: # Right Arrow
            self.cmd[2] -= self.vel_inc_w
        elif keycode == 32:  # Space
            self.paused = not self.paused
            self.cmd[:] = 0.0 # 暂停时重置指令
            print(f"Paused: {self.paused}")
        elif keycode == 257: # Enter (Reset cmd)
            self.cmd[:] = 0.0
            
        # 限制范围
        self.cmd[0] = np.clip(self.cmd[0], -1.0, 1.5)
        self.cmd[2] = np.clip(self.cmd[2], -2.0, 2.0)

# ===================== 3. 主程序 (Main) =====================
def run_simulation():
    # 1. 初始化配置
    try:
        Cfg.load_yaml()
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return

    # 2. 加载模型
    if not os.path.exists(Cfg.XML_PATH):
        print(f"❌ XML not found: {Cfg.XML_PATH}")
        return
    
    print(f"🚀 Loading MuJoCo Model: {Cfg.XML_PATH}")
    model = mujoco.MjModel.from_xml_path(Cfg.XML_PATH)
    model.opt.timestep = Cfg.sim_dt
    data = mujoco.MjData(model)

    # 3. 加载 ONNX
    print(f"🧠 Loading Policy: {Cfg.ONNX_PATH}")
    ort_session = ort.InferenceSession(Cfg.ONNX_PATH)
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    print(f"   Input Shape: {input_shape}") # 预期: [batch, 64]

    # 4. 初始化状态
    data.qpos[7:] = Cfg.default_dof_pos
    data.qpos[2] = 0.5 # 初始高度
    mujoco.mj_forward(model, data)

    # 运行时变量
    cmd_handler = CommandHandler()
    action = np.zeros(12, dtype=np.float32)
    target_dof_pos = Cfg.default_dof_pos.copy()
    
    # 5. 仿真循环
    print("🎮 Control: [Arrows] Move | [Space] Pause | [Enter] Stop")
    
    with mujoco.viewer.launch_passive(model, data, key_callback=cmd_handler.key_callback) as viewer:
        step_counter = 0
        
        while viewer.is_running():
            step_start = time.time()
            
            if not cmd_handler.paused:
                # ================= 策略循环 (100Hz) =================
                # 使用取模方式降频 (Decimation)
                if step_counter % Cfg.decimation == 0:
                    # --- A. 获取传感器数据 ---
                    qj = data.qpos[7:]
                    dqj = data.qvel[6:]
                    quat = data.qpos[3:7]  # [w, x, y, z]
                    omega = data.qvel[3:6] # 机身角速度

                    # --- B. 数据处理 ---
                    gravity_vec = np.array([0., 0., -1.], dtype=np.float32)
                    proj_gravity = quat_rotate_inverse(quat, gravity_vec)

                    # 归一化
                    qj_norm = (qj - Cfg.default_dof_pos) * Cfg.dof_pos_scale
                    dqj_norm = dqj * Cfg.dof_vel_scale
                    omega_norm = omega * Cfg.ang_vel_scale
                    cmd_norm = cmd_handler.cmd * Cfg.cmd_scale

                    # --- C. 构建观测向量 (45维) ---
                    # 顺序: AngVel(3) + Gravity(3) + Cmd(3) + DofPos(12) + DofVel(12) + LastAction(12)
                    obs = np.concatenate([
                        omega_norm,
                        proj_gravity,
                        cmd_norm,
                        qj_norm,
                        dqj_norm,
                        action
                    ]).astype(np.float32)

                    # --- D. 适配 ONNX 输入 (64维) ---
                    # 你的模型输入是 64维 (45 Obs + 19 Latent/Zeros)
                    full_input = np.zeros(64, dtype=np.float32)
                    full_input[:45] = obs
                    
                    # --- E. 推理 ---
                    ort_outs = ort_session.run(None, {input_name: full_input.reshape(1, -1)})
                    raw_action = ort_outs[0][0]
                    
                    # --- F. 后处理 ---
                    raw_action = np.clip(raw_action, -Cfg.action_clip, Cfg.action_clip)
                    action = raw_action # 更新 LastAction 用于下一帧
                    
                    # 计算目标位置: target = default + action * scale
                    target_dof_pos = (raw_action * Cfg.action_scale) + Cfg.default_dof_pos

                # ================= 物理循环 (PD Control) =================
                # PD Control: Kp * (target - current) + Kd * (0 - velocity)
                # 注意: 4438 源码中 Kd 项是 (target_dq - dq)，通常 target_dq 为 0
                tau = Cfg.kps * (target_dof_pos - data.qpos[7:]) - Cfg.kds * data.qvel[6:]
                
                # 限制力矩
                tau = np.clip(tau, -Cfg.tau_limit, Cfg.tau_limit)
                data.ctrl[:] = tau
                
                # 物理步进
                mujoco.mj_step(model, data)
                step_counter += 1
            
            # 同步画面
            viewer.sync()

            # 帧率控制 (Real-time sync)
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

if __name__ == "__main__":
    run_simulation()
    