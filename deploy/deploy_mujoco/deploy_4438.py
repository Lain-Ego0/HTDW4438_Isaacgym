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
    YAML_PATH = os.path.join(PROJECT_ROOT, "deploy/deploy_mujoco/config/htdw_4438.yaml")
    XML_PATH = os.path.join(PROJECT_ROOT, "resources/robots/htdw_4438/xml/scene.xml")
    ONNX_PATH = os.path.join(PROJECT_ROOT, "onnx/policy_1500.onnx")

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
                    
                    # --- D. 推理 ---
                    # 直接将 45维的 obs 传给模型
                    ort_outs = ort_session.run(None, {input_name: obs.reshape(1, -1)})
                    raw_action = ort_outs[0][0]

                    # --- E. 后处理 ---
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
