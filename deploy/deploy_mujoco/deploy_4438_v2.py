import numpy as np
import mujoco
import mujoco_viewer  # 切换到第三方 viewer 以获得一致的界面
import onnxruntime as ort
import os, time, yaml

try:
    import glfw
except ImportError:
    raise ImportError("请安装 glfw: pip install glfw")

# ===================== 1. 配置 (Configuration) =====================
class Cfg:
    # 路径配置自动适配项目结构
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../")) 
    YAML_PATH = os.path.join(PROJECT_ROOT, "deploy/deploy_mujoco/config/htdw_4438.yaml")
    XML_PATH = os.path.join(PROJECT_ROOT, "resources/robots/htdw_4438/xml/scene.xml")
    ONNX_PATH = os.path.join(PROJECT_ROOT, "onnx/policy_600.onnx")

    sim_dt = 0.005              # 200Hz 物理步长
    decimation = 4              # 50Hz 策略频率
    
    # 控制增量与衰减
    vel_scales = [0.05, 0.05, 0.1] # x, y, yaw 步进速度
    vel_decay = 0.95               # 自动减速系数

    @classmethod
    def load_yaml(cls):
        """从 YAML 加载关键的 PD 参数和默认关节弧度"""
        with open(cls.YAML_PATH, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        cls.kps = np.array(config['kps'], dtype=np.float32)
        cls.kds = np.array(config['kds'], dtype=np.float32)
        cls.default_dof_pos = np.array(config['default_angles'], dtype=np.float32)
        cls.action_scale = config['action_scale']
        cls.cmd_scale = np.array(config['cmd_scale'], dtype=np.float32)

        # --- 新增以下读取逻辑 ---
        cls.ang_vel_scale = config.get('ang_vel_scale', 0.25)
        cls.dof_vel_scale = config.get('dof_vel_scale', 0.05)
        cls.lin_vel_scale = config.get('lin_vel_scale', 2.0)

# ===================== 2. 控制器函数 =====================
def update_keyboard_command(window, cmd):
    """
    使用 glfw 直接读取按键，支持 Shift 组合键
    cmd: [vx, vy, yaw_rate]
    """
    # 获取按键状态
    key_up = glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS
    key_down = glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS
    key_left = glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS
    key_right = glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS
    key_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or 
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
    key_enter = glfw.get_key(window, glfw.KEY_ENTER) == glfw.PRESS

    # 1. 前后控制
    if key_up:    cmd[0] += Cfg.vel_scales[0]
    if key_down:  cmd[0] -= Cfg.vel_scales[0]
    
    # 2. 左右平移 vs 转向控制
    if key_shift: # 开启平移模式
        if key_left:  cmd[1] += Cfg.vel_scales[1]
        if key_right: cmd[1] -= Cfg.vel_scales[1]
        cmd[2] *= Cfg.vel_decay # 平移时减少转向指令
    else:         # 开启转向模式
        if key_left:  cmd[2] += Cfg.vel_scales[2]
        if key_right: cmd[2] -= Cfg.vel_scales[2]
        cmd[1] *= Cfg.vel_decay # 转向时减少平移指令

    # 3. 停止逻辑
    if key_enter: cmd[:] = 0.0
    
    # 指令后处理：衰减与限幅
    cmd[:] = np.clip(cmd * Cfg.vel_decay, -1.0, 1.5)
    if np.linalg.norm(cmd) < 0.01: cmd[:] = 0.0
    return cmd

def quat_rotate_inverse(q, v):
    """处理四元数旋转：World -> Body"""
    # 4438 模型中通常 q 是 [w, x, y, z]
    q_w, q_vec = q[0], q[1:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

# ===================== 3. 主循环 =====================
def run_simulation():
    Cfg.load_yaml()
    
    # 加载模型与策略
    model = mujoco.MjModel.from_xml_path(Cfg.XML_PATH)
    data = mujoco.MjData(model)
    ort_session = ort.InferenceSession(Cfg.ONNX_PATH)
    input_name = ort_session.get_inputs()[0].name

    # 初始化位置
    data.qpos[7:] = Cfg.default_dof_pos
    data.qpos[2] = 0.45 
    
    # 启动相同的 Viewer
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    cmd_vel = np.zeros(3, dtype=np.float32)
    last_action = np.zeros(12, dtype=np.float32)
    target_dof_pos = Cfg.default_dof_pos.copy()
    
    print("\n✅ 启动成功！界面已同步至 A1 同款风格")
    print("🎮 控制指南: [↑/↓] 前进后退 | [←/→] 左右转向 | [Shift + ←/→] 左右平移 | [Enter] 停止")

    step_counter = 0
    while viewer.is_alive:
        step_start = time.time()

        # 1. 更新按键指令
        cmd_vel = update_keyboard_command(viewer.window, cmd_vel)

        # 2. 策略推理 (100Hz)
        if step_counter % Cfg.decimation == 0:
            # 构建 45 维观测向量
            qj = (data.qpos[7:] - Cfg.default_dof_pos)
            dqj = data.qvel[6:]
            quat = data.qpos[3:7] 
            omega = data.qvel[3:6]
            proj_g = quat_rotate_inverse(quat, np.array([0., 0., -1.]))
            
            # # 组合 obs (注意顺序需要与训练代码一致)
            # obs = np.concatenate([
            #     omega, proj_g, cmd_vel * Cfg.cmd_scale, qj, dqj, last_action
            # ]).astype(np.float32).reshape(1, -1)

            # --- 乘以缩放因子 ---
            obs = np.concatenate([
                omega * Cfg.ang_vel_scale,       # 乘以 0.25
                proj_g,
                cmd_vel * Cfg.cmd_scale,
                qj,                              # 通常 scale 为 1.0
                dqj * Cfg.dof_vel_scale,         # 乘以 0.05
                last_action
            ]).astype(np.float32).reshape(1, -1)
            # -------------------------------

            # 推理并更新动作
            raw_action = ort_session.run(None, {input_name: obs})[0][0]
            last_action = np.clip(raw_action, -10.0, 10.0)
            target_dof_pos = (last_action * Cfg.action_scale) + Cfg.default_dof_pos

        # 3. PD 控制 (200Hz)
        tau = Cfg.kps * (target_dof_pos - data.qpos[7:]) - Cfg.kds * data.qvel[6:]
        data.ctrl[:] = np.clip(tau, -40, 40)

        mujoco.mj_step(model, data)
        viewer.render()
        
        # 帧率同步
        step_counter += 1
        time_until_next = Cfg.sim_dt - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

    viewer.close()

if __name__ == "__main__":
    run_simulation()