import numpy as np
import mujoco
import mujoco_viewer  # 切换到第三方 viewer 以获得一致的界面
import onnxruntime as ort
import os, time, yaml, re

try:
    import glfw
except ImportError:
    raise ImportError("请安装 glfw: pip install glfw")

# ===================== 1. 配置 (Configuration) =====================
class Cfg:
    # 路径配置自动适配项目结构
    ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "../../"))

    ROBOT_DIR = os.path.join(PROJECT_ROOT, "resources/robots/htdw_4438")
    XML_PATH = os.path.join(ROBOT_DIR, "xml/scene_debris_mixed.xml")
    MESHES_DIR = os.path.join(ROBOT_DIR, "meshes")

    YAML_PATH = os.path.join(PROJECT_ROOT, "deploy/deploy_mujoco/config/htdw_4438.yaml")
    ONNX_PATH = os.path.join(PROJECT_ROOT, "onnx/htdw_4438_standard_20260227_153916_model_600.onnx")

    sim_dt = 0.005              # 200Hz 物理步长
    decimation = 2              # 100Hz 策略频率

    # 控制增量与衰减
    vel_scales = [0.05, 0.05, 0.1] # x, y, yaw 步进速度
    vel_decay = 0.95               # 自动减速系数

    @classmethod
    def load_yaml(cls):
        """从 YAML 加载关键的 PD 参数、默认关节弧度与缩放因子"""
        with open(cls.YAML_PATH, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        cls.kps = np.array(config["kps"], dtype=np.float32)
        cls.kds = np.array(config["kds"], dtype=np.float32)
        cls.default_dof_pos = np.array(config["default_angles"], dtype=np.float32)

        cls.action_scale = float(config.get("action_scale", 0.05))
        cls.cmd_scale = np.array(config.get("cmd_scale", [2.0, 2.0, 0.25]), dtype=np.float32)
        cls.dof_pos_scale = float(config.get("dof_pos_scale", 1.0))
        cls.dof_vel_scale = float(config.get("dof_vel_scale", 0.05))
        cls.ang_vel_scale = float(config.get("ang_vel_scale", 0.25))

        cls.clip_obs = float(config.get("clip_obs", 100.0))
        cls.action_clip = float(config.get("action_clip", 100.0))

        cls.sim_dt = float(config.get("simulation_dt", cls.sim_dt))
        cls.decimation = int(config.get("control_decimation", cls.decimation))
        cls.cmd_init = np.array(config.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)

# ===================== 2. 控制器函数 =====================
def load_model(xml_path, meshes_dir):
    """参照 A1 的 deploy：用 from_xml_string + assets 规避 XML 中的绝对 meshdir/路径问题。"""
    if not os.path.exists(xml_path):
        raise FileNotFoundError(xml_path)
    if not os.path.isdir(meshes_dir):
        raise FileNotFoundError(meshes_dir)

    xml_dir = os.path.dirname(xml_path)
    assets = {}

    for filename in os.listdir(xml_dir):
        if filename.endswith(".xml") and filename != os.path.basename(xml_path):
            with open(os.path.join(xml_dir, filename), "rb") as f:
                assets[filename] = f.read()

    for filename in os.listdir(meshes_dir):
        if filename.lower().endswith((".stl", ".obj", ".dae")):
            with open(os.path.join(meshes_dir, filename), "rb") as f:
                assets[filename] = f.read()

    with open(xml_path, "r") as f:
        content = f.read()

    content = re.sub(
        r'file="[^"]*?([^\\\/"]+\.(?:stl|obj|dae))"',
        r'file="\1"',
        content,
        flags=re.IGNORECASE,
    )

    return mujoco.MjModel.from_xml_string(content, assets=assets)


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
    # q: [x, y, z, w] 与 IsaacGym/LeggedGym 一致
    q_w, q_vec = q[-1], q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

# ===================== 3. 主循环 =====================
def run_simulation():
    Cfg.load_yaml()
    
    # 加载模型与策略
    model = load_model(Cfg.XML_PATH, Cfg.MESHES_DIR)
    model.opt.timestep = Cfg.sim_dt
    data = mujoco.MjData(model)
    ort_session = ort.InferenceSession(Cfg.ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    # 初始化位置
    data.qpos[-12:] = Cfg.default_dof_pos
    data.qpos[2] = 0.15  # 初始化高度 (与训练 cfg.init_state.pos 对齐)
    mujoco.mj_forward(model, data)
    
    # 第三方 Viewer
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    cmd_vel = Cfg.cmd_init.copy()
    last_action = np.zeros(12, dtype=np.float32)
    target_dof_pos = Cfg.default_dof_pos.copy()

    ctrl_range = model.actuator_ctrlrange.copy()
    tau_limit = np.maximum(np.abs(ctrl_range[:, 0]), np.abs(ctrl_range[:, 1])).astype(np.float32)
    
    print("\n✅ 启动成功！")
    print("🎮 控制指南: [↑/↓] 前进后退 | [←/→] 左右转向 | [Shift + ←/→] 左右平移 | [空格] 停止")

    step_counter = 0
    while viewer.is_alive:
        step_start = time.time()

        # 1. 更新按键指令
        cmd_vel = update_keyboard_command(viewer.window, cmd_vel)

        # 2. 策略推理 (50Hz)
        if step_counter % Cfg.decimation == 0:
            # 构建 45 维观测向量
            q = data.qpos[-12:].astype(np.float32)
            dq = data.qvel[-12:].astype(np.float32)

            quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
            omega = data.sensor("angular-velocity").data.astype(np.float32)
            proj_g = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))

            obs = np.concatenate(
                [
                    omega * Cfg.ang_vel_scale,
                    proj_g,
                    cmd_vel * Cfg.cmd_scale,
                    (q - Cfg.default_dof_pos) * Cfg.dof_pos_scale,
                    dq * Cfg.dof_vel_scale,
                    last_action,
                ]
            ).astype(np.float32).reshape(1, -1)

            obs = np.clip(obs, -Cfg.clip_obs, Cfg.clip_obs)

            # 推理并更新动作
            raw_action = ort_session.run(None, {input_name: obs})[0][0].astype(np.float32)
            raw_action = np.clip(raw_action, -Cfg.action_clip, Cfg.action_clip)
            last_action = raw_action.copy()

            # 与训练时 LeggedRobot._compute_torques 对齐：
            # actions_scaled = actions * action_scale; actions_scaled[hip] *= 0.5
            scaled = raw_action * Cfg.action_scale
            scaled[[0, 3, 6, 9]] *= 0.5
            target_dof_pos = scaled + Cfg.default_dof_pos

        # 3. PD 控制 (200Hz)
        tau = Cfg.kps * (target_dof_pos - data.qpos[-12:]) - Cfg.kds * data.qvel[-12:]
        data.ctrl[:] = np.clip(tau, -tau_limit, tau_limit)

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
