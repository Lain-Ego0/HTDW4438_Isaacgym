import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
import onnxruntime as ort
import os
import time
import re

# 【新增】引入 GLFW 库，用于直接读取底层键盘状态
try:
    import glfw
except ImportError:
    print("❌ 错误: 缺少 glfw 库。")
    print("请先在终端运行: pip install glfw")
    exit()

# ===================== 1. 路径与全局配置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

ROBOT_ROOT = os.path.join(PROJECT_ROOT, "resources/robots/a1")
XML_FILE_PATH = os.path.join(ROBOT_ROOT, "xml/scene.xml")
MESHES_FOLDER = os.path.join(ROBOT_ROOT, "meshes")
POLICY_MODEL_PATH = os.path.join(PROJECT_ROOT, "onnx/policy_1500.onnx")

class Cfg:
    default_dof_pos = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 
                                0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.double)
    class ObsScales:
        ang_vel = 0.25
        lin_vel = 2.0
        dof_pos = 1.0
        dof_vel = 0.05
    clip_obs = 5.0
    
    kps = np.array([60] * 12, dtype=np.double)
    kds = np.array([2.0] * 12, dtype=np.double)
    tau_limit = 20.0
    
    sim_duration = 60.0
    dt = 0.005
    decimation = 4

# ===================== 2. 控制逻辑 (主动轮询版) =====================
cmd_vel = np.array([0.0, 0.0, 0.0]) # [x, y, yaw]

def update_command_polling(window):
    """
    直接询问窗口：按键是否被按下？
    这种方式比 callback 稳定得多，不会漏键。
    """
    global cmd_vel
    step_lin = 0.05
    step_ang = 0.1
    decay = 0.95
    
    # 1. 直接读取按键状态 (PRESS=1, RELEASE=0)
    # 即使窗口没有焦点，有时候 glfw 也能捕获，但最好还是点一下窗口
    is_up    = glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS
    is_down  = glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS
    is_left  = glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS
    is_right = glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS
    is_enter = glfw.get_key(window, glfw.KEY_ENTER) == glfw.PRESS

    # 2. 根据状态更新速度
    if is_up:    cmd_vel[0] += step_lin
    if is_down:  cmd_vel[0] -= step_lin
    
    # 左右键控制旋转 (Yaw)
    if is_left:  cmd_vel[2] += step_ang
    if is_right: cmd_vel[2] -= step_ang
    
    # 急停
    if is_enter: cmd_vel[:] = 0.0

    # 3. 衰减与限幅
    cmd_vel[0:2] = np.clip(cmd_vel[0:2] * decay, -1.0, 1.0)
    cmd_vel[2]   = np.clip(cmd_vel[2] * decay,   -1.0, 1.0)
    cmd_vel[np.abs(cmd_vel) < 0.01] = 0.0

# ===================== 3. 核心工具函数 =====================
def quat_rotate_inverse(q, v):
    q_w, q_vec = q[-1], q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def get_obs(data):
    q = data.qpos.astype(np.double)[-12:]
    dq = data.qvel.astype(np.double)[-12:]
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    return q, dq, quat, omega

def load_model_robust(xml_path, meshes_dir):
    if not os.path.exists(xml_path): raise FileNotFoundError(xml_path)
    xml_dir = os.path.dirname(xml_path)
    with open(xml_path, 'r') as f: xml_content = f.read()
    assets = {}
    for filename in os.listdir(xml_dir):
        if filename.endswith('.xml') and filename != os.path.basename(xml_path):
            with open(os.path.join(xml_dir, filename), 'rb') as f:
                assets[filename] = f.read()
    mesh_files = [f for f in os.listdir(meshes_dir) if f.endswith('.stl')]
    for mf in mesh_files:
        with open(os.path.join(meshes_dir, mf), 'rb') as f:
            assets[mf] = f.read()
    xml_content = re.sub(r'file="[^"]*?([^\/"]+\.stl)"', r'file="\1"', xml_content)
    return mujoco.MjModel.from_xml_string(xml_content, assets=assets)

# ===================== 4. 主循环 =====================
if __name__ == '__main__':
    print(f"🧠 加载策略: {POLICY_MODEL_PATH}")
    policy = ort.InferenceSession(POLICY_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name, output_name = policy.get_inputs()[0].name, policy.get_outputs()[0].name

    model = load_model_robust(XML_FILE_PATH, MESHES_FOLDER)
    model.opt.timestep = Cfg.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data) 

    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    # 【关键】获取底层窗口句柄，用于直接查询键盘
    window_handle = viewer.window 

    last_action = np.zeros(12, dtype=np.float32)
    target_q = Cfg.default_dof_pos.copy()
    
    print("\n" + "="*50)
    print("🚀 仿真启动! (主动轮询模式 - 极度稳定)")
    print("⌨️  按键说明:")
    print("   ⬆ / ⬇  : 前进 / 后退")
    print("   ⬅ / ➡  : 左转 / 右转")
    print("   Enter   : 急停")
    print("ℹ️  请务必点击一次黑色仿真窗口以获取焦点！")
    print("="*50 + "\n")

    obs_list = []
    
    # 循环
    for i in tqdm(range(int(Cfg.sim_duration / Cfg.dt))):
        if not viewer.is_alive: break
        
        # 【核心修改】不再依赖 callback，每一帧主动去查键盘
        # 即使窗口系统卡顿，这行代码也会强制检查按键状态
        update_command_polling(window_handle)
        
        # 调试打印：只要有速度就显示，证明控制生效
        if i % 50 == 0 and np.linalg.norm(cmd_vel) > 0.1:
            print(f"🎮 速度: X={cmd_vel[0]:.2f} Yaw={cmd_vel[2]:.2f}")

        # === 策略更新 (50Hz) ===
        if i % Cfg.decimation == 0:
            q, dq, quat, omega = get_obs(data)
            proj_gravity = quat_rotate_inverse(quat, np.array([0., 0., -1.]))
            
            obs_list = [
                omega * Cfg.ObsScales.ang_vel,            
                proj_gravity,                              
                cmd_vel * [Cfg.ObsScales.lin_vel, Cfg.ObsScales.lin_vel, Cfg.ObsScales.ang_vel],
                (q - Cfg.default_dof_pos) * Cfg.ObsScales.dof_pos, 
                dq * Cfg.ObsScales.dof_vel,                
                last_action                                
            ]
            obs = np.concatenate(obs_list).astype(np.float32).reshape(1, -1)
            obs = np.clip(obs, -Cfg.clip_obs, Cfg.clip_obs)
            
            raw_action = policy.run([output_name], {input_name: obs})[0][0]
            raw_action = np.clip(raw_action, -10, 10)
            last_action = raw_action.copy()
            
            scaled_action = raw_action.copy()
            scaled_action[[0, 3, 6, 9]] *= 0.5 
            scaled_action *= 0.25              
            target_q = scaled_action + Cfg.default_dof_pos

        # === PD控制 ===
        tau = Cfg.kps * (target_q - data.qpos[-12:]) + Cfg.kds * (0 - data.qvel[-12:])
        tau = np.clip(tau, -Cfg.tau_limit, Cfg.tau_limit)
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        
        # 这里如果不休眠，仿真会跑得比实时快很多
        # 如果你觉得反应慢，可以把这个时间改小或者注释掉
        time.sleep(Cfg.dt)

    viewer.close()