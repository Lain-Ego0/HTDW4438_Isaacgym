import time
import os
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
import yaml
from collections import deque
from pynput import keyboard
from threading import Lock

# ===================== 1. 工程路径配置（跨机器兼容，相对路径优先）=====================
# 自动获取当前脚本所在目录为项目根目录，避免硬编码路径跨机器失效
try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    # 若脚本不在根目录，可手动指定上级目录，例：
    # PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # 兼容jupyter/交互式运行场景
    PROJECT_ROOT = "/home/sunteng/Documents/GitHub/Lain_isaacgym"

# 配置/模型文件路径（集中管理，便于修改）
CONFIG = {
    "yaml_path": os.path.join(PROJECT_ROOT, "deploy/deploy_mujoco/configs/htdw_4438.yaml"),
    "mj_xml_path": os.path.join(PROJECT_ROOT, "resources/robots/htdw_4438/xml/scene.xml"),
    "onnx_policy_path": os.path.join(PROJECT_ROOT, "onnx/HTDW_4438.onnx"),
    # 仿真核心参数（与训练严格对齐，优先从yaml读取，此处为兜底默认值）
    "sim_dt": 0.005,                # 物理仿真步长 200Hz
    "control_decimation": 4,        # 控制分频 与训练decimation一致！50Hz策略控制
    "history_len": 6,               # 历史观测帧长度 与训练严格对齐
    "obs_dim": 45,                  # 单帧观测维度 与训练严格对齐
    "dof_num": 12,                  # 关节数量 四足机器人3*4=12
    "fall_threshold": 0.25,         # 摔倒检测高度阈值（机身z轴高度）
    "action_smooth_alpha": 0.25,    # 动作一阶低通滤波系数 与第一段代码对齐
    "cmd_dead_zone": 0.05,          # 指令死区，过滤微小噪声
}

# ===================== 2. 线程安全的状态管理器（消除全局变量隐患）=====================
class RobotState:
    """线程安全的机器人状态管理，解决键盘子线程与主线程的读写冲突"""
    def __init__(self):
        self._lock = Lock()
        # 控制指令 [vx, vy, yaw_rate] 单位：m/s, m/s, rad/s
        self._cmd = np.zeros(3, dtype=np.float32)
        # 仿真状态
        self._paused = False
        self._reset_flag = False
        self._emergency_stop = False
        self._exit_flag = False
        # 速度档位
        self._speed_scale = 1.0
        self._max_speed_scale = 2.0
        self._min_speed_scale = 0.2

    # 指令读写
    @property
    def cmd(self):
        with self._lock:
            return self._cmd.copy()
    
    def set_cmd(self, axis, value):
        with self._lock:
            self._cmd[axis] = value

    # 暂停状态
    @property
    def paused(self):
        with self._lock:
            return self._paused
    
    def toggle_pause(self):
        with self._lock:
            self._paused = not self._paused
            print(f"[INFO] 仿真状态: {'暂停' if self._paused else '运行'}")

    # 复位标志
    @property
    def reset_flag(self):
        with self._lock:
            flag = self._reset_flag
            self._reset_flag = False
            return flag
    
    def trigger_reset(self):
        with self._lock:
            self._reset_flag = True
            self._cmd = np.zeros(3, dtype=np.float32)
            print("[INFO] 触发机器人复位")

    # 急停标志
    @property
    def emergency_stop(self):
        with self._lock:
            return self._emergency_stop
    
    def toggle_emergency_stop(self):
        with self._lock:
            self._emergency_stop = not self._emergency_stop
            self._cmd = np.zeros(3, dtype=np.float32)
            print(f"[WARN] 紧急停止: {'开启' if self._emergency_stop else '关闭'}")

    # 退出标志
    @property
    def exit_flag(self):
        with self._lock:
            return self._exit_flag
    
    def trigger_exit(self):
        with self._lock:
            self._exit_flag = True
            print("[INFO] 触发程序退出")

    # 速度档位调节
    def speed_up(self):
        with self._lock:
            self._speed_scale = min(self._speed_scale + 0.2, self._max_speed_scale)
            print(f"[INFO] 速度档位: {self._speed_scale:.1f}x")

    def speed_down(self):
        with self._lock:
            self._speed_scale = max(self._speed_scale - 0.2, self._min_speed_scale)
            print(f"[INFO] 速度档位: {self._speed_scale:.1f}x")
    
    @property
    def speed_scale(self):
        with self._lock:
            return self._speed_scale

# ===================== 3. 核心工具函数（与训练代码严格对齐）=====================
def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    四元数逆旋转，将世界坐标系向量转换到机体坐标系
    与Legged Gym训练代码完全对齐，输入格式严格匹配MuJoCo
    :param q: 四元数 [w, x, y, z] (MuJoCo qpos原生格式)
    :param v: 世界坐标系3维向量
    :return: 机体坐标系下的3维向量
    """
    # 四元数归一化，防止长时间仿真漂移
    q = q / np.linalg.norm(q)
    q_w = q[0]
    q_vec = q[1:4]
    
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def pd_control(
    target_q: np.ndarray,
    current_q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    current_dq: np.ndarray,
    kd: np.ndarray
) -> np.ndarray:
    """
    PD关节控制，与训练代码底层逻辑完全对齐
    :param target_q: 目标关节位置
    :param current_q: 当前关节位置
    :param kp: 比例增益
    :param target_dq: 目标关节速度
    :param current_dq: 当前关节速度
    :param kd: 微分增益
    :return: 关节控制力矩
    """
    return kp * (target_q - current_q) + kd * (target_dq - current_dq)

def load_yaml_config(yaml_path: str) -> dict:
    """加载并校验YAML配置文件，缺失关键参数直接抛出明确异常"""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        try:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            raise RuntimeError(f"YAML配置文件解析失败: {e}") from e

    # 关键参数校验
    required_keys = ["kps", "kds", "default_angles", "lin_vel_scale", "ang_vel_scale",
                     "dof_pos_scale", "dof_vel_scale", "action_scale", "cmd_scale", "tau_limit"]
    for key in required_keys:
        if key not in yaml_config:
            raise KeyError(f"YAML配置缺失关键参数: {key}")
    
    # 维度校验
    dof_num = CONFIG["dof_num"]
    if len(yaml_config["kps"]) != dof_num or len(yaml_config["kds"]) != dof_num:
        raise ValueError(f"KP/KD维度必须为{dof_num}，当前KP: {len(yaml_config['kps'])}, KD: {len(yaml_config['kds'])}")
    if len(yaml_config["default_angles"]) != dof_num:
        raise ValueError(f"默认关节角度维度必须为{dof_num}，当前: {len(yaml_config['default_angles'])}")
    if len(yaml_config["cmd_scale"]) != 3:
        raise ValueError(f"指令缩放维度必须为3[vx, vy, yaw]，当前: {len(yaml_config['cmd_scale'])}")
    
    print("[SUCCESS] YAML配置加载与校验完成")
    return yaml_config

def load_onnx_policy(onnx_path: str, use_gpu: bool = True) -> tuple[ort.InferenceSession, str, tuple]:
    """加载ONNX策略模型，自动适配硬件，校验输入维度"""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX策略模型不存在: {onnx_path}")
    
    # 推理provider配置，优先GPU，兜底CPU
    providers = []
    if use_gpu:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    try:
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        raise RuntimeError(f"ONNX模型加载失败: {e}") from e

    # 输入输出信息提取
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    output_name = ort_session.get_outputs()[0].name

    # 打印硬件适配信息
    used_provider = ort_session.get_providers()[0]
    print(f"[SUCCESS] ONNX模型加载完成，使用推理后端: {used_provider}")
    print(f"[INFO] 模型输入维度: {input_shape}, 输入名: {input_name}, 输出名: {output_name}")
    
    return ort_session, input_name, output_name, input_shape

# ===================== 4. 键盘控制回调（全功能指令集）=====================
def create_keyboard_handlers(state: RobotState):
    """创建键盘事件回调，绑定机器人状态管理器"""
    # 基础速度映射（可根据机器人调试修改）
    BASE_VX_MAX = 0.8    # 前后最大速度 m/s
    BASE_VY_MAX = 0.4    # 横向最大速度 m/s
    BASE_YAW_MAX = 1.0   # 转向最大角速度 rad/s

    def on_press(key):
        if state.exit_flag:
            return False  # 退出监听
        
        speed_scale = state.speed_scale
        try:
            # 移动控制
            if key == keyboard.Key.up:
                state.set_cmd(0, BASE_VX_MAX * speed_scale)   # 前进
            elif key == keyboard.Key.down:
                state.set_cmd(0, -BASE_VX_MAX * speed_scale)  # 后退
            elif key == keyboard.Key.left:
                state.set_cmd(2, BASE_YAW_MAX * speed_scale)  # 左转
            elif key == keyboard.Key.right:
                state.set_cmd(2, -BASE_YAW_MAX * speed_scale) # 右转
            elif key == keyboard.Key.shift_l:
                state.set_cmd(1, BASE_VY_MAX * speed_scale)   # 左移
            elif key == keyboard.Key.ctrl_l:
                state.set_cmd(1, -BASE_VY_MAX * speed_scale)  # 右移
            
            # 功能控制
            elif key == keyboard.Key.space:
                state.toggle_pause()               # 空格暂停/继续
            elif key == keyboard.Key.r:
                state.trigger_reset()               # R键复位机器人
            elif key == keyboard.Key.esc:
                state.trigger_exit()                 # ESC键退出程序
            elif key == keyboard.Key.e:
                state.toggle_emergency_stop()        # E键急停/恢复
            elif key == keyboard.Key.equals:
                state.speed_up()                      # =/+ 键提速
            elif key == keyboard.Key.minus:
                state.speed_down()                    # - 键降速

        except AttributeError:
            pass

    def on_release(key):
        if state.exit_flag:
            return False  # 退出监听
        
        try:
            # 对应轴指令归零
            if key in (keyboard.Key.up, keyboard.Key.down):
                state.set_cmd(0, 0.0)
            elif key in (keyboard.Key.left, keyboard.Key.right):
                state.set_cmd(2, 0.0)
            elif key in (keyboard.Key.shift_l, keyboard.Key.ctrl_l):
                state.set_cmd(1, 0.0)
        except AttributeError:
            pass

    return on_press, on_release

def viewer_key_callback(keycode: int, state: RobotState):
    """MuJoCo Viewer内置键盘回调，兼容窗口聚焦时的按键输入"""
    if chr(keycode) == ' ':
        state.toggle_pause()
    elif chr(keycode) == 'r':
        state.trigger_reset()
    elif chr(keycode) == 'q':
        state.trigger_exit()
    elif chr(keycode) == 'e':
        state.toggle_emergency_stop()

# ===================== 5. 机器人复位函数 ======================
def reset_robot(model: mujoco.MjModel, data: mujoco.MjData, default_angles: np.ndarray):
    """复位机器人到初始姿态，清除速度，避免累积误差"""
    # 机身位姿复位
    data.qpos[:] = 0.0
    data.qpos[2] = 0.35  # 初始机身高度，避免触地
    data.qpos[3] = 1.0   # 四元数初始值 [w,x,y,z] 无旋转
    # 关节角度复位
    data.qpos[7:7+CONFIG["dof_num"]] = default_angles
    # 速度全清零
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    # 前向动力学计算，更新状态
    mujoco.mj_forward(model, data)
    print("[INFO] 机器人已复位到初始姿态")

# ===================== 6. 主仿真循环 ======================
def run_simulation():
    # 初始化机器人状态管理器
    robot_state = RobotState()
    on_press, on_release = create_keyboard_handlers(robot_state)

    # 启动键盘监听（守护线程，主程序退出自动销毁）
    keyboard_listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
        daemon=True
    )
    keyboard_listener.start()

    try:
        # 1. 加载配置与模型
        yaml_config = load_yaml_config(CONFIG["yaml_path"])
        mj_model = mujoco.MjModel.from_xml_path(CONFIG["mj_xml_path"])
        mj_data = mujoco.MjData(mj_model)
        
        # 2. 仿真参数配置（与训练严格对齐）
        sim_dt = yaml_config.get("sim_dt", CONFIG["sim_dt"])
        control_decimation = yaml_config.get("control_decimation", CONFIG["control_decimation"])
        mj_model.opt.timestep = sim_dt
        control_freq = 1 / (sim_dt * control_decimation)
        print(f"[INFO] 物理仿真频率: {1/sim_dt:.0f}Hz, 策略控制频率: {control_freq:.0f}Hz")

        # 3. 加载ONNX策略模型
        ort_session, input_name, output_name, input_shape = load_onnx_policy(
            CONFIG["onnx_policy_path"], use_gpu=True
        )

        # 4. 提取控制参数
        kps = np.array(yaml_config["kps"], dtype=np.float32)
        kds = np.array(yaml_config["kds"], dtype=np.float32)
        default_dof_pos = np.array(yaml_config["default_angles"], dtype=np.float32)
        tau_limit = np.array(yaml_config["tau_limit"], dtype=np.float32)
        
        # 归一化缩放参数
        lin_vel_scale = yaml_config["lin_vel_scale"]
        ang_vel_scale = yaml_config["ang_vel_scale"]
        dof_pos_scale = yaml_config["dof_pos_scale"]
        dof_vel_scale = yaml_config["dof_vel_scale"]
        action_scale = yaml_config["action_scale"]
        cmd_scale = np.array(yaml_config["cmd_scale"], dtype=np.float32)
        clip_obs = yaml_config.get("clip_observations", 100.0)
        clip_action = yaml_config.get("clip_actions", 10.0)

        # 5. 状态变量初始化
        reset_robot(mj_model, mj_data, default_dof_pos)
        step_counter = 0
        last_action = np.zeros(CONFIG["dof_num"], dtype=np.float32)
        smoothed_action = np.zeros(CONFIG["dof_num"], dtype=np.float32)
        
        # 历史观测队列初始化（与训练严格对齐）
        history_len = CONFIG["history_len"]
        obs_dim = CONFIG["obs_dim"]
        obs_history_buffer = deque(
            [np.zeros(obs_dim, dtype=np.float32) for _ in range(history_len)],
            maxlen=history_len
        )

        # 关节安全限位（从模型中读取，避免超行程）
        dof_pos_lower = mj_model.jnt_range[1:, 0]  # 跳过底座自由关节
        dof_pos_upper = mj_model.jnt_range[1:, 1]
        print(f"[INFO] 关节限位加载完成，共{CONFIG['dof_num']}个关节")

        # 6. 启动MuJoCo Viewer
        with mujoco.viewer.launch_passive(
            mj_model, mj_data,
            key_callback=lambda keycode: viewer_key_callback(keycode, robot_state)
        ) as viewer:
            viewer.cam.distance = 3.0  # 初始视角距离
            viewer.cam.elevation = -20.0  # 初始视角俯仰
            print("\n" + "="*50)
            print("✅ 仿真启动完成！控制指令说明：")
            print("方向键 ↑↓：前进/后退 | ←→：左转/右转")
            print("左Shift/Ctrl：左移/右移 | 空格：暂停/继续")
            print("R键：复位机器人 | E键：紧急停止 | ESC键：退出程序")
            print("+/-键：调节速度档位")
            print("="*50 + "\n")

            # 主循环
            while viewer.is_running() and not robot_state.exit_flag:
                loop_start_time = time.time()

                # 复位检测
                if robot_state.reset_flag:
                    reset_robot(mj_model, mj_data, default_dof_pos)
                    last_action = np.zeros_like(last_action)
                    smoothed_action = np.zeros_like(smoothed_action)
                    obs_history_buffer = deque(
                        [np.zeros(obs_dim, dtype=np.float32) for _ in range(history_len)],
                        maxlen=history_len
                    )
                    step_counter = 0

                # 急停处理
                if robot_state.emergency_stop:
                    data.ctrl[:] = 0.0
                    viewer.sync()
                    time.sleep(0.001)
                    continue

                # 暂停处理
                if not robot_state.paused:
                    # ===================== 策略推理（按分频执行）=====================
                    if step_counter % control_decimation == 0:
                        # 1. 提取机器人状态（与训练观测源严格对齐）
                        current_qpos = mj_data.qpos.copy()
                        current_qvel = mj_data.qvel.copy()
                        
                        base_quat = current_qpos[3:7]          # 机身处四元数 [w,x,y,z]
                        base_omega = current_qvel[3:6]          # 机身角速度
                        dof_pos = current_qpos[7:7+CONFIG["dof_num"]]  # 关节位置
                        dof_vel = current_qvel[6:6+CONFIG["dof_num"]]  # 关节速度
                        
                        # 2. 重力向量投影（机体坐标系）
                        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                        gravity_body = quat_rotate_inverse(base_quat, gravity_world)

                        # 3. 指令死区与归一化
                        raw_cmd = robot_state.cmd
                        raw_cmd[np.abs(raw_cmd) < CONFIG["cmd_dead_zone"]] = 0.0
                        cmd_norm = raw_cmd * cmd_scale

                        # 4. 状态归一化（与训练严格对齐）
                        omega_norm = base_omega * ang_vel_scale
                        dof_pos_norm = (dof_pos - default_dof_pos) * dof_pos_scale
                        dof_vel_norm = dof_vel * dof_vel_scale

                        # 5. 构建单帧观测向量（顺序必须与训练100%一致！）
                        current_obs = np.concatenate([
                            omega_norm,        # 0-2: 机体角速度
                            gravity_body,      # 3-5: 机体坐标系重力向量
                            cmd_norm,          # 6-8: 归一化控制指令
                            dof_pos_norm,      # 9-20: 归一化关节位置
                            dof_vel_norm,      # 21-32: 归一化关节速度
                            last_action        # 33-44: 上一帧动作
                        ]).astype(np.float32)

                        # 6. 观测裁剪（防止超出训练分布）
                        current_obs = np.clip(current_obs, -clip_obs, clip_obs)

                        # 7. 历史观测队列更新（Legged Gym标准：最新帧在队首）
                        obs_history_buffer.appendleft(current_obs)

                        # 8. 构建模型输入（适配你的ONNX模型维度）
                        # --- 情况1：模型输入为单帧45维 ---
                        if input_shape[-1] == obs_dim:
                            model_input = current_obs.reshape(1, -1)
                        # --- 情况2：模型输入为历史帧拼接 6*45=270维 ---
                        elif input_shape[-1] == history_len * obs_dim:
                            model_input = np.concatenate(obs_history_buffer).reshape(1, -1)
                        # --- 情况3：模型输入为45维观测+19维隐变量=64维 ---
                        elif input_shape[-1] == 64:
                            model_input = np.zeros((1, 64), dtype=np.float32)
                            model_input[0, :45] = current_obs
                        else:
                            raise ValueError(f"不支持的模型输入维度: {input_shape}，请检查CONFIG中obs_dim/history_len配置")

                        # 9. ONNX模型推理
                        inference_start = time.time()
                        model_output = ort_session.run([output_name], {input_name: model_input})
                        raw_action = model_output[0][0].astype(np.float32)
                        inference_time = (time.time() - inference_start) * 1000

                        # 10. 动作处理与裁剪
                        raw_action = np.clip(raw_action, -clip_action, clip_action)
                        # 一阶低通滤波，平滑动作，减少关节抖动
                        smoothed_action = CONFIG["action_smooth_alpha"] * raw_action + (1 - CONFIG["action_smooth_alpha"]) * smoothed_action
                        # 计算目标关节位置
                        target_dof_pos = smoothed_action * action_scale + default_dof_pos
                        # 关节限位保护，防止超出行程
                        target_dof_pos = np.clip(target_dof_pos, dof_pos_lower, dof_pos_upper)
                        # 更新上一帧动作
                        last_action = smoothed_action.copy()

                        # 调试打印（可选，频率过高可注释）
                        if step_counter % (control_decimation * 100) == 0:
                            print(f"[DEBUG] 步长: {step_counter} | 推理耗时: {inference_time:.2f}ms | 指令: {raw_cmd} | 机身高度: {current_qpos[2]:.3f}m")

                    # ===================== 底层PD控制与物理步长 =====================
                    # 计算PD控制力矩
                    tau = pd_control(
                        target_q=target_dof_pos,
                        current_q=dof_pos,
                        kp=kps,
                        target_dq=np.zeros_like(dof_vel),
                        current_dq=dof_vel,
                        kd=kd
                    )
                    # 力矩限幅，安全保护
                    tau = np.clip(tau, -tau_limit, tau_limit)
                    mj_data.ctrl[:] = tau

                    # 物理步进
                    mujoco.mj_step(mj_model, mj_data)
                    step_counter += 1

                    # 摔倒检测与自动复位
                    if mj_data.qpos[2] < CONFIG["fall_threshold"] and step_counter > 1000:
                        print("[WARN] 检测到机器人摔倒，自动复位")
                        reset_robot(mj_model, mj_data, default_dof_pos)
                        last_action = np.zeros_like(last_action)
                        smoothed_action = np.zeros_like(smoothed_action)
                        obs_history_buffer = deque(
                            [np.zeros(obs_dim, dtype=np.float32) for _ in range(history_len)],
                            maxlen=history_len
                        )
                        step_counter = 0

                # Viewer画面同步
                viewer.sync()

                # 精准帧率控制，保证仿真实时性
                loop_time = time.time() - loop_start_time
                sleep_time = mj_model.opt.timestep - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except Exception as e:
        print(f"[ERROR] 仿真运行异常: {e}")
        raise
    finally:
        # 资源释放，避免僵尸线程/进程
        if keyboard_listener.is_alive():
            keyboard_listener.stop()
        mujoco.mj_resetData(mj_model, mj_data)
        print("[INFO] 程序退出，资源已释放")

# ===================== 程序入口 ======================
if __name__ == "__main__":
    # 屏蔽ONNX冗余日志
    os.environ['ORT_LOG_LEVEL'] = 'ERROR'
    run_simulation()

