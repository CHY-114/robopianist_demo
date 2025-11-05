import numpy as np
from typing import Dict, Tuple, List


'''
Damped Least Squares (DLS) IK for 4-DOF drum stick manipulator.
'''

# ---------- 旋转与齐次变换 ----------
def RotZ(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c,-s,0],
                     [s, c,0],
                     [0, 0,1]], dtype=float)

def RotY(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[ c,0, s],
                     [ 0,1, 0],
                     [-s,0, c]], dtype=float)

def T_from(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

# ---------- 前向运动学：给 q -> 末端（stick_tip）世界坐标 ----------
def fk_tip(q: np.ndarray,
           base_pos: np.ndarray,
           cfg: Dict[str, float]) -> np.ndarray:
    """q=[q1,q2,q3,q4]，轴序 Z-Y-Y-Y；返回 stick_tip 世界坐标 (3,)."""
    d1 = cfg["d1"]
    L_upper = cfg["L_upper"]
    L_fore  = cfg["L_fore"]
    L_wrist = cfg["L_wrist"]
    stick_length = cfg.get("stick_length", 0.35)

    # 基座 -> J1 位置：base_pos + [0,0,d1]
    T = T_from(np.eye(3), np.asarray(base_pos, float)) @ T_from(np.eye(3), np.array([0,0,d1]))

    # J1: yaw about Z
    T = T @ T_from(RotZ(q[0]), np.zeros(3))

    # J2: pitch about Y, 再沿 +Z 走 L_upper
    T = T @ T_from(RotY(q[1]), np.zeros(3)) @ T_from(np.eye(3), np.array([0,0,L_upper]))

    # J3: pitch about Y, 再沿 +Z 走 L_fore
    T = T @ T_from(RotY(q[2]), np.zeros(3)) @ T_from(np.eye(3), np.array([0,0,L_fore]))

    # J4: pitch about Y（与腕段平行），再沿 +Z 走 L_wrist
    T = T @ T_from(RotY(q[3]), np.zeros(3)) @ T_from(np.eye(3), np.array([0,0,L_wrist]))

    # 鼓槌沿本地 +Z 方向 stick_length
    tip_h = T @ np.array([0, 0, stick_length, 1.0])
    return tip_h[:3]

# ---------- 数值雅可比（末端位置对关节）的有限差分 ----------
def jacobian_numeric(q, base_pos, cfg, eps=1e-6):
    J = np.zeros((3,4))
    p0 = fk_tip(q, base_pos, cfg)
    for i in range(4):
        dq = q.copy()
        dq[i] += eps
        pi = fk_tip(dq, base_pos, cfg)
        J[:, i] = (pi - p0) / eps
    return J

# ---------- 阻尼最小二乘 IK（只做位置） ----------
def ik_dls(q0: np.ndarray,
           target_pos: np.ndarray,
           base_pos: np.ndarray,
           cfg: Dict[str, float],
           limits: Dict[str, Tuple[float,float]],
           max_iters: int = 200,
           tol: float = 1e-4,
           damping: float = 1e-3,
           alpha: float = 1.0) -> np.ndarray:
    """返回满足末端位置的 q*（若不可达则返回最近解）"""
    q = q0.copy()
    for _ in range(max_iters):
        p = fk_tip(q, base_pos, cfg)
        e = target_pos - p                         # 位置误差 (3,)
        if np.linalg.norm(e) < tol:
            break
        J = jacobian_numeric(q, base_pos, cfg)     # (3,4)
        # DLS: dq = J^T (J J^T + λ^2 I)^-1 e
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (damping**2)*np.eye(3), e)
        q += alpha * dq

        # 关节限幅
        q[0] = np.clip(q[0], *limits["j1"])
        q[1] = np.clip(q[1], *limits["j2"])
        q[2] = np.clip(q[2], *limits["j3"])
        q[3] = np.clip(q[3], *limits["j4"])
    return q

# ---------- 五次多项式（最小 jerk）插值 ----------
def min_jerk_timeseries(q0: np.ndarray, qT: np.ndarray, T: float, n: int = 21):
    """返回等间隔 keyframe 时间与关节角值（n>=2）。"""
    t = np.linspace(0.0, T, n)
    s = (t / T) if T > 0 else np.zeros_like(t)  # 归一化时间
    # 10 s^3 - 15 s^4 + 6 s^5
    S = 10*s**3 - 15*s**4 + 6*s**5
    Q = q0[None, :] + (qT - q0)[None, :] * S[:, None]   # (n,4)
    return t.tolist(), Q.tolist()

# ---------- 总函数 ----------
def plan_strike_trajectory(initial_pos: List[float],
                           initial_configs: List[float],
                           output_pos: List[float],
                           time_duration: float,
                           arm_config: Dict) -> Tuple[List[float], List[List[float]]]:
    """
    input:
      - initial_pos: 机械臂基座 world 坐标 (x,y,z) —— 你的 arm_base
      - initial_configs: 初始 4 关节角 [J1,J2,J3,J4] (rad)
      - output_pos: 期望 stick_tip 的 world 坐标 (x,y,z)
      - time_duration: 轨迹总时长 (s)
      - arm_config: 字典，需含：
            d1, L_upper, L_fore, L_wrist, stick_length(可选),
            j1_range, j2_range, j3_range, j4_range
    output:
      - keyframe_times: List[float], 长度 n（默认 21 个关键帧，等间隔）
      - keyframe_values: List[List[float]]，形状 (n,4)，每行是 4 关节角 (rad)
    """
    base_pos = np.array(initial_pos, float)
    q0 = np.array(initial_configs, float)
    target = np.array(output_pos, float)

    cfg = {
        "d1": arm_config["d1"],
        "L_upper": arm_config["L_upper"],
        "L_fore": arm_config["L_fore"],
        "L_wrist": arm_config["L_wrist"],
        "stick_length": arm_config.get("stick_length", 0.35),
    }
    limits = {
        "j1": tuple(arm_config["j1_range"]),
        "j2": tuple(arm_config["j2_range"]),
        "j3": tuple(arm_config["j3_range"]),
        "j4": tuple(arm_config["j4_range"]),
    }

    # 逆解：DLS IK
    qT = ik_dls(q0, target, base_pos, cfg, limits,
                max_iters=200, tol=1e-4, damping=1e-3, alpha=1.0)

    # 关键帧（minimum-jerk）
    keyframe_times, keyframe_values = min_jerk_timeseries(q0, qT, time_duration, n=11)

    # 最后再做一次限幅（数值稳定）
    j1_lo, j1_hi = limits["j1"]
    j2_lo, j2_hi = limits["j2"]
    j3_lo, j3_hi = limits["j3"]
    j4_lo, j4_hi = limits["j4"]
    for i in range(len(keyframe_values)):
        q = np.array(keyframe_values[i])
        q[0] = float(np.clip(q[0], j1_lo, j1_hi))
        q[1] = float(np.clip(q[1], j2_lo, j2_hi))
        q[2] = float(np.clip(q[2], j3_lo, j3_hi))
        q[3] = float(np.clip(q[3], j4_lo, j4_hi))
        keyframe_values[i] = q.tolist()

    return keyframe_times, np.array(keyframe_values)

if __name__ == "__main__":
    # 目标点（任选其一）
    kick_top = [0.6, 0.0, 0.225 + 0.45]     # 你给的
    ride_pos = [0.95, 0.7, 1.15]

    arm_base = [1.2, 0.2, 0.35]
    q_init = [-0.3, 0.4, 0.6, 0.0]          # Z-Y-Y-Y
    T = 0.60                                 # 0.6 s 完成

    arm_cfg = dict(
        d1=0.267, L_upper=0.293, L_fore=0.3425, L_wrist=0.097,
        stick_length=0.35,
        j1_range=(-2*np.pi, 2*np.pi),
        j2_range=(np.deg2rad(-118), np.deg2rad(120)),
        j3_range=(np.deg2rad(-11),  np.deg2rad(225)),
        j4_range=(-np.pi, np.pi),
    )

    # 规划到 kick_top
    times, values = plan_strike_trajectory(arm_base, q_init, kick_top, T, arm_cfg)

    # 规划到 ride
    times2, values2 = plan_strike_trajectory(arm_base, values[-1], ride_pos, 0.8, arm_cfg)

    # 保留2位小数
    times = [round(t, 2) for t in times]
    values = [[round(qi, 4) for qi in q] for q in values]
    times2 = [round(t, 2) for t in times2]
    values2 = [[round(qi, 4) for qi in q] for q in values2]

    print("To kick top:")
    print("Keyframe times (s)", times)
    print("Keyframe values (rad):", values)

    print("\nTo ride cymbal:")
    print("Keyframe times (s)", times2)
    print("Keyframe values (rad):", values2)

    total_times = times + [t + times[-1] for t in times2[1:]]
    total_values = values + values2[1:]
    print("\nTotal trajectory:")
    print("Keyframe times (s):", total_times)
    print("Keyframe values (rad):", total_values)
