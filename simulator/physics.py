"""物理机理模型：炉体热过程 + 水冷系统水力/热平衡 + 辅助系统。

每个 dt=1s 步进一次，状态量连续演化，传感器读数在状态量上叠加测量噪声，
保证：
1. 时间戳连续递增、无丢包/乱序/空值；
2. 各物理量之间严格满足物理约束（热平衡、管网特性、电气公式）。
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .config import SimConfig


@dataclass
class PlantState:
    """被仿真的物理状态量（非传感器读数）。"""
    furnace_temp: float = 35.0        # 炉内温度 ℃
    flow: float = 8.0                 # 冷却水流量 L/s
    tank_level: float = 2.0           # 水箱液位 m
    conductivity: float = 550.0       # 电导率 µS/cm
    refilling: bool = False           # 补水阀状态


class PlantModel:
    """中频炉 + 水冷系统机理模型。"""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.t: float = 0.0  # 当前仿真秒，由 generator 注入
        self.state = PlantState(
            furnace_temp=cfg.ambient_temp + 5.0,
            flow=cfg.rated_flow,
            tank_level=cfg.tank_level_init,
            conductivity=cfg.conductivity_init,
        )

    # ---------------- 炉体 ----------------

    def power_command(self, target_temp: float, phase: str) -> float:
        """比例功率调节：远离目标满功率，接近目标按比例降功率。"""
        cfg = self.cfg
        if phase in ("idle", "tapping"):
            return 0.0
        ratio = np.clip((target_temp - self.state.furnace_temp) / cfg.power_band, 0.0, 1.0)
        return float(cfg.rated_power * ratio)

    def step_furnace(self, power: float, phase: str, dt: float) -> None:
        """C·dT/dt = P·η - h·(T - T_amb)；出炉时钢水带走大量热量。"""
        cfg = self.cfg
        st = self.state
        loss = cfg.furnace_loss_coef * (st.furnace_temp - cfg.ambient_temp)
        dT = (power * cfg.furnace_eta - loss) / cfg.furnace_heat_capacity * dt
        st.furnace_temp += dT
        if phase == "tapping":
            # 出钢带走热量：等效附加散热
            st.furnace_temp -= (st.furnace_temp - 900.0) * 0.002 * dt
        if phase == "idle":
            # 加入冷料：炉温向环境温度回落加速
            st.furnace_temp -= (st.furnace_temp - cfg.ambient_temp) * 0.003 * dt

    # ---------------- 水冷系统 ----------------

    def step_hydraulics(self, effects: Dict[str, float], dt: float) -> Dict[str, float]:
        """管网水力模型。

        Q_ss = sqrt((P_pump - P_static) / R_total)
        P_pipe = P_static + R_coil·Q²
        故障通过 effects 修改阻抗/泵压/泄漏量。
        """
        cfg = self.cfg
        st = self.state

        pump_head = cfg.pump_head + effects.get("pump_head_delta", 0.0)
        r_total = (
            cfg.coil_resistance * effects.get("r_coil_mult", 1.0)
            + cfg.filter_resistance * effects.get("r_filter_mult", 1.0)
        )
        q_ss = np.sqrt(max(pump_head - cfg.static_pressure, 0.0) / r_total)
        # 泄漏分流：主回路可用流量下降
        q_ss = max(q_ss - effects.get("leak_flow", 0.0), 0.0)

        # 一阶惯性趋近稳态流量
        st.flow += (q_ss - st.flow) / cfg.flow_tau * dt
        st.flow = max(st.flow, 0.0)

        # 气蚀：泵压振荡传导到流量
        osc = effects.get("pump_osc_amp", 0.0)
        if osc > 0.0:
            # 振荡频率 0.17Hz（周期~6s）：避免与 1Hz 采样混叠（0.5Hz 时 sin(πn)≡0）
            st.flow += osc / r_total * 0.5 * np.sin(2 * np.pi * 0.17 * effects["t_rel"])
            st.flow = max(st.flow, 0.0)

        pressure = cfg.static_pressure + cfg.coil_resistance * effects.get("r_coil_mult", 1.0) * st.flow ** 2
        if osc > 0.0:
            pressure += osc * np.sin(2 * np.pi * 0.17 * effects["t_rel"])
        pressure = max(pressure, 0.0)

        velocity = st.flow * 1e-3 / cfg.pipe_area  # L/s -> m/s
        return {"flow": st.flow, "pressure": pressure, "velocity": velocity}

    def outlet_temp(self, power: float, flow: float) -> float:
        """冷却水热平衡: T_out = T_in + Q_heat / (c·ṁ)。流量过低时温差封顶。"""
        cfg = self.cfg
        t_in = self.inlet_temp
        q_heat = cfg.coil_loss_frac * power + cfg.furnace_water_coef * (
            self.state.furnace_temp - t_in
        )
        q_heat = max(q_heat, 0.0)
        if flow < 0.3:
            return t_in + cfg.max_delta_t
        delta = q_heat / (cfg.water_cp * flow)
        return t_in + min(delta, cfg.max_delta_t)

    # ---------------- 辅助系统 ----------------

    @property
    def inlet_temp(self) -> float:
        """进水温度：基线 + 昼夜缓变（闭式冷却塔受环境温度影响）。"""
        cfg = self.cfg
        day_phase = 2 * np.pi * (self.t % 86400) / 86400
        return cfg.inlet_temp_base + cfg.inlet_temp_drift * np.sin(day_phase - np.pi / 2)

    def step_tank(self, effects: Dict[str, float], dt: float) -> None:
        """水箱质量平衡：泄漏损失 + 补水阀滞环控制。"""
        cfg = self.cfg
        st = self.state
        leak = effects.get("leak_flow", 0.0)
        if st.tank_level < cfg.tank_level_low:
            st.refilling = True
        elif st.tank_level > cfg.tank_level_high:
            st.refilling = False
        inflow = cfg.refill_rate if st.refilling else 0.0
        dv = (inflow - leak - 0.02) * 1e-3 * dt   # L/s -> m³/s，0.02L/s 蒸发飞溅
        st.tank_level = max(st.tank_level + dv / cfg.tank_area, 0.0)

    def step_conductivity(self, dt: float) -> None:
        """电导率：浓缩缓慢上升；补水稀释向 300 µS/cm 回落。"""
        cfg = self.cfg
        st = self.state
        st.conductivity += cfg.conductivity_drift / 3600.0 * dt
        if st.refilling:
            st.conductivity += (300.0 - st.conductivity) * 0.002 * dt

    def cabinet_temp(self, power: float) -> float:
        return self.cfg.ambient_temp + self.cfg.cabinet_temp_coef * power

    def cabinet_humidity(self, effects: Dict[str, float]) -> float:
        cfg = self.cfg
        day_phase = 2 * np.pi * (self.t % 86400) / 86400
        hum = cfg.humidity_base + cfg.humidity_daily_amp * np.sin(day_phase + np.pi / 3)
        hum += effects.get("humidity_add", 0.0)
        return float(np.clip(hum, 20.0, 100.0))

    def electric_current(self, power: float) -> float:
        """I = P·1000 / (√3·U·cosφ)，功率为 0 时电流为 0。"""
        cfg = self.cfg
        if power <= 0.0:
            return 0.0
        return power * 1000.0 / (np.sqrt(3) * cfg.line_voltage * cfg.power_factor)
