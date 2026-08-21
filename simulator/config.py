"""模拟器全局配置与默认工况循环。

所有参数按 5 吨中频炉 + 配套闭式循环水冷系统的典型量级整定，
保证各物理量之间的关系满足热平衡 / 水力特性 / 电气公式。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SimConfig:
    # ---- 仿真基础 ----
    dt: float = 1.0                       # 采样步长（秒），题目要求每秒 1 条
    seed: Optional[int] = 42              # 随机种子，保证可复现
    start_time: str = "2026-08-20 00:00:00"

    # ---- 炉体热模型 ----
    furnace_heat_capacity: float = 4.2e3  # 有效热容 C (kJ/K)，约 5t 钢水
    furnace_eta: float = 0.72             # 电-热转换效率
    furnace_loss_coef: float = 0.12       # 炉体表面散热系数 h (kW/K)
    ambient_temp: float = 30.0            # 环境温度 (℃)
    rated_power: float = 3000.0           # 额定功率 (kW)
    power_band: float = 50.0              # 功率比例调节带宽 (K)

    # ---- 电气参数（P = √3·U·I·cosφ）----
    line_voltage: float = 1000.0          # 线电压 (V)
    power_factor: float = 0.92            # 功率因数

    # ---- 水冷系统水力模型 ----
    pump_head: float = 0.30               # 水泵出口扬程对应压力 (MPa)
    static_pressure: float = 0.12         # 管网静压 (MPa)
    coil_resistance: float = 0.0022       # 线圈管路阻抗 R (MPa/(L/s)^2)
    filter_resistance: float = 0.0006     # 过滤器洁净时阻抗 (MPa/(L/s)^2)
    flow_tau: float = 8.0                 # 流量一阶惯性时间常数 (s)
    pipe_area: float = 3.318e-3           # 管道截面积 (m^2)，DN65
    rated_flow: float = 8.0               # 额定流量 (L/s)
    max_delta_t: float = 60.0             # 断流保护：进出水温差上限 (℃)

    # ---- 冷却水热模型 ----
    inlet_temp_base: float = 28.0         # 进水温度基线 (℃，夏季)
    inlet_temp_drift: float = 2.0         # 进水温度昼夜波动幅值 (℃)
    coil_loss_frac: float = 0.08          # 电功率进入冷却水的比例（线圈损耗）
    furnace_water_coef: float = 0.10      # 炉体向冷却水传热系数 (kW/K)
    water_cp: float = 4.186               # 水的比热容 (kJ/(kg·K))

    # ---- 水箱 ----
    tank_area: float = 4.0                # 水箱截面积 (m^2)
    tank_level_init: float = 2.0          # 初始液位 (m)
    tank_level_low: float = 1.9           # 补水阀开启液位 (m)
    tank_level_high: float = 2.1          # 补水阀关闭液位 (m)
    refill_rate: float = 8.0              # 补水量 (L/s)

    # ---- 水质 / 电气柜 ----
    conductivity_init: float = 550.0      # 初始电导率 (µS/cm)
    conductivity_drift: float = 0.5       # 自然漂移 (µS/cm 每小时)
    cabinet_temp_coef: float = 0.004      # 电气柜温升系数 (K/kW)
    humidity_base: float = 50.0           # 湿度基线 (%RH)
    humidity_daily_amp: float = 8.0       # 湿度昼夜波动 (%RH)

    @classmethod
    def from_settings(cls) -> "SimConfig":
        """从集中配置（config/settings.yaml simulator 段）构造。"""
        from config import get_settings
        return get_settings().to_sim_config()


# 工况相位: (名称, 持续秒数, 炉温目标℃)
PhaseSpec = Tuple[str, int, float]


def default_phase_schedule() -> List[PhaseSpec]:
    """默认熔炼循环：冷炉启动 -> 熔炼 -> 保温 -> 出炉 -> 待机，循环往复。"""
    return [
        ("startup", 3600, 1650.0),   # 熔炼启动：满功率升温
        ("melting", 1800, 1650.0),   # 熔炼：保温大功率
        ("holding", 1800, 1550.0),   # 保温：低功率维持
        ("tapping", 600, 900.0),     # 出炉：断电，钢水带出热量
        ("idle", 600, 35.0),         # 待机/加冷料
    ]
