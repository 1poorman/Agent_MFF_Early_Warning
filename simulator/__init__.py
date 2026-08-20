"""中频炉水冷系统数据模拟智能体。

基于物理机理模型生成贴近真实的传感器时序数据：
- 炉体热平衡:  C·dT/dt = P·η - h·(T - T_amb)
- 冷却水热平衡: T_out = T_in + Q_heat / (c · ṁ)
- 管网水力:    P_pipe = P_static + R·Q²,  Q_ss = sqrt((P_pump - P_static) / R_total)
- 三相电:      P = √3 · U · I · cosφ
"""

from .config import SimConfig, default_phase_schedule
from .generator import DataSimulator
from .faults import FaultSpec, parse_fault_spec

__all__ = [
    "SimConfig",
    "default_phase_schedule",
    "DataSimulator",
    "FaultSpec",
    "parse_fault_spec",
]
