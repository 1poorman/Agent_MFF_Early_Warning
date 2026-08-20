"""数据生成器：驱动物理模型按 1Hz 输出带时间戳的传感器记录。"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PhaseSpec, SimConfig, default_phase_schedule
from .faults import FaultScheduler, FaultSpec
from .physics import PlantModel

# 传感器测量噪声标准差（叠加在真实状态上，不影响状态演化）
# 传感器测量噪声标准差（叠加在真实状态上，不影响状态演化）
SENSOR_NOISE = {
    "inlet_temp": 0.05,
    "outlet_temp": 0.05,
    "flow_rate": 0.03,
    "flow_velocity": 0.01,
    "conductivity": 1.0,
    "cabinet_temp": 0.1,
    "cabinet_humidity": 0.3,
    "furnace_temp": 0.5,
    "electric_power": 5.0,
    "electric_current": 3.0,
}

# 单位换算说明（题目要求 1 位小数，MPa/m 量级下分辨率不足，故输出采用 kPa/cm）：
#   pressure:   MPa -> kPa（正常 150~300 kPa）
#   tank_level: m   -> cm （正常 190~210 cm）
UNIT_SCALE = {"pressure": 1000.0, "tank_level": 100.0}
UNIT_NOISE = {"pressure": 0.5, "tank_level": 0.5}

COLUMNS = [
    "timestamp",
    "inlet_temp", "outlet_temp", "pressure", "flow_rate", "flow_velocity",
    "tank_level", "conductivity",
    "cabinet_temp", "cabinet_humidity",
    "furnace_temp", "electric_power", "electric_current",
    "operating_condition", "fault_label",
]


class DataSimulator:
    """中频炉水冷系统数据模拟智能体。"""

    def __init__(
        self,
        config: Optional[SimConfig] = None,
        schedule: Optional[List[PhaseSpec]] = None,
        faults: Optional[List[FaultSpec]] = None,
    ):
        self.cfg = config or SimConfig()
        self.plant = PlantModel(self.cfg)
        self.schedule = schedule or default_phase_schedule()
        self.faults = FaultScheduler(faults or [])
        self.rng = np.random.default_rng(self.cfg.seed)
        self._phase_bounds = self._build_phase_bounds()
        self.t = 0.0

    # ---------------- 工况调度 ----------------

    def _build_phase_bounds(self) -> List[Tuple[float, str, float]]:
        bounds, acc = [], 0.0
        for name, dur, target in self.schedule:
            bounds.append((acc, name, target))
            acc += dur
        self.cycle_len = acc
        return bounds

    def phase_at(self, t: float) -> Tuple[str, float]:
        pos = t % self.cycle_len
        for start, name, target in reversed(self._phase_bounds):
            if pos >= start:
                return name, target
        return self._phase_bounds[0][1], self._phase_bounds[0][2]

    # ---------------- 单步仿真 ----------------

    def step(self) -> Dict:
        cfg, plant = self.cfg, self.plant
        plant.t = self.t
        dt = cfg.dt

        phase, target = self.phase_at(self.t)
        power = plant.power_command(target, phase)
        plant.step_furnace(power, phase, dt)

        effects = self.faults.effects(self.t)
        hyd = plant.step_hydraulics(effects, dt)
        t_in = plant.inlet_temp
        t_out = plant.outlet_temp(power, hyd["flow"])

        plant.step_tank(effects, dt)
        plant.step_conductivity(dt)

        labels = self.faults.active_labels(self.t)
        row = {
            "inlet_temp": t_in,
            "outlet_temp": t_out,
            "pressure": hyd["pressure"],
            "flow_rate": hyd["flow"],
            "flow_velocity": hyd["velocity"],
            "tank_level": plant.state.tank_level,
            "conductivity": plant.state.conductivity,
            "cabinet_temp": plant.cabinet_temp(power),
            "cabinet_humidity": plant.cabinet_humidity(effects),
            "furnace_temp": plant.state.furnace_temp,
            "electric_power": power,
            "electric_current": plant.electric_current(power),
            "operating_condition": phase,
            "fault_label": "+".join(labels) if labels else "none",
        }
        self.t += dt
        return row

    # ---------------- 批量生成 ----------------

    def run(self, duration: float) -> pd.DataFrame:
        n = int(round(duration / self.cfg.dt))
        rows: List[Dict] = []
        for _ in range(n):
            row = self.step()
            rows.append(self._sense(row))
        df = pd.DataFrame(rows, columns=COLUMNS)
        df["timestamp"] = pd.date_range(
            start=pd.Timestamp(self.cfg.start_time), periods=n, freq=f"{int(self.cfg.dt)}s"
        ).strftime("%Y-%m-%d %H:%M:%S")
        return df

    def _sense(self, row: Dict) -> Dict:
        """状态量 -> 传感器读数：叠加测量噪声并保留 1 位小数。

        电流读数由加噪后的功率读数按 I=P·1000/(√3·U·cosφ) 推导，
        保证功率与电流两条通道的测量值物理一致。
        """
        out = dict(row)
        for key, sigma in SENSOR_NOISE.items():
            out[key] = round(float(row[key] + self.rng.normal(0.0, sigma)), 1)
        for key, scale in UNIT_SCALE.items():
            out[key] = round(float(row[key] * scale + self.rng.normal(0.0, UNIT_NOISE[key])), 1)
        # 功率/电流物理约束：不能为负（停机工况允许为 0）
        out["electric_power"] = max(out["electric_power"], 0.0)
        if out["electric_power"] > 0:
            current = out["electric_power"] * 1000.0 / (
                np.sqrt(3) * self.cfg.line_voltage * self.cfg.power_factor
            )
            out["electric_current"] = round(max(current + self.rng.normal(0.0, 1.0), 0.0), 1)
        else:
            out["electric_current"] = 0.0
        return out

    def run_to_file(self, path: str, duration: float, fmt: str = "csv") -> str:
        df = self.run(duration)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "csv":
            df.to_csv(out, index=False)
        elif fmt == "parquet":
            df.to_parquet(out, index=False)
        else:
            raise ValueError(f"不支持的输出格式: {fmt}，可选 csv/parquet")
        return str(out)
