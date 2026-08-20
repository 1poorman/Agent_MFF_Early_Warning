"""故障注入模块：模拟中频炉水冷系统典型劣化/故障模式。

故障通过 effects 字典作用于物理模型，保证异常数据依然满足物理规律：
- filter_clog    过滤器堵塞：过滤器阻抗上升 -> 流量衰减 -> 出水温度按热平衡升高
- pump_cavitation 水泵气蚀：泵压均值下滑 + 压力/流量低频振荡
- pipe_leak      管道泄漏：泄漏分流 -> 压力/流量下降、水箱液位下降、局部湿度上升
- scale_buildup  水垢累积：线圈管路阻抗缓升 -> 流量缓降、出水温度缓慢爬升（缓变型）
"""

import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FaultSpec:
    """故障配置。

    name:     故障类型（filter_clog / pump_cavitation / pipe_leak / scale_buildup）
    start:    故障起始时刻（相对仿真开始的秒数）
    ramp:     劣化爬升时长（秒），期间故障程度从 0 线性爬到 severity
    severity: 故障严重程度 0~1
    duration: 持续秒数，None 表示持续到仿真结束
    """
    name: str
    start: float
    ramp: float = 600.0
    severity: float = 1.0
    duration: float = None

    def progress(self, t: float) -> float:
        """返回 t 时刻的故障程度 0~severity；未开始或已结束返回 0。"""
        if t < self.start:
            return 0.0
        if self.duration is not None and t > self.start + self.duration:
            return 0.0
        rel = t - self.start
        if self.ramp > 0:
            return self.severity * min(rel / self.ramp, 1.0)
        return self.severity


def _apply_filter_clog(spec: FaultSpec, prog: float, t_rel: float, eff: Dict[str, float]) -> None:
    # 过滤器阻抗最高升至 1 + 15·severity 倍
    eff["r_filter_mult"] = eff.get("r_filter_mult", 1.0) * (1.0 + 15.0 * prog)


def _apply_pump_cavitation(spec: FaultSpec, prog: float, t_rel: float, eff: Dict[str, float]) -> None:
    eff["pump_head_delta"] = eff.get("pump_head_delta", 0.0) - 0.04 * prog
    # 振荡幅度整定：0.004·prog MPa 使压力波动 std 约 6~12kPa（去趋势后可稳定检出，不至失真）
    eff["pump_osc_amp"] = eff.get("pump_osc_amp", 0.0) + 0.004 * prog
    eff["t_rel"] = t_rel


def _apply_pipe_leak(spec: FaultSpec, prog: float, t_rel: float, eff: Dict[str, float]) -> None:
    eff["leak_flow"] = eff.get("leak_flow", 0.0) + 1.5 * prog      # 泄漏量 L/s
    eff["humidity_add"] = eff.get("humidity_add", 0.0) + 25.0 * prog  # 水汽 -> 湿度上升


def _apply_scale_buildup(spec: FaultSpec, prog: float, t_rel: float, eff: Dict[str, float]) -> None:
    eff["r_coil_mult"] = eff.get("r_coil_mult", 1.0) * (1.0 + 2.0 * prog)


FAULT_REGISTRY = {
    "filter_clog": _apply_filter_clog,
    "pump_cavitation": _apply_pump_cavitation,
    "pipe_leak": _apply_pipe_leak,
    "scale_buildup": _apply_scale_buildup,
}


class FaultScheduler:
    """按时间调度多个故障，合成每个仿真步的 effects。"""

    def __init__(self, faults: List[FaultSpec]):
        for f in faults:
            if f.name not in FAULT_REGISTRY:
                raise ValueError(f"未知故障类型: {f.name}，可选: {list(FAULT_REGISTRY)}")
        self.faults = faults

    def effects(self, t: float) -> Dict[str, float]:
        eff: Dict[str, float] = {"t_rel": 0.0}
        for f in self.faults:
            prog = f.progress(t)
            if prog > 0.0:
                FAULT_REGISTRY[f.name](f, prog, t - f.start, eff)
        return eff

    def active_labels(self, t: float) -> List[str]:
        return [f.name for f in self.faults if f.progress(t) > 0.0]


def parse_fault_spec(text: str) -> FaultSpec:
    """解析 CLI 故障表达式，如 "filter_clog@1800:600:0.8"（名称@起始:爬升:程度）。"""
    name, _, rest = text.partition("@")
    parts = rest.split(":") if rest else []
    start = float(parts[0]) if len(parts) > 0 and parts[0] else 0.0
    ramp = float(parts[1]) if len(parts) > 1 and parts[1] else 600.0
    severity = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
    return FaultSpec(name=name, start=start, ramp=ramp, severity=severity)
