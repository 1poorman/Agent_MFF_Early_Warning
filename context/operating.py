"""工况运行表（Operating Schedule）。

记录生产班次/工况阶段安排，作为 LLM 推理的工况上下文——
当前处于熔炼/保温/出炉等阶段，直接影响参数正常基线与根因判定。
"""

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class OperatingPhase:
    """单个工况阶段。"""
    start: str              # 开始时间
    end: str                # 结束时间
    phase: str              # startup/melting/holding/tapping/idle
    load_rate: float = 1.0  # 负载率
    note: str = ""


@dataclass
class OperatingSchedule:
    """工况运行表。"""
    phases: List[OperatingPhase] = field(default_factory=list)

    def phase_at(self, timestamp: str) -> Optional[OperatingPhase]:
        ts = pd.Timestamp(timestamp)
        for p in self.phases:
            if pd.Timestamp(p.start) <= ts <= pd.Timestamp(p.end):
                return p
        return None

    def to_prompt_text(self, current_time: Optional[str] = None) -> str:
        if not self.phases:
            return "无工况安排"
        cur = self.phase_at(current_time) if current_time else None
        lines = []
        for p in self.phases[-6:]:  # 最近 6 个阶段
            mark = " ←当前" if (cur and p is cur) else ""
            lines.append(f"- {p.start}~{p.end} {p.phase} 负载{p.load_rate:.0%}{mark}")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(p) for p in self.phases])


def default_operating_schedule() -> OperatingSchedule:
    """默认工况运行表（一个完整熔炼班次，对应 simulator 工况循环）。"""
    return OperatingSchedule(phases=[
        OperatingPhase("2026-07-25 08:00", "2026-07-25 09:00", "startup", 1.0, "冷炉启动熔炼"),
        OperatingPhase("2026-07-25 09:00", "2026-07-25 09:30", "melting", 1.0, "满功率熔炼"),
        OperatingPhase("2026-07-25 09:30", "2026-07-25 10:00", "holding", 0.4, "保温待浇"),
        OperatingPhase("2026-07-25 10:00", "2026-07-25 10:10", "tapping", 0.0, "出炉"),
        OperatingPhase("2026-07-25 10:10", "2026-07-25 10:20", "idle", 0.0, "加冷料待机"),
        OperatingPhase("2026-07-25 10:20", "2026-07-25 11:20", "startup", 1.0, "第二炉启动"),
    ])
