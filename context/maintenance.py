"""近期维修工单（Maintenance Work Orders）。

记录设备近期维修/保养历史，作为 LLM 根因推理的关键上下文——
例如"3号阀上月更换密封圈"是诊断"3号阀后管道泄漏"的决定性证据。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class WorkOrder:
    """单条维修工单。"""
    order_id: str
    date: str               # 维修日期
    component: str          # 维修部件
    action: str             # 维修内容
    note: str = ""          # 备注


@dataclass
class MaintenanceLog:
    """维修工单台账。"""
    orders: List[WorkOrder] = field(default_factory=list)

    def recent(self, days: int = 30, ref_date: Optional[str] = None) -> List[WorkOrder]:
        """返回近 days 天的工单（按 ref_date 起算，默认最新日期）。"""
        if not self.orders:
            return []
        ref = pd.Timestamp(ref_date) if ref_date else max(pd.Timestamp(o.date) for o in self.orders)
        cutoff = ref - pd.Timedelta(days=days)
        return [o for o in self.orders if pd.Timestamp(o.date) >= cutoff]

    def for_component(self, component: str) -> List[WorkOrder]:
        return [o for o in self.orders if component in o.component]

    def to_prompt_text(self, days: int = 60) -> str:
        """生成注入 LLM 提示词的文本。"""
        orders = self.recent(days=days)
        if not orders:
            return "近期无维修记录"
        lines = [f"- [{o.date}] {o.component}: {o.action}" + (f"（{o.note}）" if o.note else "")
                 for o in orders]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(o) for o in self.orders])


def default_maintenance_log() -> MaintenanceLog:
    """默认近期维修工单（对齐知识库 6.1 案例：3号阀 2026-06-20 更换密封圈）。"""
    return MaintenanceLog(orders=[
        WorkOrder("WO-20260725-01", "2026-07-25", "循环水泵", "例行盘车检查", "运行正常"),
        WorkOrder("WO-20260720-02", "2026-07-20", "Y型过滤器", "清洗滤网", "滤网有少量杂质"),
        WorkOrder("WO-20260715-03", "2026-07-15", "板式换热器", "拆洗", "换热效率恢复"),
        WorkOrder("WO-20260620-04", "2026-06-20", "3号阀", "更换DN50密封圈", "关键记录：阀后管道需重点关注"),
        WorkOrder("WO-20260610-05", "2026-06-10", "离子交换树脂", "再生处理", "电导率恢复正常"),
        WorkOrder("WO-20260515-06", "2026-05-15", "感应线圈", "高压风干+绝缘测试", "绝缘电阻 15MΩ"),
    ])
