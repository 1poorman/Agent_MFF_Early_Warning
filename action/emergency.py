"""应急预案联动（知识库模块五）。

红色预警自动关联应急预案：
- 突发停电 -> 柴油备用泵 / 高位重力水箱供水，压力不足则倾炉
- 漏水入铁水 -> 立即倾炉+切断水源，人员撤离（防爆）
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EmergencyPlan:
    """应急预案。"""
    plan_id: str
    name: str
    trigger_fault: str          # 关联故障
    risk: str                   # 风险分析
    steps: List[str]            # 应急响应流程
    forbidden: List[str] = field(default_factory=list)  # 禁止事项


# 预案库（源自 docs/故障处理.md 模块五）
EMERGENCY_PLANS: Dict[str, EmergencyPlan] = {
    "管道泄漏": EmergencyPlan(
        plan_id="EP-002",
        name="炉体漏水入铁水应急",
        trigger_fault="管道泄漏",
        risk="水接触高温铁水瞬间汽化，体积膨胀上千倍，引发蒸汽爆炸",
        steps=[
            "第一优先级：操作人员立即撤离至安全防爆区",
            "立即倾炉倒出铁水",
            "切断水源",
            "保护现场，事故调查，设备检修更换",
        ],
        forbidden=["严禁向漏水的炉体直接浇水灭火"],
    ),
    "线圈烧穿": EmergencyPlan(
        plan_id="EP-001",
        name="突发全厂停电/穿炉应急",
        trigger_fault="线圈烧穿",
        risk="水泵停转炉内铁水热量无法带走，导致线圈烧毁甚至铁水穿炉爆炸",
        steps=[
            "系统自动启动柴油备用泵或高位重力水箱供水",
            "检查供水压力：正常则继续冷却，不足则立即倾炉",
            "倾炉操作：铁水倒入备用铁水包或安全区域",
        ],
        forbidden=["严禁炉内留铁水"],
    ),
}


class EmergencyPlanner:
    """应急预案联动器。"""

    def match(self, root_cause: str, level: str) -> Optional[EmergencyPlan]:
        """红色预警且命中预案库故障时，返回关联应急预案。"""
        if level != "red":
            return None
        return EMERGENCY_PLANS.get(root_cause)

    def attach(self, work_order) -> Optional[EmergencyPlan]:
        """为工单挂载应急预案（若匹配）。"""
        plan = self.match(work_order.root_cause, work_order.level)
        if plan:
            work_order.sop = [f"【应急预案 {plan.plan_id} {plan.name}】"] + plan.steps + work_order.sop
        return plan
