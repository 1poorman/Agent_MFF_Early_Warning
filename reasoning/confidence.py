"""置信度分级门控（防幻觉第三层：低置信度人工兜底）。

- ≥90%: 直接输出诊断结论并生成工单
- 70~90%: 输出 Top3 候选根因，附置信度，供运维参考
- <70%:  不输出结论，转人工辅助诊断
"""

from dataclasses import dataclass
from typing import List, Optional

THRESH_HIGH = 0.90
THRESH_LOW = 0.70

ACTION_DIRECT = "direct"       # 直接输出
ACTION_CANDIDATES = "candidates"  # Top3 候选
ACTION_MANUAL = "manual"       # 人工兜底


@dataclass
class GateDecision:
    action: str
    confidence: float
    reason: str


class ConfidenceGate:
    """置信度分级门控。"""

    def decide(self, confidence: float) -> GateDecision:
        if confidence >= THRESH_HIGH:
            return GateDecision(ACTION_DIRECT, confidence, "置信度≥90%，直接输出诊断结论")
        if confidence >= THRESH_LOW:
            return GateDecision(ACTION_CANDIDATES, confidence, "置信度70~90%，输出Top3候选根因")
        return GateDecision(ACTION_MANUAL, confidence, "置信度<70%，转人工辅助诊断")
