"""标准化运维工单生成器。

输入 L3 诊断结果（DiagnosisResult），输出结构化 JSON 工单，
字段对齐文档 6.1 展示格式：预警级别/触发时间/异常特征/根因推理/
防幻觉校验结果/处置 SOP/备件。
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from reasoning.root_cause import DiagnosisResult

# 故障 -> 备件映射
SPARE_PARTS = {
    "管道泄漏": ["DN50密封圈", "检漏仪"],
    "过滤器堵塞": ["Y型滤网", "清洗剂"],
    "水泵气蚀": ["备用泵机封", "轴承"],
    "线圈结垢": ["弱酸清洗剂", "除垢剂"],
    "水质恶化": ["去离子水", "离子交换树脂"],
    "电气柜凝露": ["除湿机", "绝缘干燥剂"],
}

# 故障 -> 级别（默认，可被诊断覆盖）
FAULT_LEVEL = {
    "管道泄漏": "red",
    "线圈结垢": "orange",
    "过滤器堵塞": "orange",
    "水泵气蚀": "orange",
    "水质恶化": "orange",
    "电气柜凝露": "yellow",
}


@dataclass
class MaintenanceWorkOrder:
    """结构化运维工单。"""
    order_id: str
    level: str                      # red/orange/yellow
    trigger_time: str
    features: str                   # 异常特征描述
    root_cause: str
    confidence: float
    evidence: List[str]
    hallucination_check: Dict
    sop: List[str]
    spare_parts: List[str] = field(default_factory=list)
    manual_required: bool = False

    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "level": self.level,
            "trigger_time": self.trigger_time,
            "features": self.features,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "hallucination_check": self.hallucination_check,
            "sop": self.sop,
            "spare_parts": self.spare_parts,
            "manual_required": self.manual_required,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class WorkOrderGenerator:
    """工单生成器。"""

    _counter = 0

    def generate(self, diag: DiagnosisResult,
                 features_text: str = "",
                 trigger_time: Optional[str] = None) -> MaintenanceWorkOrder:
        """由诊断结果生成工单。"""
        WorkOrderGenerator._counter += 1
        ts = trigger_time or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        oid = f"WO-{pd.Timestamp.now().strftime('%Y%m%d')}-{WorkOrderGenerator._counter:04d}"

        level = diag.level if diag.level in ("red", "orange", "yellow") else \
            FAULT_LEVEL.get(diag.root_cause, "orange")
        # 人工兜底/低置信度降级为黄色（不直接下红/橙工单）
        if diag.manual_required:
            level = "yellow"

        return MaintenanceWorkOrder(
            order_id=oid,
            level=level,
            trigger_time=ts,
            features=features_text or "；".join(diag.evidence[:2]),
            root_cause=diag.root_cause,
            confidence=diag.confidence,
            evidence=diag.evidence,
            hallucination_check={
                "physics": diag.check.physics_ok if diag.check else None,
                "kg": diag.check.kg_ok if diag.check else None,
                "confidence": diag.confidence,
            },
            sop=diag.sop,
            spare_parts=SPARE_PARTS.get(diag.root_cause, []),
            manual_required=diag.manual_required,
        )
