"""三层防幻觉校验机制。

第一层：物理定律硬约束（热力学第二定律、热平衡、质量守恒）
第二层：知识图谱交叉验证（推理路径因果关系必须在图谱中存在）
第三层：低置信度人工兜底（由 ConfidenceGate 承载，此处输出校验结论）
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .knowledge_graph import KnowledgeGraph

# 物理量合理上下限（超出即违背物理定律/工程常识）
PHYSICAL_LIMITS = {
    "出水温度": (0.0, 100.0),
    "进水温度": (0.0, 60.0),
    "压力": (0.0, 1000.0),      # kPa
    "流量": (0.0, 50.0),        # L/s
    "水箱液位": (0.0, 500.0),   # cm
    "电导率": (0.0, 3000.0),
    "湿度": (0.0, 100.0),
    "炉内温度": (0.0, 2000.0),
}


@dataclass
class CheckResult:
    """防幻觉校验结果。"""
    physics_ok: bool
    kg_ok: bool
    physics_violations: List[str] = field(default_factory=list)
    kg_violations: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.physics_ok and self.kg_ok


class AntiHallucinationChecker:
    """三层防幻觉校验器。"""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    # ---------------- 第一层：物理定律 ----------------

    def check_physics(self, features: Dict[str, float],
                      claims: Optional[List[str]] = None) -> List[str]:
        """校验特征数值与推理声明是否违背物理定律。返回违规描述列表。"""
        violations = []
        # 1) 数值物理界限
        for name, val in features.items():
            if name in PHYSICAL_LIMITS:
                lo, hi = PHYSICAL_LIMITS[name]
                if not (lo <= val <= hi):
                    violations.append(f"{name}={val} 超出物理界限[{lo},{hi}]")
        # 2) 热力学第二定律：出水温度不得低于进水温度（冷却水吸热）
        if "出水温度" in features and "进水温度" in features:
            if features["出水温度"] < features["进水温度"] - 1.0:
                violations.append(
                    f"出水温度{features['出水温度']}℃ 低于进水温度{features['进水温度']}℃，违背热力学第二定律")
        # 3) 文本声明中的明显物理矛盾（简单规则）
        for c in claims or []:
            if "出水温度" in c and "低于进水" in c and "正常" in c:
                violations.append(f"声明矛盾: {c}")
        return violations

    # ---------------- 第二层：知识图谱交叉验证 ----------------

    def check_kg(self, root_cause: str, evidence_sensors: List[str]) -> List[str]:
        """校验根因与证据传感器之间的因果关系是否在图谱中。返回违规列表。

        只对图谱中已注册的传感器做严格校验；未注册的证据字段（如文字描述）不参与校验。
        """
        violations = []
        if root_cause not in self.kg.fault_names():
            violations.append(f"根因「{root_cause}」不在知识图谱故障域中")
            return violations
        known = self.kg.sensor_names()
        for s in evidence_sensors:
            if s not in known:
                continue  # 非传感器证据（文字描述）跳过
            if not self.kg.causal_path_exists(root_cause, s):
                violations.append(f"推理路径「{root_cause} -> {s}」在知识图谱中无对应因果关系")
        return violations

    # ---------------- 综合 ----------------

    def check(self, features: Dict[str, float], root_cause: str,
              evidence_sensors: List[str], claims: Optional[List[str]] = None) -> CheckResult:
        pv = self.check_physics(features, claims)
        kv = self.check_kg(root_cause, evidence_sensors)
        return CheckResult(physics_ok=not pv, kg_ok=not kv,
                           physics_violations=pv, kg_violations=kv)
