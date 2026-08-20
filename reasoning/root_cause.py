"""大模型根因推理执行体：多跳因果推理 + 置信度评分 + 三层防幻觉闭环。

流程：
1. 异常特征 -> 知识图谱召回候选根因（先验）
2. 构造 CoT 提示（注入工况上下文 + 候选根因 + 图谱事实）-> LLM 多跳推理
3. 解析结构化诊断结果（根因/置信度/证据链/SOP）
4. 三层防幻觉校验，失败重试 ≤3 次，仍失败转人工
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .anti_hallucination import AntiHallucinationChecker, CheckResult
from .confidence import ConfidenceGate, GateDecision
from .knowledge_graph import KnowledgeGraph
from .llm_client import LLMClient

MAX_RETRY = 3


@dataclass
class DiagnosisResult:
    """结构化诊断结果（对齐文档 6.1 输出格式）。"""
    root_cause: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    sop: List[str] = field(default_factory=list)
    level: str = "orange"
    check: Optional[CheckResult] = None
    gate: Optional[GateDecision] = None
    retries: int = 0
    manual_required: bool = False
    raw: str = ""

    def to_dict(self) -> Dict:
        return {
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "sop": self.sop,
            "level": self.level,
            "hallucination_check": {
                "physics": self.check.physics_ok if self.check else None,
                "kg": self.check.kg_ok if self.check else None,
            },
            "gate_action": self.gate.action if self.gate else None,
            "retries": self.retries,
            "manual_required": self.manual_required,
        }


class RootCauseReasoner:
    """根因推理器。"""

    def __init__(self, llm: Optional[LLMClient] = None, kg: Optional[KnowledgeGraph] = None):
        self.llm = llm or LLMClient()
        self.kg = kg or KnowledgeGraph()
        self.checker = AntiHallucinationChecker(self.kg)
        self.gate = ConfidenceGate()

    # ---------------- CoT 提示 ----------------

    def _build_prompt(self, report: Dict, candidates: List[str], extra_hint: str = "") -> str:
        """report 为 L1/L2 上报 + 上下文 的完整诊断输入包。"""
        kg_facts = "；".join(
            f"{c}的关联处置: {','.join(self.kg.actions_for_fault(c)) or '无'}" for c in candidates
        )
        l1 = report.get("l1_alerts", [])
        l1_text = "\n".join(f"- [{a.get('rule_id')}] {a.get('message')}" for a in l1) or "无"
        l2 = report.get("l2_forecast", {})
        l2_text = "\n".join(f"- {k}: {v}" for k, v in l2.items()) or "无"
        return f"""你是中频炉水冷系统的故障诊断专家。综合 L1 规则预警、L2 趋势预测、近期维修工单与工况运行表，做多跳因果推理，定位最可能的物理根因。

【L1 规则预警（实时越限）】
{l1_text}

【L2 趋势预测（模型外推）】
{l2_text}

【实时异常特征】{json.dumps(report.get('features', {}), ensure_ascii=False)}

【当前工况】{report.get('condition', 'unknown')}
【工况运行表】
{report.get('operating_schedule', '无')}

【近期维修工单】
{report.get('maintenance_log', '无')}

【知识图谱候选根因】{', '.join(candidates) if candidates else '（无先验候选）'}
【图谱事实】{kg_facts or '无'}
【鉴别要点】过滤器堵塞=流量显著下降；水泵气蚀=压力/流量震荡；管道泄漏=压力降+液位降+湿度升（若近期维修过相关阀门/管道，泄漏概率大幅提升）；线圈结垢=出水温差大但流量/压力正常（热阻增大）。
{extra_hint}

请严格按以下 JSON 输出（不要输出多余内容）：
{{
  "root_cause": "根因名称（必须来自候选根因或图谱故障域）",
  "confidence": 0.0到1.0的置信度,
  "evidence": ["证据1", "证据2"],
  "sop": ["处置步骤1", "处置步骤2"]
}}"""

    # ---------------- 解析 ----------------

    @staticmethod
    def _parse(text: str) -> Optional[Dict]:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    # ---------------- 主推理 ----------------

    def diagnose(self, features: Optional[Dict[str, float]] = None, condition: str = "unknown",
                 sensor_names: Optional[List[str]] = None,
                 report: Optional[Dict] = None) -> DiagnosisResult:
        """多跳因果推理 + 防幻觉校验 + 置信度分级。

        两种调用方式：
        - 简单：diagnose(features, condition, sensor_names)
        - 完整上下文：diagnose(report={features, condition, l1_alerts, l2_forecast,
                                       operating_schedule, maintenance_log})
        """
        # 统一组装 report
        if report is None:
            report = {"features": features or {}, "condition": condition}
        features = report.get("features", features or {})
        condition = report.get("condition", condition)

        # 1) 图谱先验召回（节点 ID -> 中文名），带命中数
        sensors = sensor_names or list(features.keys())
        ranked = self.kg.faults_for_sensors(sensors)
        candidates = [self.kg.nodes[fid].name for fid, _ in ranked][:3]
        # 图谱 Top1 先验根因（命中数最多），用于稳定兜底
        kg_top1 = candidates[0] if candidates else "未知"
        kg_top1_hits = ranked[0][1] if ranked else 0

        # 2) LLM 推理 + 防幻觉重试
        result: Optional[DiagnosisResult] = None
        hint = ""
        for attempt in range(MAX_RETRY):
            prompt = self._build_prompt(report, candidates, hint)
            raw = self.llm.chat(prompt, max_tokens=4000, temperature=0.1)
            parsed = self._parse(raw) or {}
            rc_llm = parsed.get("root_cause", "")
            conf_llm = float(parsed.get("confidence", 0.5))
            # 根因仲裁（双模型一致性）：
            #  图谱 Top1 与 LLM 一致 -> 采纳并提升置信度；
            #  不一致时，若 LLM 在候选集且高置信 -> 采納 LLM（其利用了流量/压力等连续值鉴别），
            #  否则回退图谱 Top1（物理先验兜底，保证稳定性）。
            if rc_llm == kg_top1:
                rc, conf = kg_top1, min(conf_llm + 0.1, 0.98)
            elif rc_llm in candidates and conf_llm >= 0.8:
                rc, conf = rc_llm, conf_llm
            else:
                rc, conf = kg_top1, max(conf_llm - 0.1, 0.5)
            evidence = parsed.get("evidence", sensors)
            sop = parsed.get("sop", self.kg.actions_for_fault(rc))

            check = self.checker.check(features, rc, evidence)
            result = DiagnosisResult(rc, conf, evidence, sop, check=check, raw=raw, retries=attempt)
            if check.passed:
                break
            # 校验失败，把违规原因反馈给 LLM 重新推理
            hint = ("【上次推理未通过校验，请修正】物理违规: " +
                    ";".join(check.physics_violations) + " 图谱违规: " + ";".join(check.kg_violations))

        # 3) 置信度分级（第三层）
        gate = self.gate.decide(result.confidence)
        result.gate = gate
        if not result.check.passed and result.retries >= MAX_RETRY - 1:
            result.manual_required = True
        if gate.action == "manual":
            result.manual_required = True
        return result
