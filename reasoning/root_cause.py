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
        stats = report.get("stats", {})
        stats_text = "\n".join(f"- {k}: {v}" for k, v in stats.items()) or "无"
        return f"""你是中频炉水冷系统的故障诊断专家。综合 L1 规则预警、L2 趋势预测、统计鉴别特征、近期维修工单与工况运行表，做多跳因果推理，定位最可能的物理根因。

【L1 规则预警（实时越限）】
{l1_text}

【L2 趋势预测（模型外推）】
{l2_text}

【统计鉴别特征（窗口计算，判定依据，必须优先依据此节）】
{stats_text}
判定规则（严格遵循）：
- 压力波动幅度_std（已去趋势）> 3kPa 且 湿度均值 ≤ 65%RH → 水泵气蚀（压力震荡、湿度正常）
- 压力 < 150kPa 且 湿度均值 > 70%RH → 管道泄漏（湿度显著升高是泄漏的必要条件，湿度均值 ≤ 65%RH 时禁止判泄漏）
- 流量 < 6.4L/s 且 压力 < 230kPa 且无压力震荡 → 过滤器堵塞（过滤器阻抗升高：流量低+压力低）
- 流量 < 6.4L/s 且 压力 > 230kPa → 线圈结垢（线圈热阻增大：流量低+压力偏高）
- 进出水温差 > 20℃ 且流量正常 → 线圈结垢

【实时异常特征】{json.dumps(report.get('features', {}), ensure_ascii=False)}

【当前工况】{report.get('condition', 'unknown')}
【工况运行表】
{report.get('operating_schedule', '无')}

【近期维修工单】
{report.get('maintenance_log', '无')}
注意：维修工单仅是辅助证据，必须与实时统计特征吻合才能支持对应根因；湿度均值未超 70%RH 时不得诊断为管道泄漏。

【知识图谱候选根因】{', '.join(candidates) if candidates else '（无先验候选）'}
【图谱事实】{kg_facts or '无'}
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
        # 统计预鉴别结果并入候选集（保证 stats 特征指向的根因可被选中）
        for extra in report.get("extra_candidates", []):
            if extra in self.kg.fault_names() and extra not in candidates:
                candidates.append(extra)
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

            # 统计强先验仲裁：extra_candidates 由确定性物理规则产生（如湿度>70%RH=泄漏铁证）。
            # LLM 与统计先验矛盾时回退统计先验（非思考模式 LLM 易混淆湿度成因：凝露 vs 泄漏）。
            extra_cands = report.get("extra_candidates", [])
            if extra_cands and rc not in extra_cands:
                rc = extra_cands[0]
                conf = max(min(conf_llm, 0.85), 0.7)
                sop = self.kg.actions_for_fault(rc) or sop
            evidence = parsed.get("evidence", sensors)
            sop = parsed.get("sop", self.kg.actions_for_fault(rc))

            # 统计硬校验（防 LLM 误判，演示稳定性保障）：
            #  湿度均值未显著升高(≤65%RH)时不得诊断为管道泄漏；
            #  此时若压力波动显著(std>3kPa)则应为水泵气蚀。
            stats = report.get("stats", {})
            press_std = float(stats.get("压力波动幅度_std_kPa", 0.0) or 0.0)
            hum_mean = float(stats.get("湿度均值_pctRH", 50.0) or 50.0)
            if rc == "管道泄漏" and hum_mean <= 65.0:
                if press_std > 3.0:
                    rc = "水泵气蚀"  # 压力震荡+湿度正常 -> 气蚀
                    sop = self.kg.actions_for_fault(rc)
                conf = min(conf, 0.75)  # 与统计特征矛盾，置信度封顶

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
