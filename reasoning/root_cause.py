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
from .confidence import ConfidenceGate, GateDecision, CascadeGate
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
    # ---- MS6 级联元数据 ----
    tier: str = "single"            # single | front_2b | fallback_27b
    upgraded: bool = False          # 是否发生级联升级（前置→兜底）
    upgrade_reason: str = ""        # 升级原因（CascadeGate 判据；空=未升级）
    parse_ok: bool = True           # 模型输出 JSON 是否解析成功
    endpoint_error: str = ""        # 端点异常描述（空=正常）
    prompt: str = ""                # 输入提示（DPO/再训快照，不进 to_dict 以免膨胀 API）

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
            "tier": self.tier,
            "upgraded": self.upgraded,
            "upgrade_reason": self.upgrade_reason,
        }

    def training_snapshot(self) -> Dict:
        """DPO/再训样本快照：输入 prompt + 模型原始输出 + 结构化结论。

        供 action/feedback.py 归档，是构造偏好对的**前置**（含 prompt 才能配对）。
        故意与 to_dict() 分离：prompt/raw 体量大，不应进入实时 API 响应。
        """
        return {
            "prompt": self.prompt,
            "output": self.raw,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "sop": self.sop,
            "tier": self.tier,
            "upgraded": self.upgraded,
            "manual_required": self.manual_required,
        }


class RootCauseReasoner:
    """根因推理器。

    两种模式：
    - 单模型（默认，fallback_llm=None）：行为与级联前一致；`arbitrate` 控制是否
      启用代码仲裁（27B 兜底路径 True）。
    - 级联（MS6，fallback_llm 非空）：`llm`=2B SFT 前置（**关仲裁**，报告 §8.1），
      仅在「JSON 解析失败 / 防幻觉不过 / 端点异常」时升级到 `fallback_llm`=27B
      （**关键：不依赖模型自报置信度**，决策 B / 报告 §6）。
    """

    def __init__(self, llm: Optional[LLMClient] = None, kg: Optional[KnowledgeGraph] = None,
                 fallback_llm: Optional[LLMClient] = None, arbitrate: bool = True,
                 cascade_gate: Optional[CascadeGate] = None):
        self.llm = llm or LLMClient()
        self.fallback_llm = fallback_llm      # None=单模型；非空=级联（llm 为 2B 前置）
        self.arbitrate = arbitrate            # 单模型路径是否启用代码仲裁
        self.kg = kg or KnowledgeGraph()
        self.checker = AntiHallucinationChecker(self.kg)
        self.gate = ConfidenceGate()
        self.cascade_gate = cascade_gate or CascadeGate()

    # ---------------- CoT 提示 ----------------

    def _build_prompt(self, report: Dict, candidates: List[str], extra_hint: str = "") -> str:
        """report 为 L1/L2 上报 + 上下文 的完整诊断输入包。"""
        kg_facts = "；".join(
            f"{c}的关联处置: {','.join(self.kg.actions_for_fault(c)) or '无'}" for c in candidates
        )
        # L1 高频规则会在一个窗口产生数百条重复告警；只保留最近 10 条，
        # 避免把重复证据灌入 2B/27B prompt，拖慢首 token 并挤压统计特征。
        l1 = report.get("l1_alerts", [])[-10:]
        l1_text = "\n".join(f"- [{a.get('rule_id')}] {a.get('message')}" for a in l1) or "无"
        l2 = report.get("l2_forecast", {})
        l2_text = "\n".join(f"- {k}: {v}" for k, v in l2.items()) or "无"
        stats = report.get("stats", {})
        stats_text = "\n".join(f"- {k}: {v}" for k, v in stats.items()) or "无"
        diag_ctx = report.get("diag_context") or {}
        cond_transition = "【工况状态】当前处于工况切换/升降温阶段，L2 异常分升高可能是工况变化导致的正常现象。必须优先依据 L1 规则与统计特征判断：若 L1 未命中且统计特征均正常（无湿度超标、无压力震荡、流量/压力/温差正常），请判定为「工况切换导致的正常波动，未发生故障」，不要误报。" if diag_ctx.get("condition_transition") else ""
        cond_stable = "【工况状态】当前工况稳定，L2 异常分升高更可能是真实故障，请结合统计特征确诊。" if diag_ctx.get("condition_transition") is False else ""
        return f"""你是中频炉水冷系统的故障诊断专家。综合 L1 规则预警、L2 趋势预测、统计鉴别特征、近期维修工单与工况运行表，做多跳因果推理，定位最可能的物理根因。

【L1 规则预警（实时越限）】
{l1_text}

【L2 趋势预测（模型外推）】
{l2_text}

【统计鉴别特征（窗口计算，判定依据，必须优先依据此节）】
{stats_text}
判定规则（严格遵循）：
- 湿度均值 > 70%RH，或 湿度上升量 > 4%RH 且流量/压力下降 → 管道泄漏（水汽逸散铁证；湿度趋势上升即使绝对值未达70也判泄漏）
- 压力波动幅度_std（已去趋势）> 3kPa 且 湿度均值 ≤ 65%RH → 水泵气蚀（压力震荡、湿度正常）
- PQ特性偏移 ≥ +10%（实测压力显著高于水力模型预测 P_model=120+2.2·Q²，线圈/管网阻抗抬升的确定性证据）且无压力震荡且湿度未上升 → 线圈结垢（线圈热阻增大；流量通常伴随下降）
- 流量 < 6.4L/s 且 PQ特性偏移 < +10% 且无压力震荡且湿度未上升 → 过滤器堵塞（过滤器段阻抗升高：流量下降但线圈侧压力贴近模型，湿度不变）
- 进出水温差 > 20℃ 且流量正常或温差持续爬升 → 线圈结垢（同功率下温差拉大是线圈热阻增大的表现）
- 单位电耗温升率明显下降（炉温爬升变缓而电功率不变）→ 支持换热效率衰减类根因（如线圈结垢）
注意1：判定堵塞/结垢必须依据 PQ 特性偏移的相对量证据，不要依赖压力绝对数值。
注意2：湿度持续上升是泄漏区别于堵塞/结垢的关键——后两者不改变湿度，泄漏使湿度单调上升。

【实时异常特征】{json.dumps(report.get('features', {}), ensure_ascii=False)}

【当前工况】{report.get('condition', 'unknown')}
{cond_transition}
{cond_stable}
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

        单模型 vs 级联由构造时是否传入 fallback_llm 决定（见类文档）。
        """
        # 统一组装 report
        if report is None:
            report = {"features": features or {}, "condition": condition}
        features = report.get("features", features or {})
        condition = report.get("condition", condition)

        # 1) 图谱先验召回（节点 ID -> 中文名）
        sensors = sensor_names or list(features.keys())
        ranked = self.kg.faults_for_sensors(sensors)
        candidates = [self.kg.nodes[fid].name for fid, _ in ranked][:3]
        # 统计预鉴别结果并入候选集（保证 stats 特征指向的根因可被选中）
        for extra in report.get("extra_candidates", []):
            if extra in self.kg.fault_names() and extra not in candidates:
                candidates.append(extra)
        # 图谱 Top1 先验根因（命中数最多），用于稳定兜底
        kg_top1 = candidates[0] if candidates else "未知"

        # 2) 级联 or 单模型
        if self.fallback_llm is not None:
            return self._diagnose_cascade(report, sensors, candidates, kg_top1, features)
        result = self._run_model(report, sensors, candidates, kg_top1, features,
                                 arbitrate=self.arbitrate, llm=self.llm)
        return self._finalize(result)

    # ---------------- 级联（MS6） ----------------

    def _diagnose_cascade(self, report: Dict, sensors: List[str], candidates: List[str],
                          kg_top1: str, features: Dict) -> DiagnosisResult:
        """2B SFT 前置（关仲裁）→ 仅在可验证信号命中时升级 27B（不依赖模型置信度）。"""
        # 前置 2B：关仲裁，信任 SFT 原始输出（报告 §8.1）
        front = self._run_model(report, sensors, candidates, kg_top1, features,
                                arbitrate=False, llm=self.llm, max_tokens=1000)
        front.tier = "front_2b"
        decision = self.cascade_gate.decide(
            parse_ok=front.parse_ok,
            check_passed=bool(front.check and front.check.passed),
            endpoint_error=front.endpoint_error or None,
            root_cause=front.root_cause,
            expected_candidates=report.get("extra_candidates", []))
        front.upgrade_reason = decision.reason
        if not decision.upgrade:
            # 2B 直出：gate 仅用于呈现分级，不因 2B 未校准置信度转人工（决策 B）
            front.gate = self.gate.decide(front.confidence)
            return front
        # 升级 27B：含仲裁 + 重试（原鲁棒路径）
        fb = self._run_model(report, sensors, candidates, kg_top1, features,
                             arbitrate=True, llm=self.fallback_llm, max_tokens=512)
        fb.tier = "fallback_27b"
        fb.upgraded = True
        fb.upgrade_reason = decision.reason
        return self._finalize(fb)

    # ---------------- 单模型推理循环（可复用于前置/兜底） ----------------

    def _run_model(self, report: Dict, sensors: List[str], candidates: List[str],
                   kg_top1: str, features: Dict, *, arbitrate: bool,
                   llm: LLMClient, max_tokens: int = 4000) -> DiagnosisResult:
        """对单个端点做「推理→（可选）仲裁→防幻觉校验→重试」，返回结果。

        arbitrate=True 时启用代码仲裁（27B 兜底/单模型）；arbitrate=False 时信任
        模型原始输出（2B SFT 前置）。端点异常被捕获为 endpoint_error 占位结果，
        交由上层（级联/finalize）处理，不向外抛出。
        """
        result: Optional[DiagnosisResult] = None
        hint = ""
        for attempt in range(MAX_RETRY):
            prompt = self._build_prompt(report, candidates, hint)
            try:
                raw = llm.chat(prompt, max_tokens=max_tokens, temperature=0.1)
            except Exception as e:  # 端点不可用/超时：占位结果 + 端点异常标记
                return DiagnosisResult(
                    kg_top1 or "未知", 0.5, list(sensors),
                    self.kg.actions_for_fault(kg_top1), retries=attempt,
                    parse_ok=False, prompt=prompt,
                    endpoint_error=f"{type(e).__name__}: {e}")
            parsed = self._parse(raw)
            parse_ok = parsed is not None
            parsed = parsed or {}
            rc_llm = parsed.get("root_cause", "")
            conf_llm = float(parsed.get("confidence", 0.5) or 0.5)
            evidence = parsed.get("evidence", sensors)

            if arbitrate:
                rc, conf, sop = self._arbitrate(rc_llm, conf_llm, kg_top1,
                                                candidates, report, parsed)
            else:
                rc, conf = rc_llm, conf_llm
                sop = parsed.get("sop", self.kg.actions_for_fault(rc))

            check = self.checker.check(features, rc, evidence)
            result = DiagnosisResult(rc, conf, evidence, sop, check=check, raw=raw,
                                     retries=attempt, parse_ok=parse_ok, prompt=prompt)
            # 无论是否启用仲裁，最终对外输出都必须来自可解析 JSON；
            # 级联前置的 parse_ok=False 会升级，27B 兜底仍失败则转人工。
            if check.passed and parse_ok:
                break
            # 校验失败，把违规原因反馈给 LLM 重新推理
            hint = ("【上次推理未通过校验，请修正】物理违规: " +
                    ";".join(check.physics_violations) + " 图谱违规: " +
                    ";".join(check.kg_violations))
        return result

    def _arbitrate(self, rc_llm: str, conf_llm: float, kg_top1: str,
                   candidates: List[str], report: Dict, parsed: Dict):
        """代码仲裁（双模型一致性 + 统计强先验 + 三条统计硬校验）。

        注：为弱基线设计的兜底修正；离线评估证明它会改错 2B SFT 的正确答案
        （报告 §8.1），故级联前置路径**不启用**（arbitrate=False）。
        """
        # 双模型一致性
        if rc_llm == kg_top1:
            rc, conf = kg_top1, min(conf_llm + 0.1, 0.98)
        elif rc_llm in candidates and conf_llm >= 0.8:
            rc, conf = rc_llm, conf_llm
        else:
            rc, conf = kg_top1, max(conf_llm - 0.1, 0.5)
        # 统计强先验（extra_candidates 由确定性物理规则产生，如湿度>70%RH=泄漏铁证）
        extra_cands = report.get("extra_candidates", [])
        if extra_cands and rc not in extra_cands:
            rc = extra_cands[0]
            conf = max(min(conf_llm, 0.85), 0.7)
        sop = parsed.get("sop", self.kg.actions_for_fault(rc))
        # 三条统计硬校验
        stats = report.get("stats", {})
        press_std = float(stats.get("压力波动幅度_std_kPa", 0.0) or 0.0)
        hum_mean = float(stats.get("湿度均值_pctRH", 50.0) or 50.0)
        hum_delta = float(stats.get("湿度上升量_pctRH", 0.0) or 0.0)
        pq_off = float(stats.get("PQ特性偏移_pct", 0.0) or 0.0)
        if rc == "管道泄漏" and hum_mean <= 65.0 and hum_delta <= 4.0:
            if press_std > 3.0:
                rc = "水泵气蚀"  # 压力震荡+湿度正常 -> 气蚀
                sop = self.kg.actions_for_fault(rc)
            conf = min(conf, 0.75)  # 与统计特征矛盾，置信度封顶
        elif rc in ("过滤器堵塞", "线圈结垢") and hum_delta > 4.0:
            rc = "管道泄漏"  # 湿度显著上升=水汽逸散，堵塞/结垢不改变湿度
            sop = self.kg.actions_for_fault(rc)
            conf = max(min(conf, 0.85), 0.7)
        if rc == "过滤器堵塞" and hum_delta <= 4.0 and press_std <= 3.0 \
                and pq_off >= 10.0:
            rc = "线圈结垢"  # PQ 偏移≥+10% 说明线圈阻抗抬升
            sop = self.kg.actions_for_fault(rc)
            conf = max(min(conf, 0.85), 0.7)
        return rc, conf, sop

    def _finalize(self, result: DiagnosisResult) -> DiagnosisResult:
        """置信度分级（呈现）+ 转人工判定（校验失败/端点异常/低置信）。"""
        result.gate = self.gate.decide(result.confidence)
        if result.endpoint_error or not result.parse_ok:
            result.manual_required = True
        if result.check is not None and not result.check.passed \
                and result.retries >= MAX_RETRY - 1:
            result.manual_required = True
        if result.gate.action == "manual":
            result.manual_required = True
        return result
