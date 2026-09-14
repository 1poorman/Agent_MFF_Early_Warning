"""置信度分级门控（防幻觉第三层）+ 级联升级门控（MS6）。

`ConfidenceGate`（基于模型自报置信度）——仅用于**结果呈现**分级：
- ≥90%: 直接输出诊断结论并生成工单
- 70~90%: 输出 Top3 候选根因，附置信度，供运维参考
- <70%:  不输出结论，转人工辅助诊断

`CascadeGate`（MS6，**决策 B**）——用于 2B 前置→27B 兜底的**升级路由**：
**不依赖模型自报置信度**（离线评估证明 SFT 阶段置信度未校准、不可作安全闸门，
详见 design/SFT_EVAL_REPORT.md 第六节），仅依据可验证信号（JSON 解析 /
防幻觉校验 / 端点可用性）决定是否升级到 27B。
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
    """置信度分级门控（仅用于结果呈现，不用于级联路由）。"""

    def decide(self, confidence: float) -> GateDecision:
        if confidence >= THRESH_HIGH:
            return GateDecision(ACTION_DIRECT, confidence, "置信度≥90%，直接输出诊断结论")
        if confidence >= THRESH_LOW:
            return GateDecision(ACTION_CANDIDATES, confidence, "置信度70~90%，输出Top3候选根因")
        return GateDecision(ACTION_MANUAL, confidence, "置信度<70%，转人工辅助诊断")


@dataclass
class UpgradeDecision:
    """级联升级决策。"""
    upgrade: bool
    reason: str


class CascadeGate:
    """级联升级门控（MS6，决策 B：不依赖模型自报置信度）。

    升级触发条件仅来自**可验证信号**，任一命中即升级 27B：
      1. 前置端点异常（endpoint_error 非空）——服务不可用/超时；
      2. JSON 解析失败（parse_ok=False）——输出不可结构化；
      3. 防幻觉校验不过（check_passed=False）——物理/图谱违规，重试耗尽仍不过。
      4. 与确定性统计预判不一致（expected_candidates 非空且模型根因不在其中）。
    2B SFT 关仲裁下，上述信号是"模型对该样本不可靠"的客观证据；
    模型自报的 confidence **不参与**该决策（见 SFT_EVAL_REPORT.md 第六节）。
    """

    def decide(self, *, parse_ok: bool, check_passed: bool,
               endpoint_error: Optional[str] = None,
               root_cause: str = "",
               expected_candidates: Optional[List[str]] = None) -> UpgradeDecision:
        if endpoint_error:
            return UpgradeDecision(True, f"前置端点异常，升级兜底: {endpoint_error}")
        if not parse_ok:
            return UpgradeDecision(True, "前置输出 JSON 解析失败，升级兜底")
        if not check_passed:
            return UpgradeDecision(True, "前置防幻觉校验不过（物理/图谱违规），升级兜底")
        if expected_candidates and root_cause not in expected_candidates:
            return UpgradeDecision(
                True,
                "前置根因与统计预判不一致（期望: %s，实际: %s），升级兜底"
                % ("/".join(expected_candidates), root_cause or "空"),
            )
        return UpgradeDecision(False, "前置结果通过校验，2B 直出")
