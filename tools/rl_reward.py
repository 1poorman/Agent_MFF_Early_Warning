#!/usr/bin/env python3
"""RL 后训练的统一奖励函数（DPO 组对 / GRPO 在线 / RM 训练共用同一真源）。

设计依据：design/SFT_LORA_COMPARISON.md 第 5 节残余缺口 + docs/SFT_STATUS.md MS7。
SFT 两臂的故障准确率已饱和在 99.41%，但**拒诊率（80~90%）**与**置信度 gap（为负）**
过不了门槛，且模仿学习无法再改善。本奖励的靶点即这两项，而非再抬准确率。

分项与权重（JSON 不可解析时硬门闸直接给 0）：
    R = 0.50*r_gold + 0.15*r_calib + 0.15*r_hallu + 0.10*r_faith + 0.05*r_json + 0.05*r_sop

r_calib 用 Brier 严格恰当评分 1-(conf-correct)^2：最优策略就是如实报告自身正确概率，
判对抬 conf、判错压 conf 都得分，正好把 gap 推向正号。

⚠️ 本模块**不得** import tests/eval_sft_model.py：后者在 import 时执行
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
会把训练进程的 GPU 直接藏掉。因此下面的纯函数/常量是从该评估器移植的副本，
并由 tools/check_rl_reward.py 断言两边逐条同分（G0 决策门第 3 项）。
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoning.anti_hallucination import AntiHallucinationChecker  # noqa: E402
from reasoning.knowledge_graph import KnowledgeGraph  # noqa: E402

NO_FAULT = "无故障（工况切换正常波动）"
DATA_BAD = "数据质量异常（建议人工核查）"

# 与 tests/eval_sft_model.py:97 同源：模型需内化的判定阈值，引用不计为幻觉。
THRESHOLD_CONSTANTS = {2.2, 3, 4, 6.4, 7.8, 10, 20, 50, 55, 60, 65, 70, 100, 120}

# 奖励权重（和必须为 1.0，构造时断言）
WEIGHTS = {
    "gold": 0.50,
    "calib": 0.15,
    "hallu": 0.15,
    "faith": 0.10,
    "json": 0.05,
    "sop": 0.05,
}
REQUIRED_KEYS = ("root_cause", "confidence", "evidence", "sop")


# ---------------- 移植自评估器的纯函数（副本，由 G0 一致性测试守护） ----------------

def numbers_in(text: str) -> list[float]:
    """提取带符号数值（负号不能丢）。对齐 tests/eval_sft_model.py:90。"""
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]


def extract_features(prompt: str) -> dict:
    """从退化版 prompt 的【实时异常特征】段还原 features。对齐 eval_sft_model.py:68。"""
    m = re.search(r"【实时异常特征】\s*(\{.*?\})", prompt, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}


def evidence_faithful(evidence, input_nums) -> tuple[bool, list[float]]:
    """证据中每个数值须能在输入 prompt 找到（±5% 或 0.15 容差）。

    对齐 tests/eval_sft_model.py:100。返回 (ok, bad_nums)。
    """
    bad = []
    for ev in evidence or []:
        for num in numbers_in(str(ev)):
            if num in THRESHOLD_CONSTANTS:
                continue
            if not any(abs(num - v) <= max(0.05 * abs(v), 0.15) for v in input_nums):
                bad.append(num)
    return (not bad), bad


def parse_output(text: str) -> dict | None:
    """解析模型输出末尾的 JSON。与 reasoning/root_cause.py:169 _parse 同语义。"""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------- 打分 ----------------

@dataclass
class Score:
    """单条 rollout 的奖励明细（全部留痕，便于事后审计奖励劫持）。"""
    total: float
    parts: dict[str, float] = field(default_factory=dict)
    root_cause: str | None = None
    confidence: float | None = None
    correct: bool = False
    json_ok: bool = False
    faith_ok: bool = False
    hallu_ok: bool = False
    bad_nums: list = field(default_factory=list)
    violations: list = field(default_factory=list)


class RewardScorer:
    """构造一次、复用多次。KnowledgeGraph 与 checker 都是纯内存结构，无模型加载。"""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(weights or WEIGHTS)
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"reward weights must sum to 1.0, got {total}")
        self.kg = KnowledgeGraph()
        self.checker = AntiHallucinationChecker(self.kg)
        self._fault_names = self.kg.fault_names()

    def score(self, prompt: str, completion: str, gold: str) -> Score:
        """prompt 为完整 user 文本（含 stats/实时特征），completion 为模型原始输出。"""
        parsed = parse_output(completion)
        if parsed is None or any(k not in parsed for k in REQUIRED_KEYS):
            # 硬门闸：解析不出或缺键，整条 0 分，不进入加权。
            return Score(total=0.0, parts={k: 0.0 for k in self.weights}, json_ok=False)

        root_cause = parsed.get("root_cause")
        try:
            confidence = float(parsed["confidence"])
        except (TypeError, ValueError):
            return Score(total=0.0, parts={k: 0.0 for k in self.weights})
        evidence, sop = parsed["evidence"], parsed["sop"]
        if (not isinstance(root_cause, str) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1 or isinstance(parsed["confidence"], bool)
                or not isinstance(evidence, list) or not isinstance(sop, list)
                or any(not isinstance(s, str) for s in evidence + sop)):
            return Score(total=0.0, parts={k: 0.0 for k in self.weights})

        correct = root_cause == gold
        features = extract_features(prompt)

        faith_ok, bad_nums = evidence_faithful(evidence, numbers_in(prompt))
        # 只惩罚**模型自己造成**的物理违规：antihall 输入本身就是故意越界的，
        # 把输入固有的违规算到模型头上，会反过来训练模型不要正确拒诊。
        # 评估器同样把 antihall 排除在防幻觉分母外（tests/eval_sft_model.py:278）。
        input_physics = set(self.checker.check_physics(features, []))
        physics = [v for v in self.checker.check_physics(features, evidence)
                   if v not in input_physics]
        kg_violations = self.checker.check_kg(root_cause, evidence) \
            if root_cause in self._fault_names else []
        hallu_ok = (not physics) and (not kg_violations)

        parts = {
            "json": 1.0,
            "gold": 1.0 if correct else 0.0,
            # Brier 严格恰当评分：如实报告自身正确概率才是最优策略。
            "calib": 1.0 - (confidence - (1.0 if correct else 0.0)) ** 2,
            "hallu": 1.0 if hallu_ok else 0.0,
            "faith": 1.0 if faith_ok else 0.0,
            "sop": self._sop_score(root_cause, sop),
        }
        total = sum(self.weights[k] * parts[k] for k in self.weights)
        return Score(total=total, parts=parts, root_cause=root_cause,
                     confidence=confidence, correct=correct, json_ok=True,
                     faith_ok=faith_ok, hallu_ok=hallu_ok, bad_nums=bad_nums,
                     violations=physics + kg_violations)

    def _sop_score(self, root_cause: str | None, sop: Any) -> float:
        """sop 非空即得基础分；属图谱故障域时再看是否落在该故障的动作域内。"""
        if not isinstance(sop, list) or not sop:
            return 0.0
        if root_cause not in self._fault_names:
            # 无故障/数据质量异常没有图谱动作域，非空即满分。
            return 1.0
        allowed = set(self.kg.actions_for_fault(root_cause))
        if not allowed:
            return 1.0
        hit = sum(1 for step in sop if step.strip() and any(a in step or step in a for a in allowed))
        return min(hit / len(sop), 1.0)


def make_reward_func(prompt_to_gold: dict[str, str], scorer: RewardScorer | None = None,
                     on_score=None):
    """构造 TRL GRPO/RLOO 需要的 reward_funcs 回调。

    TRL 以 (prompts, completions, **kwargs) 批量调用，要求返回 list[float]。
    数据集里除 `prompt` 之外的列会被 TRL 原样透传进 kwargs，本函数用到两列：

    * `gold`      —— 金标根因；缺省时退回 prompt 文本查表。
    * `user_text` —— **必须给**。打分要用的是 user 原文，不是套过模板的 prompt：
      渲染后的串里含 system 提示，而 system 里写着阈值数字，`numbers_in()` 会把
      它们也算成"输入里出现过的数"，于是 `evidence_faithful` 在训练时比采样打分
      时更宽松 —— 同一条输出在 rollouts.jsonl 里和 GRPO 训练中得分不同，奖励就
      不再是单一真源。缺这一列时退回 prompts（仅为兼容，会打印口径警告）。

    `on_score(score, gold, index)` 可选回调，用于外部累计奖励劫持监控指标
    （置信度分布标准差、判错平均置信度）。
    """
    scorer = scorer or RewardScorer()
    warned = {"user_text": False}

    def reward_func(prompts, completions, **kwargs) -> list[float]:
        golds = kwargs.get("gold")
        users = kwargs.get("user_text")
        if users is None and not warned["user_text"]:
            warned["user_text"] = True
            print("[rl_reward] 警告：数据集缺 user_text 列，改用渲染后 prompt 打分，"
                  "证据忠实度口径会比采样阶段更宽松。", flush=True)
        out = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            text = completion if isinstance(completion, str) else completion[-1]["content"]
            gold = golds[i] if golds is not None else prompt_to_gold.get(prompt, "")
            source = users[i] if users is not None else prompt
            s = scorer.score(source, text, gold)
            if on_score is not None:
                on_score(s, gold, i)
            out.append(s.total)
        return out

    reward_func.__name__ = "mff_rule_reward"
    return reward_func


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="对 jsonl 里的 assistant 金标答案自检打分")
    ap.add_argument("--dataset", default="data/sft/eval.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    scorer = RewardScorer()
    rows = [json.loads(l) for l in open(args.dataset, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    totals = []
    for row in rows:
        msgs = {m["role"]: m["content"] for m in row["messages"]}
        s = scorer.score(msgs["user"], msgs["assistant"], row["meta"]["gold"])
        totals.append(s.total)
    n = len(totals)
    print(f"n={n} mean_reward={sum(totals)/max(n,1):.4f} "
          f"min={min(totals):.4f} max={max(totals):.4f}")
    print("金标答案应接近满分；明显偏低说明奖励分项与数据生成口径不一致。")
