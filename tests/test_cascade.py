"""L4 级联集成测试（docs/SFT_TEST_PLAN.md 第六节，MS6）。

级联架构（蓝图 4.2 + 决策 B）：2B SFT 前置（**关仲裁**）+ 27B 兜底，
升级触发**仅**依据可验证信号（JSON 解析 / 防幻觉校验 / 端点可用性），
**不依赖模型自报置信度**（见 design/SFT_EVAL_REPORT.md 第六节）。

用例映射（相较测试方案原表，L4.2 按决策 B 修订为"低置信不升级"）：
  L4.1 高置信直出        —— 明确故障，2B 直出，不请求 27B
  L4.2 低置信不升级      —— 边界低置信但校验通过，2B 直出（决策 B：不因低置信升级）
  L4.3 防幻觉失败升级    —— 前置图谱外根因，重试耗尽后升级 27B
  L4.4 JSON 解析失败升级 —— 前置非 JSON 输出，升级 27B
  L4.5 端点异常升级      —— 前置端点不可用，升级 27B
  L4.6 双级失败兜底      —— 前置+兜底均不可用，manual_required，不崩溃
  L4.7 升级率 ≤40%       —— 批量样本升级比例达标
  L4.8 CascadeGate 单元  —— 判据正确且 conf 不参与

不依赖任何在线端点（全用 FakeLLM 打桩）；末尾附本地 3761 的可选真机冒烟。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoning.knowledge_graph import KnowledgeGraph
from reasoning.root_cause import RootCauseReasoner
from reasoning.confidence import CascadeGate

KG = KnowledgeGraph()


class FakeLLM:
    """打桩 LLM：按脚本逐次返回；脚本项为 Exception 时抛出（模拟端点异常）。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def chat(self, prompt, **kw):
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def _json(rc, conf=0.88, evidence=None, sop=None):
    return json.dumps({
        "root_cause": rc, "confidence": conf,
        "evidence": evidence or ["湿度上升"], "sop": sop or ["测漏仪检查管路"],
    }, ensure_ascii=False)


def _cascade(front_llm, fallback_llm):
    return RootCauseReasoner(llm=front_llm, kg=KG, fallback_llm=fallback_llm, arbitrate=True)


# 明确故障样本（features 物理合法，湿度铁证；stats 触发泄漏强先验）
LEAK_REPORT = {
    "features": {"湿度": 74.0, "压力": 140.0},
    "condition": "holding",
    "stats": {"湿度均值_pctRH": 74.0, "湿度上升量_pctRH": 6.0},
    "extra_candidates": ["管道泄漏"],
}
LEAK_SENSORS = ["湿度", "压力"]


def test_L4_8_cascade_gate_unit():
    g = CascadeGate()
    assert g.decide(parse_ok=True, check_passed=True).upgrade is False
    assert g.decide(parse_ok=True, check_passed=True, endpoint_error="conn").upgrade is True
    assert g.decide(parse_ok=False, check_passed=True).upgrade is True
    assert g.decide(parse_ok=True, check_passed=False).upgrade is True
    # 决策 B：conf 不是 decide() 的入参（签名层面杜绝按置信度路由）
    import inspect
    params = set(inspect.signature(g.decide).parameters)
    assert "confidence" not in params and "conf" not in params, \
        f"CascadeGate.decide 不得含置信度入参: {params}"
    print("[PASS] L4.8 CascadeGate: 三类可验证信号触发升级, 置信度不参与路由（决策 B）")


def test_L4_1_high_conf_direct():
    front = FakeLLM([_json("管道泄漏", 0.88)])
    fb = FakeLLM([_json("管道泄漏", 0.9)])
    r = _cascade(front, fb).diagnose(report=LEAK_REPORT, sensor_names=LEAK_SENSORS)
    assert r.tier == "front_2b" and not r.upgraded, f"应 2B 直出: {r.tier}/{r.upgraded}"
    assert fb.calls == 0, f"高置信直出不得请求 27B, 实际 {fb.calls} 次"
    assert r.root_cause == "管道泄漏"
    print(f"[PASS] L4.1 高置信直出: tier={r.tier} 27B请求={fb.calls} 根因={r.root_cause}")


def test_L4_2_low_conf_no_upgrade():
    # 边界样本、模型自报低置信（0.10），但 JSON/防幻觉均通过 -> 决策 B：不升级
    front = FakeLLM([_json("管道泄漏", 0.10)])
    fb = FakeLLM([_json("管道泄漏", 0.9)])
    r = _cascade(front, fb).diagnose(report=LEAK_REPORT, sensor_names=LEAK_SENSORS)
    assert r.tier == "front_2b" and not r.upgraded, \
        f"决策 B：低置信不应升级, 实际 tier={r.tier} upgraded={r.upgraded}"
    assert fb.calls == 0, f"低置信不得升级 27B, 实际 {fb.calls} 次"
    assert abs(r.confidence - 0.10) < 1e-6, f"关仲裁应保留原始置信度: {r.confidence}"
    print(f"[PASS] L4.2 低置信不升级（决策 B）: tier={r.tier} conf={r.confidence} 27B请求={fb.calls}")


def test_L4_3_antihall_upgrade():
    # 前置返回图谱外根因 -> 校验失败重试耗尽 -> 升级；兜底返回合法根因
    front = FakeLLM([_json("锅炉爆炸", 0.95)])          # 不在知识图谱故障域
    fb = FakeLLM([_json("管道泄漏", 0.9)])
    r = _cascade(front, fb).diagnose(report=LEAK_REPORT, sensor_names=LEAK_SENSORS)
    assert r.upgraded and r.tier == "fallback_27b", f"应升级 27B: {r.tier}/{r.upgraded}"
    assert front.calls == 3, f"前置应重试至 MAX_RETRY: {front.calls}"
    assert fb.calls == 1 and r.root_cause == "管道泄漏"
    assert "防幻觉" in r.upgrade_reason, r.upgrade_reason
    print(f"[PASS] L4.3 防幻觉失败升级: 前置重试={front.calls} 升级原因={r.upgrade_reason}")


def test_L4_4_json_fail_upgrade():
    front = FakeLLM(["这不是JSON，模型胡言乱语"])
    fb = FakeLLM([_json("管道泄漏", 0.9)])
    r = _cascade(front, fb).diagnose(report=LEAK_REPORT, sensor_names=LEAK_SENSORS)
    assert r.upgraded and r.tier == "fallback_27b", f"JSON 失败应升级: {r.tier}"
    assert r.root_cause == "管道泄漏"
    print(f"[PASS] L4.4 JSON 解析失败升级: 升级原因={r.upgrade_reason}")


def test_L4_5_endpoint_down_upgrade():
    front = FakeLLM([ConnectionError("2B endpoint refused")])
    fb = FakeLLM([_json("管道泄漏", 0.9)])
    r = _cascade(front, fb).diagnose(report=LEAK_REPORT, sensor_names=LEAK_SENSORS)
    assert r.upgraded and r.tier == "fallback_27b", f"端点异常应升级: {r.tier}"
    assert front.calls == 1, f"端点异常应立即升级（不重试）: {front.calls}"
    assert fb.calls == 1 and r.root_cause == "管道泄漏"
    assert "端点异常" in r.upgrade_reason
    print(f"[PASS] L4.5 端点异常升级: 前置调用={front.calls} 升级原因={r.upgrade_reason}")


def test_L4_6_both_fail_manual():
    front = FakeLLM([ConnectionError("2B down")])
    fb = FakeLLM([TimeoutError("27B timeout")])
    r = _cascade(front, fb).diagnose(report=LEAK_REPORT, sensor_names=LEAK_SENSORS)
    assert r.upgraded and r.tier == "fallback_27b"
    assert r.manual_required, "双级失败应置 manual_required"
    # 不崩溃，仍给出图谱兜底根因（可序列化）
    d = r.to_dict()
    assert d["manual_required"] and d["root_cause"], d
    print(f"[PASS] L4.6 双级失败兜底: manual_required={r.manual_required} 根因={r.root_cause}(图谱兜底)")


def test_L4_7_upgrade_rate():
    # 10 个样本：8 个前置通过（直出），2 个触发升级 -> 升级率 20% ≤ 40%
    scripts = [_json("管道泄漏", 0.88)] * 8 + ["非JSON", _json("锅炉爆炸", 0.9)]
    upgraded = 0
    for s in scripts:
        front = FakeLLM([s])
        fb = FakeLLM([_json("管道泄漏", 0.9)])
        r = _cascade(front, fb).diagnose(report=LEAK_REPORT, sensor_names=LEAK_SENSORS)
        upgraded += int(r.upgraded)
    rate = upgraded / len(scripts)
    assert rate <= 0.40, f"升级率 {rate:.0%} > 40%"
    print(f"[PASS] L4.7 升级率: {upgraded}/{len(scripts)} = {rate:.0%} (≤40%)")


def test_live_smoke():
    """可选：本地 3761（2B SFT）真机冒烟——明确故障走前置直出。端点不可用则跳过。"""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:3761/v1/models", timeout=3)
    except Exception as e:
        print(f"[SKIP] L4 真机冒烟: 本地 3761 不可用({type(e).__name__})")
        return
    from reasoning.llm_client import LLMClient
    from config import get_settings
    cfg = get_settings().llm.to_client_dict()
    front = LLMClient.front(config=cfg)
    # 兜底端点在本冒烟中不触发（明确故障前置直出）；给同一前置占位即可
    reasoner = RootCauseReasoner(llm=front, kg=KG, fallback_llm=front, arbitrate=True)
    r = reasoner.diagnose(report=LEAK_REPORT, sensor_names=LEAK_SENSORS)
    assert r.parse_ok, "2B 输出应可解析"
    assert r.tier == "front_2b" and not r.upgraded, \
        f"明确故障应前置直出: tier={r.tier} upgraded={r.upgraded} reason={r.upgrade_reason}"
    print(f"[PASS] L4 真机冒烟(3761): tier={r.tier} 根因={r.root_cause} conf={r.confidence:.2f}")


def main():
    test_L4_8_cascade_gate_unit()
    test_L4_1_high_conf_direct()
    test_L4_2_low_conf_no_upgrade()
    test_L4_3_antihall_upgrade()
    test_L4_4_json_fail_upgrade()
    test_L4_5_endpoint_down_upgrade()
    test_L4_6_both_fail_manual()
    test_L4_7_upgrade_rate()
    test_live_smoke()
    print("\nL4 级联集成测试全部通过 ✔")


if __name__ == "__main__":
    main()
