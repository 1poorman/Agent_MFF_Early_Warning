"""MS5 验收测试：工单生成 + 分级推送 + 应急联动 + 反馈闭环。

验收标准（design/MILESTONES.md）：
- 工单字段完整性 100%（7 类字段齐全）
- 分级推送正确性：级别->渠道->对象映射 100% 正确
- 应急预案联动：红色预警自动挂载预案
- 反馈闭环：反馈入库 + 触发微调标记
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from action import (
    EmergencyPlanner, Feedback, FeedbackStore, Notifier, WorkOrderGenerator,
)
from reasoning.anti_hallucination import CheckResult
from reasoning.root_cause import DiagnosisResult

REQUIRED_FIELDS = ["order_id", "level", "trigger_time", "features", "root_cause",
                   "confidence", "evidence", "hallucination_check", "sop", "spare_parts"]


def make_diag(root_cause, confidence=0.95, level="red", manual=False):
    return DiagnosisResult(
        root_cause=root_cause, confidence=confidence,
        evidence=[f"{root_cause}特征证据1", "证据2"],
        sop=["步骤1", "步骤2"], level=level,
        check=CheckResult(physics_ok=True, kg_ok=True),
        manual_required=manual,
    )


def test_work_order_fields():
    gen = WorkOrderGenerator()
    wo = gen.generate(make_diag("管道泄漏"), features_text="压力140kPa且湿度74%RH")
    d = wo.to_dict()
    missing = [f for f in REQUIRED_FIELDS if f not in d or d[f] in (None, "", [])]
    # spare_parts 允许为空但字段需在
    missing = [f for f in missing if f != "spare_parts"]
    assert not missing, f"工单字段缺失: {missing}"
    assert "DN50密封圈" in wo.spare_parts, "备件映射缺失"
    assert wo.hallucination_check["physics"] and wo.hallucination_check["kg"]
    print(f"[PASS] 工单字段完整性: {len(REQUIRED_FIELDS)} 类字段齐全, 备件={wo.spare_parts}")


def test_notify_routing():
    notifier = Notifier()
    gen = WorkOrderGenerator()
    # 红/橙/黄三级
    cases = [("red", "管道泄漏", {"sound_light_alarm", "sms", "phone"}, {"厂长", "安全主管", "运维班长"}),
             ("orange", "过滤器堵塞", {"mobile_app", "sms"}, {"运维班长", "维修工"}),
             ("yellow", "电气柜凝露", {"mobile_app"}, {"值班运维"})]
    for level, fault, exp_ch, exp_rcv in cases:
        wo = gen.generate(make_diag(fault, level=level))
        recs = notifier.push(wo)
        ch = {r.channel for r in recs}
        rcv = {r.receiver for r in recs}
        assert ch == exp_ch, f"{level} 渠道错误: {ch}"
        assert rcv == exp_rcv, f"{level} 接收人错误: {rcv}"
        print(f"  {level}: 渠道{sorted(ch)} 对象{sorted(rcv)} ✔")
    print("[PASS] 分级推送: 红/橙/黄 渠道与对象映射 100% 正确")


def test_emergency_attach():
    gen = WorkOrderGenerator()
    planner = EmergencyPlanner()
    # 红色泄漏 -> 挂载漏水入铁水预案
    wo = gen.generate(make_diag("管道泄漏", level="red"))
    plan = planner.attach(wo)
    assert plan is not None and plan.plan_id == "EP-002", "红色泄漏未挂载预案"
    assert any("倾炉" in s for s in wo.sop), "预案步骤未并入 SOP"
    assert any("严禁" in f for f in plan.forbidden), "预案禁止事项缺失"
    # 橙色 -> 不挂载
    wo2 = gen.generate(make_diag("过滤器堵塞", level="orange"))
    assert planner.attach(wo2) is None, "非红色误挂载预案"
    print(f"[PASS] 应急联动: 红色泄漏自动挂载 {plan.plan_id}({plan.name})，橙色不挂载")


def test_feedback_loop(tmp_path="data/feedback"):
    import shutil
    shutil.rmtree(tmp_path, ignore_errors=True)
    store = FeedbackStore(f"{tmp_path}/feedback.jsonl")
    # 5 条真实故障反馈触发微调标记
    for i in range(5):
        store.archive(Feedback(f"WO-20260820-{i:04d}", "管道泄漏", True, 30.0, "修复成功"))
    store.archive(Feedback("WO-20260820-0099", "误报", False, 5.0, "无需处理"))
    stats = store.stats()
    assert stats["total_feedback"] == 6, "反馈未全部入库"
    assert stats["true_faults"] == 5 and stats["false_alarms"] == 1
    assert store.should_retrain(min_samples=5), "未触发微调标记"
    # 持久化校验
    lines = open(store.store_path, encoding="utf-8").read().strip().split("\n")
    assert len(lines) == 6, "JSONL 持久化条数错误"
    store.mark_retrained()
    assert not store.should_retrain(), "微调标记未清除"
    print(f"[PASS] 反馈闭环: 6 条归档(5真1误报), 微调标记触发并清除, JSONL 持久化 ✔")


def test_end_to_end():
    """端到端：诊断->工单->应急->推送->反馈 全链路。"""
    gen, notifier, planner = WorkOrderGenerator(), Notifier(), EmergencyPlanner()
    store = FeedbackStore("data/feedback/e2e.jsonl")
    diag = make_diag("管道泄漏", confidence=0.96, level="red")
    wo = gen.generate(diag, features_text="压力140kPa且湿度74%RH")
    planner.attach(wo)
    recs = notifier.push(wo)
    store.archive(Feedback(wo.order_id, "管道泄漏", True, 25.0, "漏点补焊完成"), diag.to_dict())
    assert wo.level == "red" and len(recs) == 9  # 3渠道x3人
    assert store.stats()["total_feedback"] >= 1
    print(f"[PASS] 端到端: 红色泄漏工单 {wo.order_id} -> 预案+9条推送 -> 反馈归档 ✔")
    print("\n工单示例:\n" + wo.to_json()[:400] + "...")


def main():
    test_work_order_fields()
    test_notify_routing()
    test_emergency_attach()
    test_feedback_loop()
    test_end_to_end()
    print("\nMS5 全部验收通过 ✔")


if __name__ == "__main__":
    main()
