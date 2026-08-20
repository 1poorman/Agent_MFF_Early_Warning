"""四大智能体实现（基于共享服务层封装）。

1. 数据管理智能体 DataManagementAgent    —— 传感器数据接收/采集接口/预处理
2. 预警分析智能体 WarningAnalysisAgent    —— L1~L3 多级预警与根因诊断
3. 故障处置智能体 FaultHandlingAgent      —— 工单生成与预警通知
4. 持续优化智能体 ContinuousOptimizerAgent —— 反馈归档与知识库持续更新

四大智能体串联即完整工作流：
  数据管理 -> 预警分析 -> 故障处置 -> 持续优化
"""

import time
from dataclasses import asdict
from typing import Dict, List, Optional

import pandas as pd

from .service import AgentService


# ---------------------------------------------------------------------------
# 1. 数据管理智能体
# ---------------------------------------------------------------------------

class DataManagementAgent:
    """负责传感器数据接收、预留传感器传输接口、数据预处理。

    输出 L1/L2 可直接使用的规整数据格式（含质量报告与字段 schema）。
    """

    def __init__(self, svc: AgentService):
        self.svc = svc

    # ---- 预留传感器传输接口 ----

    def collect(self, duration: int = 300, fault: Optional[str] = None,
                fault_start: int = 120, severity: float = 0.9) -> Dict:
        """传感器数据采集接口（预留：Modbus/OPC UA/MQTT/RTSP 真实接入）。

        当前由物理机理仿真器提供数据源；真实部署时替换为协议适配层，
        返回格式保持不变（L1/L2 数据契约）。
        """
        from simulator import DataSimulator, FaultSpec, SimConfig
        faults = [FaultSpec(name=fault, start=fault_start, ramp=300, severity=severity)] if fault else []
        sim = DataSimulator(config=SimConfig(seed=42), faults=faults)
        df = sim.run(duration)
        return self.ingest(df.to_dict("records"))

    # ---- 数据接收与预处理 ----

    def ingest(self, records: List[Dict]) -> Dict:
        """接收原始传感器记录 -> 质量管控（对齐/插补/剔除）-> 规整数据。"""
        df = pd.DataFrame(records)
        if "timestamp" not in df.columns:
            raise ValueError("缺少 timestamp 字段")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        clean, report = self.svc.qc.process(df)
        return {
            "agent": "data_manager",
            "records": clean.astype(object).where(clean.notna(), None).to_dict("records"),
            "quality": {
                "total_in": report.total_in,
                "total_out": report.total_out,
                "duplicates_removed": report.duplicates_removed,
                "gaps_filled": report.gaps_filled,
                "missing_filled": report.missing_filled,
                "outliers_removed": report.outliers_removed,
                "completeness": round(report.completeness, 4),
            },
            "schema": {
                "timestamp": "str, 1s 连续递增",
                "numeric_fields": ["inlet_temp", "outlet_temp", "pressure", "flow_rate",
                                   "flow_velocity", "tank_level", "conductivity",
                                   "cabinet_temp", "cabinet_humidity", "furnace_temp",
                                   "electric_power", "electric_current"],
                "context_fields": ["operating_condition", "fault_label"],
                "precision": "1 位小数（L1/L2 直接可用）",
            },
        }

    def schema(self) -> Dict:
        """返回 L1/L2 可直接使用的数据格式说明。"""
        return {
            "agent": "data_manager",
            "contract": {
                "timestamp": "str %Y-%m-%d %H:%M:%S，1s 连续递增，无丢包/乱序/空值",
                "fields": {
                    "inlet_temp": "float ℃ 冷却水进水温度",
                    "outlet_temp": "float ℃ 冷却水出水温度",
                    "pressure": "float kPa 管道压力",
                    "flow_rate": "float L/s 冷却水流量",
                    "flow_velocity": "float m/s 管道流速",
                    "tank_level": "float cm 水箱液位",
                    "conductivity": "float µS/cm 电导率",
                    "cabinet_temp": "float ℃ 电气柜表面温度",
                    "cabinet_humidity": "float %RH 电气柜附近湿度",
                    "furnace_temp": "float ℃ 炉内温度",
                    "electric_power": "float kW 电功率",
                    "electric_current": "float A 电流",
                    "operating_condition": "str 工况上下文（startup/melting/holding/tapping/idle）",
                },
            },
        }


# ---------------------------------------------------------------------------
# 2. 预警分析智能体
# ---------------------------------------------------------------------------

class WarningAnalysisAgent:
    """L1~L3 多级预警与根因诊断。

    输入：数据管理智能体返回的规整数据（records）。
    上下文：知识图谱、知识库（维修工单）、工况运行表自动注入。
    输出：分级预警信息 + 根因诊断（json）。
    """

    def __init__(self, svc: AgentService):
        self.svc = svc

    def analyze(self, records: List[Dict]) -> Dict:
        """执行 L1 规则 -> L2 异常/趋势 -> L3 根因诊断 的多级分析。"""
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        last = df.iloc[-1]

        # L1 规则预警
        l1_alerts = self.svc.rule_engine.evaluate(df)
        l1 = l1_alerts.to_dict("records")

        # L2 异常检测 + 趋势预测
        l2 = {"anomaly_score": 0.0, "anomaly_triggered": False, "trend_exceed": None}
        det = self.svc.pipeline.detector
        if det is not None and len(df) >= det.window:
            try:
                score = float(det.score(df).iloc[-1])
                l2["anomaly_score"] = round(score, 3)
                l2["anomaly_triggered"] = score > det.threshold
            except Exception:
                pass
        fm = self.svc.pipeline.fast_models.get("outlet_temp")
        if fm is not None and len(df) >= 120:
            try:
                eta = fm.predict_exceedance(df["outlet_temp"], 55.0)
                if eta is not None:
                    l2["trend_exceed"] = {"eta_s": round(eta, 1),
                                          "message": f"预测 {eta/60:.1f}min 后出水温度越限 55℃"}
            except Exception:
                pass

        # L3 根因诊断（L1 触发或 L2 异常超阈时）
        triggered = bool(l1) or l2["anomaly_triggered"]
        diagnosis = None
        if triggered:
            from workflow.pipeline import RULE_TO_SENSOR
            sensors = list({RULE_TO_SENSOR.get(a["rule_id"], "") for a in l1} - {""}) or \
                      ["出水温度", "压力", "流量", "湿度"]
            col_map = {"出水温度": "outlet_temp", "进水温度": "inlet_temp", "压力": "pressure",
                       "流量": "flow_rate", "水箱液位": "tank_level", "湿度": "cabinet_humidity",
                       "电导率": "conductivity"}
            features = {s: float(last[c]) for s, c in col_map.items() if c in df.columns}
            # 窗口统计鉴别特征（气蚀/泄漏/堵塞/结垢区分的决定性依据，注入 LLM 提示）
            # 取最近 120s 尾段统计，避免故障前正常段稀释（爬升期整窗均值会失真）
            tail = df.iloc[-120:] if len(df) >= 120 else df
            stats = {}
            if "pressure" in tail.columns:
                stats["压力波动幅度_std_kPa"] = round(self._detrended_std(tail["pressure"]), 2)
                stats["压力均值_kPa"] = round(float(tail["pressure"].mean()), 1)
            if "cabinet_humidity" in tail.columns:
                stats["湿度均值_pctRH"] = round(float(tail["cabinet_humidity"].mean()), 1)
                features["湿度"] = stats["湿度均值_pctRH"]
            if "flow_rate" in tail.columns:
                stats["流量均值_Lps"] = round(float(tail["flow_rate"].mean()), 2)
            if "tank_level" in tail.columns:
                stats["液位均值_cm"] = round(float(tail["tank_level"].mean()), 1)
            features["_press_std"] = stats.get("压力波动幅度_std_kPa", 0.0)
            # 统计预鉴别：依据 stats 计算倾向根因，并入 LLM 候选集（防图谱召回漏检）
            extra_cands = self._stat_precheck(features, stats)
            try:
                diag = self.svc.diagnose(
                    features, str(last.get("operating_condition", "unknown")), sensors,
                    l1_alerts=l1, l2_forecast={"anomaly_score": l2["anomaly_score"]},
                    stats=stats, extra_candidates=extra_cands)
            except Exception:
                diag = self.svc.pipeline._fallback_diagnose(sensors, features)
            diagnosis = diag.to_dict()

        # 汇总预警级别
        level = "none"
        if diagnosis is not None:
            level = diagnosis.get("level", "orange")
            if diagnosis.get("manual_required"):
                level = "yellow"
        elif l1 or l2["anomaly_triggered"]:
            level = "yellow"

        return {
            "agent": "warning_analyzer",
            "timestamp": str(last["timestamp"]),
            "condition": str(last.get("operating_condition", "unknown")),
            "level": level,                # none/yellow/orange/red
            "l1": {"triggered": bool(l1), "alerts": l1},
            "l2": l2,
            "l3": diagnosis,               # 根因诊断（json），未触发为 null
            "context_used": {
                "knowledge_graph": "五域知识图谱（设备-部件-传感器-故障-处置）",
                "maintenance_log": "近期维修工单（知识库）",
                "operating_schedule": "工况运行表",
            },
        }

    @staticmethod
    def _detrended_std(s) -> float:
        """去趋势标准差：剔除线性趋势后残差的 std，专测震荡幅度（爬升趋势不计入）。"""
        import numpy as np
        v = s.to_numpy(dtype=float)
        if len(v) < 10:
            return float(np.std(v))
        t = np.arange(len(v))
        slope, intercept = np.polyfit(t, v, 1)
        resid = v - (slope * t + intercept)
        return float(np.std(resid))

    @staticmethod
    def _stat_precheck(features: Dict, stats: Dict) -> List[str]:
        """统计预鉴别：返回倾向根因列表（与提示词判定规则一致）。"""
        out = []
        hum = float(stats.get("湿度均值_pctRH", 50.0))
        press = float(stats.get("压力均值_kPa", features.get("压力", 240.0)))
        press_std = float(stats.get("压力波动幅度_std_kPa", 0.0))
        flow = float(stats.get("流量均值_Lps", features.get("流量", 8.0)))
        if hum > 70:
            out.append("管道泄漏")
        elif press_std > 3.0:
            out.append("水泵气蚀")
        if flow < 6.4:
            out.append("线圈结垢" if press > 230 else "过滤器堵塞")
        return out


# ---------------------------------------------------------------------------
# 3. 故障处置智能体
# ---------------------------------------------------------------------------

class FaultHandlingAgent:
    """接收预警分析智能体输出 -> 工单生成 + 应急预案联动 + 分级预警通知。"""

    def __init__(self, svc: AgentService):
        self.svc = svc

    def handle(self, analysis: Dict) -> Dict:
        """由预警分析结果生成工单并推送。

        analysis: 预警分析智能体的返回（含 l3 诊断）；无诊断时仅记录预警。
        """
        s = self.svc
        diagnosis = analysis.get("l3")
        if diagnosis is None:
            return {
                "agent": "fault_handler",
                "handled": False,
                "reason": "无 L3 诊断（预警未升级或正常数据），无需生成工单",
                "notifications": [],
            }

        from reasoning.root_cause import DiagnosisResult
        from reasoning.anti_hallucination import CheckResult
        diag = DiagnosisResult(
            root_cause=diagnosis["root_cause"], confidence=diagnosis["confidence"],
            evidence=diagnosis.get("evidence", []), sop=diagnosis.get("sop", []),
            level=analysis.get("level", "orange"),
            check=CheckResult(physics_ok=diagnosis["hallucination_check"]["physics"],
                              kg_ok=diagnosis["hallucination_check"]["kg"]),
            manual_required=diagnosis.get("manual_required", False),
        )
        features_text = "；".join(diagnosis.get("evidence", [])[:2]) or \
            f"L1预警{analysis['l1']['triggered']}/异常分{analysis['l2']['anomaly_score']}"
        wo = s.wo_gen.generate(diag, features_text=features_text,
                               trigger_time=analysis.get("timestamp"))
        plan = s.emergency.attach(wo)
        recs = s.notifier.push(wo)
        data = wo.to_dict()
        data["agent"] = "fault_handler"
        data["handled"] = True
        data["emergency_plan"] = (
            {"plan_id": plan.plan_id, "name": plan.name,
             "risk": plan.risk, "steps": plan.steps, "forbidden": plan.forbidden}
            if plan else None)
        data["notifications"] = [vars(r) for r in recs]
        data["notification_summary"] = {
            "total": len(recs),
            "channels": sorted({r.channel for r in recs}),
            "receivers": sorted({r.receiver for r in recs}),
        }
        return data


# ---------------------------------------------------------------------------
# 4. 持续优化智能体
# ---------------------------------------------------------------------------

class ContinuousOptimizerAgent:
    """基于处置反馈持续迭代：反馈归档 -> 知识库更新 -> 微调触发。"""

    def __init__(self, svc: AgentService):
        self.svc = svc

    def feedback(self, order_id: str, actual_root_cause: str, is_true_fault: bool,
                 handling_time_min: float, effect: str,
                 work_order: Optional[Dict] = None) -> Dict:
        """接收故障处置反馈，归档训练样本并按需更新知识库。"""
        s = self.svc
        snapshot = {"work_order": work_order} if work_order else None
        s.submit_feedback(order_id, actual_root_cause, is_true_fault,
                          handling_time_min, effect)
        out = {
            "agent": "continuous_optimizer",
            "archived": True,
            "order_id": order_id,
            "knowledge_updated": False,
            "stats": s.feedback_store.stats(),
        }
        # 真实故障且根因与诊断不一致 -> 知识库/工单台账更新（新增维修工单）
        if is_true_fault and work_order and actual_root_cause != work_order.get("root_cause"):
            self._update_maintenance(order_id, actual_root_cause)
            out["knowledge_updated"] = True
            out["knowledge_update"] = f"新增维修工单记录（修正根因: {actual_root_cause}）"
        return out

    def _update_maintenance(self, order_id: str, component: str):
        from context.maintenance import WorkOrder
        self.svc.maint.orders.append(
            WorkOrder(order_id, pd.Timestamp.now().strftime("%Y-%m-%d"),
                      component, "故障处置（反馈驱动记录）", "持续优化智能体自动归档"))

    def status(self) -> Dict:
        s = self.svc
        return {
            "agent": "continuous_optimizer",
            "feedback_stats": s.feedback_store.stats(),
            "retrain_due": s.feedback_store.should_retrain(min_samples=5),
            "knowledge_base": {
                "maintenance_orders": len(s.maint.orders),
                "recent_30d": len(s.maint.recent(days=30)),
                "operating_phases": len(s.sched.phases),
            },
        }

    def update_knowledge(self, order: Dict) -> Dict:
        """手动更新知识库（新增维修工单/工况记录）。"""
        from context.maintenance import WorkOrder
        wo = WorkOrder(
            order_id=order.get("order_id", f"WO-MAN-{int(time.time())}"),
            date=order.get("date", pd.Timestamp.now().strftime("%Y-%m-%d")),
            component=order.get("component", ""),
            action=order.get("action", ""),
            note=order.get("note", ""),
        )
        self.svc.maint.orders.append(wo)
        return {"agent": "continuous_optimizer", "knowledge_updated": True,
                "added": asdict(wo), "total_orders": len(self.svc.maint.orders)}
