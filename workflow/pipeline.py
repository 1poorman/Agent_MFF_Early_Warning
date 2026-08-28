"""端到端预警工作流编排。

串联 MS1~MS5：
  数据接入+质量管控 -> L1规则 -> L2预测/异常 -> 特征过滤(≥90%)
  -> L3根因推理(LLM+防幻觉) -> 工单/应急/推送 -> 反馈归档

每个节点定义输入/输出/校验/异常处理；节点失败有兜底，保证 7×24 可用。
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from action import (EmergencyPlanner, Feedback, FeedbackStore, Notifier,
                    WorkOrderGenerator)
from context import default_maintenance_log, default_operating_schedule
from detection import RuleEngine
from detection.fast_anomaly import FastAnomalyDetector
from detection.fast_track import FastTrackForecaster
from detection.router import ModelRouter
from perception import DataIngestor, QualityController
from reasoning import KnowledgeGraph, LLMClient, RootCauseReasoner

# 传感器中文名映射（L1 规则 -> 知识图谱传感器名）
RULE_TO_SENSOR = {
    "OUTLET_TEMP_HIGH": "出水温度", "INLET_TEMP_HIGH": "进水温度",
    "DELTA_T_HIGH": "出水温度", "PRESSURE_LOW": "压力", "PRESSURE_HIGH": "压力",
    "PRESSURE_OSC": "压力",
    "FLOW_LOW": "流量", "FLOW_HIGH": "流量", "CONDUCTIVITY_HIGH": "电导率",
    "DEW_MARGIN_LOW": "湿度", "COMBO_LEAK_SUSPECT": "压力",
}


@dataclass
class NodeLog:
    name: str
    status: str          # ok / degraded / failed
    latency_ms: float
    detail: str = ""


@dataclass
class PipelineResult:
    """单窗口端到端处理结果。"""
    window_start: str
    alerted: bool
    work_order: Optional[Dict] = None
    push_count: int = 0
    anomaly_score: float = 0.0
    filtered_ratio: float = 0.0     # 被过滤的正常数据比例
    node_logs: List[NodeLog] = field(default_factory=list)
    total_latency_ms: float = 0.0


class EarlyWarningPipeline:
    """端到端预警流水线。"""

    def __init__(self,
                 fast_models: Optional[Dict[str, FastTrackForecaster]] = None,
                 anomaly_detector: Optional[FastAnomalyDetector] = None,
                 precise_model=None,
                 reasoner: Optional[RootCauseReasoner] = None,
                 use_llm: bool = True,
                 feedback_path: str = "data/feedback/pipeline.jsonl"):
        # 节点组件
        self.ingestor = DataIngestor()
        self.qc = QualityController()
        self.rule_engine = RuleEngine()
        self.fast_models = fast_models or {}
        self.detector = anomaly_detector
        self.router = ModelRouter(self.fast_models, precise_model, precise_available=precise_model is not None)
        # L3 推理（LLM 不可用时 use_llm=False 降级）
        self.use_llm = use_llm
        self.reasoner = reasoner
        if self.reasoner is None and use_llm:
            try:
                self.reasoner = RootCauseReasoner(LLMClient(), KnowledgeGraph())
            except Exception:
                self.use_llm = False
        # 上下文
        self.maint = default_maintenance_log()
        self.sched = default_operating_schedule()
        # 动作
        self.wo_gen = WorkOrderGenerator()
        self.notifier = Notifier()
        self.emergency = EmergencyPlanner()
        self.feedback = FeedbackStore(feedback_path)
        # 统计
        self.stats = {"windows": 0, "alerted": 0, "filtered_points": 0, "total_points": 0}

    # ---------------- 节点实现 ----------------

    def _node_ingest(self, df: pd.DataFrame, log: List[NodeLog]) -> pd.DataFrame:
        """节点1：数据接入 + 质量管控（重采兜底）。"""
        t0 = time.perf_counter()
        clean, report = self.qc.process(df)
        status = "ok" if report.completeness >= 0.99 else "degraded"
        log.append(NodeLog("ingest", status, (time.perf_counter() - t0) * 1000,
                           f"完整度{report.completeness:.1%} 插补{report.missing_filled}"))
        return clean

    def _node_perceive(self, clean: pd.DataFrame, log: List[NodeLog]):
        """节点2：边缘特征提取（L1 + L2 + 异常检测 + 过滤）。"""
        t0 = time.perf_counter()
        # L1 规则（对缺失列容错）
        try:
            l1_alerts = self.rule_engine.evaluate(clean)
        except (KeyError, AttributeError) as e:
            l1_alerts = pd.DataFrame(columns=["timestamp", "rule_id", "level", "message", "value"])
        # L2 异常分（特征列齐全才运行）
        anomaly_score = 0.0
        from detection.fast_anomaly import FEATURE_COLS
        if (self.detector is not None and len(clean) >= self.detector.window
                and all(c in clean.columns for c in FEATURE_COLS)):
            try:
                anomaly_score = float(self.detector.score(clean).iloc[-1])
            except Exception:
                anomaly_score = 0.0
        # 过滤比：无 L1 预警且低异常分 -> 视为正常过滤
        filtered = len(l1_alerts) == 0 and anomaly_score < 0.6
        self.stats["total_points"] += len(clean)
        if filtered:
            self.stats["filtered_points"] += len(clean)
        log.append(NodeLog("perceive", "ok", (time.perf_counter() - t0) * 1000,
                           f"L1预警{len(l1_alerts)}条 异常分{anomaly_score:.2f} {'过滤' if filtered else '上报'}"))
        return l1_alerts, anomaly_score, filtered

    def _node_reason(self, clean, l1_alerts, anomaly_score, log: List[NodeLog]):
        """节点3：L3 根因推理（LLM + 防幻觉，重试≤3）。"""
        t0 = time.perf_counter()
        # 组装异常特征（末值）与传感器
        last = clean.iloc[-1]
        sensors = list({RULE_TO_SENSOR.get(r, "") for r in l1_alerts.rule_id} - {""})
        # features 总是包含全部关键物理量（根因鉴别需要完整上下文，不仅限触发规则的字段）
        sensor_col = {"出水温度": "outlet_temp", "进水温度": "inlet_temp", "压力": "pressure",
                      "流量": "flow_rate", "水箱液位": "tank_level", "电导率": "conductivity",
                      "湿度": "cabinet_humidity"}
        features = {}
        for s, col in sensor_col.items():
            if col in clean.columns:
                features[s] = float(last[col])
        # 补充窗口统计特征（波动/湿度水平是气蚀/泄漏的关键区分）
        if "cabinet_humidity" in clean.columns:
            features["湿度"] = float(clean["cabinet_humidity"].mean())
        if "pressure" in clean.columns:
            features["_press_std"] = float(clean["pressure"].std())
        if "flow_rate" in clean.columns:
            features["_flow_mean"] = float(clean["flow_rate"].mean())
        # 图谱召回只用触发规则对应的传感器（聚焦异常信号）
        sensors = [s for s in sensors if s in sensor_col]
        if not sensors:
            sensors = ["压力", "流量", "湿度"]  # L2 异常上报时无 L1 规则，用全量传感器召回
        report = {
            "features": features,
            "condition": str(last.get("operating_condition", "unknown")),
            "l1_alerts": l1_alerts.to_dict("records"),
            "l2_forecast": {"异常分": f"{anomaly_score:.2f}"},
            "operating_schedule": self.sched.to_prompt_text(),
            "maintenance_log": self.maint.to_prompt_text(days=60),
        }
        if self.use_llm and self.reasoner is not None:
            diag = self.reasoner.diagnose(report=report, sensor_names=sensors)
            status = "ok" if not diag.manual_required else "degraded"
        else:
            # 降级：图谱投票 + 数值鉴别
            diag = self._fallback_diagnose(sensors, features)
            status = "degraded"
        log.append(NodeLog("reason", status, (time.perf_counter() - t0) * 1000,
                           f"根因={diag.root_cause} 置信度{diag.confidence:.2f}"))
        return diag

    def _fallback_diagnose(self, sensors, features: Optional[Dict[str, float]] = None):
        """LLM 不可用时的兜底诊断：图谱投票 + 数值鉴别。

        图谱平局时用特征数值做确定性鉴别（对齐故障物理特征）：
        - 流量<80%额定 -> 过滤器堵塞；压力/流量震荡 -> 水泵气蚀
        - 压力低+湿度高/液位降 -> 管道泄漏；温差大但流量正常 -> 线圈结垢
        """
        from reasoning.root_cause import DiagnosisResult
        from reasoning.anti_hallucination import CheckResult
        kg = self.reasoner.kg if self.reasoner else KnowledgeGraph()
        f = features or {}

        rc = None
        flow, press = f.get("流量"), f.get("压力")
        hum, level = f.get("湿度"), f.get("水箱液位")
        out_t, in_t = f.get("出水温度"), f.get("进水温度")
        hum_delta = f.get("_hum_delta", 0.0)
        # 数值鉴别（特征组合越特异越优先；不依赖未触发规则的字段）
        delta_t = (out_t - in_t) if (out_t is not None and in_t is not None) else None
        press_std = f.get("_press_std", 0.0)
        # 判定优先级：越特异的统计组合越优先
        # 湿度是气蚀/泄漏的决定性区分特征（泄漏->湿度升高，气蚀->湿度正常）
        # 湿度上升趋势>4%RH 是泄漏早期铁证（绝对值未达70也判泄漏；堵塞不改变湿度）
        # 压力水平区分堵塞/结垢：堵塞(过滤器阻抗)->流量低压力低；结垢(线圈阻抗)->流量低压力高
        if hum is not None and hum > 70 and press is not None and press < 150:
            rc = "管道泄漏"                                    # 压力低+湿度显著升高（最特异）
        elif hum is not None and hum > 70:
            rc = "管道泄漏"                                    # 湿度显著升高（泄漏水汽）
        elif hum_delta > 4.0 and (flow is not None and flow < 7.8 or press is not None and press < 240):
            rc = "管道泄漏"                                    # 湿度上升趋势+流量/压力下降（泄漏早期）
        elif press_std > 3.0 and (hum is None or hum <= 65):
            rc = "水泵气蚀"                                    # 压力去趋势震荡显著且湿度正常（气蚀脉动）
        elif flow is not None and flow < 6.4 and press is not None and press > 230:
            rc = "线圈结垢"                                    # 流量低+压力偏高（线圈热阻增大）
        elif flow is not None and flow < 6.4 and (hum is None or hum <= 70) and press_std < 3.0:
            rc = "过滤器堵塞"                                  # 流量持续不足+压力偏低+无震荡（过滤器阻抗升高）
        elif delta_t is not None and delta_t > 20 and press is not None and press > 200:
            rc = "线圈结垢"                                    # 温差大且压力偏高（热阻增大，管网未堵）
        elif press is not None and flow is not None and 150 <= press < 200 and flow < 7.5:
            rc = "水泵气蚀"                                    # 压力/流量中位震荡
        if rc is None:
            cands = kg.faults_for_sensors(sensors)
            rc = kg.nodes[cands[0][0]].name if cands else "未知故障"
        return DiagnosisResult(rc, 0.6, sensors, kg.actions_for_fault(rc),
                               check=CheckResult(True, True), manual_required=False)

    def _node_act(self, diag, log: List[NodeLog]):
        """节点4：工单生成 + 应急联动 + 分级推送。"""
        t0 = time.perf_counter()
        wo = self.wo_gen.generate(diag, features_text="；".join(diag.evidence[:2]))
        self.emergency.attach(wo)
        recs = self.notifier.push(wo)
        log.append(NodeLog("act", "ok", (time.perf_counter() - t0) * 1000,
                           f"工单{wo.order_id} {wo.level} 推送{len(recs)}条"))
        return wo, recs

    # ---------------- 主流程 ----------------

    def process_window(self, df: pd.DataFrame) -> PipelineResult:
        """处理一个数据窗口，端到端贯通五节点。"""
        t_start = time.perf_counter()
        logs: List[NodeLog] = []
        self.stats["windows"] += 1

        clean = self._node_ingest(df, logs)
        l1_alerts, anomaly_score, filtered = self._node_perceive(clean, logs)

        result = PipelineResult(
            window_start=str(clean["timestamp"].iloc[0]),
            alerted=False, anomaly_score=anomaly_score,
            filtered_ratio=self.stats["filtered_points"] / max(self.stats["total_points"], 1),
            node_logs=logs,
        )

        # 过滤：正常数据直接归档，不进入 L3
        if filtered:
            result.total_latency_ms = (time.perf_counter() - t_start) * 1000
            return result

        # 异常 -> L3 推理 -> 工单 -> 推送
        diag = self._node_reason(clean, l1_alerts, anomaly_score, logs)
        wo, recs = self._node_act(diag, logs)

        result.alerted = True
        result.work_order = wo.to_dict()
        result.push_count = len(recs)
        self.stats["alerted"] += 1
        result.total_latency_ms = (time.perf_counter() - t_start) * 1000
        return result

    def submit_feedback(self, order_id: str, actual_root_cause: str,
                        is_true_fault: bool, handling_time_min: float, effect: str,
                        diagnosis_snapshot: Optional[Dict] = None):
        """节点5：反馈归档。"""
        self.feedback.archive(
            Feedback(order_id, actual_root_cause, is_true_fault, handling_time_min, effect),
            diagnosis_snapshot)

    def get_stats(self) -> Dict:
        return {**self.stats,
                "filtered_ratio": self.stats["filtered_points"] / max(self.stats["total_points"], 1)}
