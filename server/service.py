"""共享服务层：加载模型、上下文、工作流，供 API 与 MCP 复用。

单例模式，进程启动时初始化一次（模型加载较重）。
"""

import threading
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from action import EmergencyPlanner, Feedback, FeedbackStore, Notifier, WorkOrderGenerator
from context import default_maintenance_log, default_operating_schedule
from detection import RuleEngine
from detection.fast_anomaly import FastAnomalyDetector
from detection.fast_track import FastTrackForecaster
from perception import DataIngestor, QualityController
from reasoning import KnowledgeGraph, LLMClient, RootCauseReasoner
from workflow import EarlyWarningPipeline

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FAST_COLS = ["outlet_temp", "flow_rate", "pressure"]


class AgentService:
    """预警智能体共享服务（单例）。"""

    _instance: Optional["AgentService"] = None
    _lock = threading.Lock()

    def __init__(self, use_llm: bool = True):
        self.pipeline = self._build_pipeline(use_llm)
        self.ingestor = DataIngestor()
        self.qc = QualityController()
        self.rule_engine = RuleEngine()
        self.maint = default_maintenance_log()
        self.sched = default_operating_schedule()
        self.wo_gen = WorkOrderGenerator()
        self.notifier = Notifier()
        self.emergency = EmergencyPlanner()
        self.feedback_store = FeedbackStore(str(ROOT / "data" / "feedback" / "service.jsonl"))
        self.use_llm = use_llm
        # 实时数据缓存（供界面展示）
        self.latest_window: Optional[pd.DataFrame] = None
        self.latest_result = None

    def _build_pipeline(self, use_llm: bool) -> EarlyWarningPipeline:
        fast = {}
        for col in FAST_COLS:
            p = MODELS / f"fast_{col}.pt"
            if p.exists():
                fast[col] = FastTrackForecaster.load(str(p))
        det = None
        if (MODELS / "fast_anomaly.pkl").exists():
            det = FastAnomalyDetector.load(str(MODELS / "fast_anomaly.pkl"))
        reasoner = None
        if use_llm:
            try:
                reasoner = RootCauseReasoner(LLMClient(), KnowledgeGraph())
            except Exception:
                use_llm = False
        return EarlyWarningPipeline(fast_models=fast, anomaly_detector=det,
                                    reasoner=reasoner, use_llm=use_llm,
                                    feedback_path=str(ROOT / "data" / "feedback" / "service.jsonl"))

    @classmethod
    def get(cls, use_llm: bool = True) -> "AgentService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(use_llm=use_llm)
            return cls._instance

    # ---------------- 业务能力 ----------------

    def process_window(self, df: pd.DataFrame):
        """处理一个数据窗口（端到端五节点）。"""
        result = self.pipeline.process_window(df)
        self.latest_window = df
        self.latest_result = result
        return result

    # ---------------- 流式实时处理 ----------------

    def reset_stream(self, max_points: int = 600):
        """重置实时流缓冲。"""
        self.stream_buf = []                 # 最近 max_points 条原始记录
        self.stream_max = max_points
        self.l1_log = []                     # L1 预警日志
        self.l2_log = []                     # L2 趋势/异常日志
        self.l3_log = []                     # L3 根因诊断日志
        self._stream_cols = ["inlet_temp", "outlet_temp", "pressure", "flow_rate",
                             "tank_level", "cabinet_temp", "cabinet_humidity",
                             "furnace_temp", "electric_power"]

    def stream_step(self, row: dict) -> dict:
        """逐条处理实时数据，返回各级预警与当前指标。"""
        if not hasattr(self, "stream_buf"):
            self.reset_stream()
        self.stream_buf.append(row)
        if len(self.stream_buf) > self.stream_max:
            self.stream_buf.pop(0)
        df = pd.DataFrame(self.stream_buf)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        out = {"timestamp": str(row["timestamp"]), "metrics": {}}
        for c in self._stream_cols:
            if c in row:
                out["metrics"][c] = row[c]

        # L1 规则（瞬时）
        try:
            row_clean = self.qc.process(df).iloc[-1] if len(df) > 1 else df.iloc[-1]
            l1 = self.rule_engine.evaluate_row(row_clean.to_dict())
        except Exception:
            l1 = []
        out["l1"] = [a.as_dict() for a in l1]
        for a in l1:
            self.l1_log.append(a.as_dict())

        # L2 异常检测 + 趋势（窗口足够时）
        out["l2"] = {"anomaly_score": 0.0, "exceed_eta": None}
        det = self.pipeline.detector
        if det is not None and len(df) >= det.window:
            try:
                score = float(det.score(df).iloc[-1])
                out["l2"]["anomaly_score"] = round(score, 3)
                if score > det.threshold:
                    self.l2_log.append({"timestamp": out["timestamp"],
                                        "type": "anomaly", "score": round(score, 3)})
            except Exception:
                pass
        # 趋势越限预测（出水温度）
        fm = self.pipeline.fast_models.get("outlet_temp")
        if fm is not None and len(df) >= 120:
            try:
                eta = fm.predict_exceedance(df["outlet_temp"], 55.0, lookback_s=min(120, len(df)))
                if eta is not None:
                    out["l2"]["exceed_eta"] = round(eta, 1)
                    self.l2_log.append({"timestamp": out["timestamp"], "type": "trend",
                                        "msg": f"预测出水温度 {eta/60:.1f}min 后越限 55℃"})
            except Exception:
                pass

        # L3 根因诊断（L1 触发 或 异常分超阈 时触发，节流）
        out["l3"] = None
        anomaly_high = out["l2"]["anomaly_score"] > (det.threshold if det else 0.6)
        if (l1 or anomaly_high) and self._should_diagnose():
            sensors = list({self._rule_sensor(a["rule_id"]) for a in l1} - {""}) or \
                      ["出水温度", "压力", "流量", "湿度"]
            features = {}
            col_map = {"出水温度": "outlet_temp", "进水温度": "inlet_temp", "压力": "pressure",
                       "流量": "flow_rate", "水箱液位": "tank_level", "湿度": "cabinet_humidity"}
            last = df.iloc[-1]
            for s, c in col_map.items():
                if c in df.columns:
                    features[s] = float(last[c])
            # 尾段统计鉴别特征（最近120s，避免故障前正常段稀释）
            tail = df.iloc[-120:] if len(df) >= 120 else df
            stats = {}
            if "cabinet_humidity" in tail.columns:
                stats["湿度均值_pctRH"] = round(float(tail["cabinet_humidity"].mean()), 1)
                features["湿度"] = stats["湿度均值_pctRH"]
            if "pressure" in tail.columns:
                # 去趋势 std：剔除爬升趋势后专测震荡幅度（气蚀特征）
                import numpy as np
                v = tail["pressure"].to_numpy(dtype=float)
                if len(v) >= 10:
                    t = np.arange(len(v))
                    slope, intercept = np.polyfit(t, v, 1)
                    press_std = float(np.std(v - (slope * t + intercept)))
                else:
                    press_std = float(np.std(v))
                stats["压力波动幅度_std_kPa"] = round(press_std, 2)
            features["_press_std"] = stats.get("压力波动幅度_std_kPa", 0.0)
            try:
                diag = self.diagnose(features, str(last.get("operating_condition", "unknown")),
                                     sensors, stats=stats)
            except Exception:
                # LLM 失败/超时 -> 图谱+数值鉴别兜底，不阻塞实时流
                diag = self.pipeline._fallback_diagnose(sensors, features)
                diag.confidence = min(diag.confidence, 0.65)  # 标注降级置信度
            out["l3"] = diag.to_dict()
            self.l3_log.append({"timestamp": out["timestamp"], **diag.to_dict()})

            # ---- 驱动③ 故障处置智能体：自动生成工单+推送 ----
            try:
                from .agents import FaultHandlingAgent
                fh = FaultHandlingAgent(self)
                analysis = {"level": "orange", "timestamp": out["timestamp"],
                            "l1": {"triggered": bool(l1), "alerts": l1},
                            "l2": l2, "l3": out["l3"]}
                wo = fh.handle(analysis)
                out["work_order"] = wo if wo.get("handled") else None
                if wo.get("handled"):
                    self.l3_log[-1]["order_id"] = wo.get("order_id")
                    # ---- 驱动④ 持续优化智能体：模拟运维反馈归档 ----
                    from .agents import ContinuousOptimizerAgent
                    co = ContinuousOptimizerAgent(self)
                    fb = co.feedback(wo["order_id"], wo["root_cause"], True, 25.0,
                                     "实时监测自动归档（演示）", wo)
                    out["optimization"] = {"archived": fb["archived"],
                                           "stats": fb["stats"]}
            except Exception:
                out["work_order"] = None
                out["optimization"] = None
        return out

    def _should_diagnose(self, min_interval: int = 60) -> bool:
        """L3 诊断节流：避免每条都调 LLM（默认 60s 一次）。"""
        import time as _t
        now = _t.time()
        last = getattr(self, "_last_diag_t", 0)
        if now - last >= min_interval:
            self._last_diag_t = now
            return True
        return False

    @staticmethod
    def _rule_sensor(rule_id: str) -> str:
        from workflow.pipeline import RULE_TO_SENSOR
        return RULE_TO_SENSOR.get(rule_id, "")

    def get_stream_logs(self):
        return {"l1": self.l1_log[-50:], "l2": self.l2_log[-50:], "l3": self.l3_log[-20:]}

    def diagnose(self, features, condition="unknown", sensor_names=None,
                 l1_alerts=None, l2_forecast=None, stats=None, extra_candidates=None):
        """L3 根因诊断（注入完整上下文）。"""
        report = {
            "features": features,
            "condition": condition,
            "l1_alerts": l1_alerts or [],
            "l2_forecast": l2_forecast or {},
            "stats": stats or {},
            "extra_candidates": extra_candidates or [],
            "operating_schedule": self.sched.to_prompt_text(),
            "maintenance_log": self.maint.to_prompt_text(days=60),
        }
        if self.use_llm and self.pipeline.reasoner is not None:
            return self.pipeline.reasoner.diagnose(report=report, sensor_names=sensor_names)
        return self.pipeline._fallback_diagnose(sensor_names or list(features.keys()), features)

    def submit_feedback(self, order_id, actual_root_cause, is_true_fault,
                        handling_time_min, effect):
        self.feedback_store.archive(
            Feedback(order_id, actual_root_cause, is_true_fault, handling_time_min, effect))
        return self.feedback_store.stats()
