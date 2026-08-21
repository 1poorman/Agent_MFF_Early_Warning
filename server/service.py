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

    def reset_stream(self, max_points: int = 20000):
        """重置实时流缓冲（支持预加载 GRU 所需 4.6h=16800 点）。"""
        self.stream_buf = []                 # 预加载 + 实时数据（含 GRU 窗口）
        self.stream_max = max_points
        self.l1_log = []                     # L1 预警日志
        self.l2_log = []                     # L2 趋势/异常日志
        self.l3_log = []                     # L3 根因诊断日志
        self.fh_log = []                     # ③ 工单日志
        self.co_log = []                     # ④ 反馈日志
        self.pending_l3 = None               # 后台 L3 诊断结果（新事件，前端轮询消费）
        self.pending_wo = None
        self.pending_co = None
        self._last_l2_t = 0                  # L2 预测节流
        self.l3_finalized = False            # L3 唯一性：首诊完成后不再重复诊断
        self.wo_issued = False               # 工单唯一性：同一故障场景只发一张
        self.preloaded = 0                   # 已预加载数据条数
        self._stream_cols = ["inlet_temp", "outlet_temp", "pressure", "flow_rate",
                             "tank_level", "cabinet_temp", "cabinet_humidity",
                             "furnace_temp", "electric_power"]

    # ---------------- 预加载（不推送、不分析，仅填充缓冲） ----------------

    def preload_row(self, row: dict):
        """预加载历史数据到缓冲（不触发 L1/L2/L3 分析，供 GRU 窗口与上下文）。"""
        if not hasattr(self, "stream_buf"):
            self.reset_stream()
        self.stream_buf.append(row)
        self.preloaded += 1
        if len(self.stream_buf) > self.stream_max:
            self.stream_buf.pop(0)

    def gru_ready(self) -> bool:
        """GRU 模型可用所需的最小预加载量（快轨窗口）。"""
        fm = self.pipeline.fast_models.get("outlet_temp")
        return fm is not None and len(self.stream_buf) >= fm.window

    # ---------------- 实时单步（轻量，不阻塞） ----------------

    def stream_step(self, row: dict) -> dict:
        """逐条处理实时数据（毫秒级）。

        - L1：直接对当前行瞬时判定（simulator 数据已规整，无需质量管控）
        - L2：异常分（最近窗口） + 真实 GRU 未来预测（预加载完成后可用）
        - L3：触发时启动后台线程诊断（唯一性），结果异步推送（不阻塞数据流）
        """
        if not hasattr(self, "stream_buf"):
            self.reset_stream()
        self.stream_buf.append(row)
        if len(self.stream_buf) > self.stream_max:
            self.stream_buf.pop(0)

        out = {"timestamp": str(row["timestamp"]), "metrics": {},
               "preloaded": self.preloaded}
        for c in self._stream_cols:
            if c in row:
                out["metrics"][c] = row[c]

        # ---- L1：直接对当前行瞬时判定（无需质量管控，微秒级） ----
        try:
            l1 = self.rule_engine.evaluate_row(row)
        except Exception:
            l1 = []
        out["l1"] = [a.as_dict() for a in l1]
        for a in l1:
            self.l1_log.append(a.as_dict())

        # ---- L2：异常分 + GRU 预测（预加载完成后） ----
        out["l2"] = {"anomaly_score": 0.0, "exceed_eta": None, "forecast": None}
        n = len(self.stream_buf)
        det = self.pipeline.detector
        if det is not None and n >= det.window:
            df = pd.DataFrame(self.stream_buf[-120:])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            try:
                score = float(det.score(df).iloc[-1])
                out["l2"]["anomaly_score"] = round(score, 3)
                if score > det.threshold:
                    self.l2_log.append({"timestamp": out["timestamp"], "type": "anomaly",
                                        "score": round(score, 3)})
            except Exception:
                pass

        # L2 预测：真实 GRU 模型（预加载窗口足够时），每 60 条触发一次
        fm = self.pipeline.fast_models.get("outlet_temp")
        if fm is not None and n >= fm.window and (n - self._last_l2_t) >= 60:
            df = pd.DataFrame(self.stream_buf)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            try:
                pred = fm.predict(df["outlet_temp"].iloc[-fm.window:])  # 未来 600s
                out["l2"]["forecast"] = {
                    "horizon_s": 600,
                    "end_value": round(float(pred[-1]), 2),
                    "max_value": round(float(pred.max()), 2),
                    "min_value": round(float(pred.min()), 2),
                    "series": [round(float(v), 2) for v in pred[::30]],  # 抽稀 20 点
                    "method": "GRU+attention(残差学习)",
                }
                self._last_l2_t = n
                eta = fm.predict_exceedance(df["outlet_temp"], 55.0, lookback_s=min(600, n))
                if eta is not None:
                    out["l2"]["exceed_eta"] = round(eta, 1)
                    self.l2_log.append({"timestamp": out["timestamp"], "type": "trend",
                                        "msg": f"GRU预测出水温度 {eta/60:.1f}min 后越限 55℃"})
            except Exception:
                pass

        # ---- L3：后台异步诊断（唯一性：首诊完成后不再重复） ----
        # 门槛：实时段(预加载后) ≥600 条 且 有"故障信号"，避免预加载边界处正常数据误触发
        #  故障信号 = L2 异常分超阈 / L1 泄漏·流量·压力·温度规则 / GRU 越限预测
        realtime_n = n - self.preloaded
        anomaly_high = out["l2"]["anomaly_score"] > (det.threshold if det else 0.6)
        fault_signal = bool(l1) or anomaly_high or out["l2"]["exceed_eta"] is not None
        if fault_signal and realtime_n >= 600 \
                and not self.l3_finalized and self._should_diagnose():
            snapshot = {
                "l1": [a.as_dict() for a in l1],
                "anomaly_score": out["l2"]["anomaly_score"],
                "row": dict(row),
                "buf": list(self.stream_buf[-600:]),
            }
            import threading
            threading.Thread(target=self._diagnose_async, args=(snapshot,),
                             daemon=True).start()
            out["l3"] = "diagnosing"   # 界面提示诊断中

        # 携带最近一次后台诊断完成结果（若存在）
        if self.pending_l3 is not None:
            out["l3"] = self.pending_l3
            out["work_order"] = self.pending_wo
            out["optimization"] = self.pending_co
            self.pending_l3 = self.pending_wo = self.pending_co = None
        return out

    # ---------------- 后台异步诊断 + 驱动③④ ----------------

    def _diagnose_async(self, snapshot: dict):
        """后台线程：L3 LLM 诊断 -> ③工单+推送 -> ④反馈归档，结果挂到 pending_*。"""
        try:
            buf = snapshot["buf"]
            df = pd.DataFrame(buf)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            last = df.iloc[-1]
            sensors = list({self._rule_sensor(a["rule_id"]) for a in snapshot["l1"]} - {""}) or \
                      ["出水温度", "压力", "流量", "湿度"]
            features = {}
            col_map = {"出水温度": "outlet_temp", "进水温度": "inlet_temp", "压力": "pressure",
                       "流量": "flow_rate", "水箱液位": "tank_level", "湿度": "cabinet_humidity",
                       "电导率": "conductivity"}
            for s, c in col_map.items():
                if c in df.columns:
                    features[s] = float(last[c])
            # 尾段统计鉴别特征
            tail = df.iloc[-120:]
            stats = {}
            if "cabinet_humidity" in tail.columns:
                stats["湿度均值_pctRH"] = round(float(tail["cabinet_humidity"].mean()), 1)
                features["湿度"] = stats["湿度均值_pctRH"]
                # 湿度上升趋势（泄漏早期信号）：尾段120s均值 - 前段120s均值
                if len(df) >= 240:
                    head = df.iloc[-240:-120]
                    hum_delta = round(float(tail["cabinet_humidity"].mean())
                                      - float(head["cabinet_humidity"].mean()), 1)
                    stats["湿度上升量_pctRH"] = hum_delta
                    features["_hum_delta"] = hum_delta
            if "pressure" in tail.columns:
                import numpy as np
                v = tail["pressure"].to_numpy(dtype=float)
                if len(v) >= 10:
                    t = np.arange(len(v))
                    slope, intercept = np.polyfit(t, v, 1)
                    stats["压力波动幅度_std_kPa"] = round(float(np.std(v - (slope * t + intercept))), 2)
                features["_press_std"] = stats.get("压力波动幅度_std_kPa", 0.0)
            from .agents import WarningAnalysisAgent
            extra_cands = WarningAnalysisAgent._stat_precheck(features, stats)
            try:
                diag = self.diagnose(features, str(last.get("operating_condition", "unknown")),
                                     sensors, stats=stats, extra_candidates=extra_cands)
            except Exception:
                diag = self.pipeline._fallback_diagnose(sensors, features)
                diag.confidence = min(diag.confidence, 0.65)
            diag_dict = diag.to_dict()
            # 线程安全：置唯一标志，防止并发重复诊断/重复工单
            self.l3_finalized = True
            self.l3_log.append({"timestamp": str(last.get("timestamp", "")), **diag_dict})

            # ---- ③ 故障处置智能体（工单唯一：同一故障场景只生成一次） ----
            wo, co = None, None
            try:
                if self.wo_issued:
                    # 已有工单：不再重复生成/推送，仅同步诊断结果
                    self.pending_l3 = diag_dict
                    self.pending_wo = self.pending_co = None
                    return
                from .agents import FaultHandlingAgent, ContinuousOptimizerAgent
                fh = FaultHandlingAgent(self)
                analysis = {"level": diag_dict.get("level", "orange"),
                            "timestamp": str(last.get("timestamp", "")),
                            "l1": {"triggered": bool(snapshot["l1"]), "alerts": snapshot["l1"]},
                            "l2": {"anomaly_score": snapshot["anomaly_score"]},
                            "l3": diag_dict}
                wo = fh.handle(analysis)
                if wo.get("handled"):
                    self.wo_issued = True      # 工单已发，禁止重复
                    self.fh_log.append({"order_id": wo["order_id"], "level": wo["level"],
                                        "root_cause": wo["root_cause"]})
                    # ---- ④ 持续优化智能体 ----
                    co_agent = ContinuousOptimizerAgent(self)
                    fb = co_agent.feedback(wo["order_id"], wo["root_cause"], True, 25.0,
                                           "实时监测自动归档（演示）", wo)
                    co = {"archived": fb["archived"], "stats": fb["stats"]}
                    self.co_log.append({"timestamp": str(last.get("timestamp", "")),
                                        "archived": fb["archived"],
                                        "total": fb["stats"]["total_feedback"]})
            except Exception:
                wo, co = None, None
            self.pending_l3 = diag_dict
            self.pending_wo = wo if wo and wo.get("handled") else None
            self.pending_co = co
        except Exception:
            pass

    def get_stream_logs(self):
        """返回各级预警与工单/反馈日志（供界面轮询）。"""
        return {
            "l1": self.l1_log[-40:],
            "l2": self.l2_log[-40:],
            "l3": self.l3_log[-10:],
            "work_orders": self.fh_log[-5:],
            "feedback": self.co_log[-5:],
        }

    def _should_diagnose(self, min_interval: int = 10) -> bool:
        """L3 诊断节流：避免每条都调 LLM（默认 10s 一次，演示中随故障发展自动更新诊断）。"""
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
