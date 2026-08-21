"""共享服务层：加载模型、上下文、工作流，供 API 与 MCP 复用。

单例模式，进程启动时初始化一次（模型加载较重）。
所有运行参数统一从 config/settings.yaml 集中读取（可被 .env / 环境变量覆盖）。
"""

import threading
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from action import EmergencyPlanner, Feedback, FeedbackStore, Notifier, WorkOrderGenerator
from config import get_logger, get_settings
from context import default_maintenance_log, default_operating_schedule
from detection import RuleEngine
from detection.fast_anomaly import FastAnomalyDetector
from detection.fast_track import FastTrackForecaster
from detection.forecaster import ForecastEngine
from perception import DataIngestor, QualityController
from reasoning import KnowledgeGraph, LLMClient, RootCauseReasoner
from storage import TimeSeriesDB
from tools.dew_point import CondensationPredictor, dew_point_margin
from workflow import EarlyWarningPipeline

logger = get_logger("server.service")

ROOT = Path(__file__).resolve().parent.parent
FAST_COLS = ["outlet_temp", "flow_rate", "pressure"]


class AgentService:
    """预警智能体共享服务（单例）。"""

    _instance: Optional["AgentService"] = None
    _lock = threading.Lock()

    def __init__(self, use_llm: bool = True):
        self.cfg = get_settings()
        self.pipeline = self._build_pipeline(use_llm)
        self.ingestor = DataIngestor()
        self.qc = QualityController.from_settings()
        self.rule_engine = RuleEngine(self.cfg.to_rule_thresholds())
        self.maint = default_maintenance_log()
        self.sched = default_operating_schedule()
        self.wo_gen = WorkOrderGenerator()
        self.notifier = Notifier()
        self.emergency = EmergencyPlanner()
        self.feedback_store = FeedbackStore(str(self.cfg.paths.feedback))
        self.use_llm = use_llm
        # 实时数据缓存（供界面展示）
        self.latest_window: Optional[pd.DataFrame] = None
        self.latest_result = None
        # 上传时序数据源（用户上传后作为监测数据源，点击开始监测才运行四大智能体）
        self.uploaded_data: Optional[List[Dict]] = None
        self.uploaded_meta: Optional[Dict] = None
        # 电气柜凝露预测器（露点计算 + 缓变趋势外推）
        self.dew = CondensationPredictor(
            margin_warn_c=self.cfg.rules.dew_margin_warn_c, window_s=600, horizon_s=600)
        # 时序数据库（本地 PostgreSQL 分区表；失败不影响 Demo）
        try:
            self.tsdb = TimeSeriesDB(self.cfg.to_db_config())
            self.tsdb_ok = True
        except Exception as e:
            self.tsdb_ok = False
            logger.warning("时序数据库不可用: %s", e)
        # 实时流日志缓冲（供 stream/logs 查询，未启动流时返回空）
        self.reset_stream()
        logger.info("服务初始化完成 | use_llm=%s fast_models=%s anomaly=%s tsdb=%s",
                    use_llm, list(self.pipeline.fast_models.keys()),
                    self.pipeline.detector is not None, self.tsdb_ok)

    def _build_pipeline(self, use_llm: bool) -> EarlyWarningPipeline:
        cfg = self.cfg
        fast = {}
        for col in FAST_COLS:
            p = cfg.paths.models / f"fast_{col}.pt"
            if p.exists():
                fast[col] = FastTrackForecaster.load(str(p))
        det = None
        if (cfg.paths.models / "fast_anomaly.pkl").exists():
            det = FastAnomalyDetector.load(str(cfg.paths.models / "fast_anomaly.pkl"))
        reasoner = None
        if use_llm:
            try:
                reasoner = RootCauseReasoner(LLMClient(config=cfg.llm.to_client_dict()),
                                             KnowledgeGraph())
            except Exception:
                use_llm = False
        pipeline = EarlyWarningPipeline(fast_models=fast, anomaly_detector=det,
                                        reasoner=reasoner, use_llm=use_llm,
                                        feedback_path=str(cfg.paths.feedback))
        # L2 预测后端：按 config.detection.forecast_model 可切换
        pipeline.forecast_engine = ForecastEngine(
            backend_name=cfg.detection.forecast_model,
            fast_models=fast,
            precise_model=self._load_precise(cfg),
            timesfm_weights_dir=str(cfg.detection.timesfm_weights),
            horizon_s=cfg.detection.forecast_horizon,
            models_dir=str(cfg.paths.models))
        return pipeline

    @staticmethod
    def _load_precise(cfg):
        p = cfg.paths.models / "precise.pt"
        if p.exists():
            try:
                from detection.precise_track import PreciseTrackForecaster
                return PreciseTrackForecaster.load(str(p))
            except Exception:
                return None
        return None

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

    # ---------------- 上传数据源管理 ----------------

    def set_uploaded_data(self, records: List[Dict], meta: Dict):
        """保存上传的时序数据作为监测数据源（点击开始监测后回放运行四大智能体）。

        统一归一化：timestamp 转字符串、数值字段转 float（避免 Timestamp/numpy 类型
        在 WS JSON 序列化时失败），非数值字段（工况/故障标注）保留原样。
        """
        import numpy as _np
        norm = []
        for r in records:
            row = dict(r)
            ts = row.get("timestamp")
            row["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
            for k, v in row.items():
                if k == "timestamp":
                    continue
                if isinstance(v, (_np.floating, float)):
                    row[k] = float(v)
                elif isinstance(v, (_np.integer, int)):
                    row[k] = float(v)
                elif isinstance(v, _np.ndarray):
                    row[k] = float(v[0])
            norm.append(row)
        self.uploaded_data = norm
        self.uploaded_meta = meta
        logger.info("上传数据源就绪 | %s 条=%d", meta.get("filename", ""), len(norm))

    def clear_uploaded_data(self):
        """清除上传数据源（回到仿真数据源）。"""
        self.uploaded_data = None
        self.uploaded_meta = None

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
        self.dew_log = []                    # 凝露风险预警日志
        self._stream_cols = ["inlet_temp", "outlet_temp", "pressure", "flow_rate",
                             "flow_velocity", "tank_level", "conductivity",
                             "cabinet_temp", "cabinet_humidity", "furnace_temp",
                             "electric_power", "electric_current"]

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
               "preloaded": self.preloaded, "dew": None}
        for c in self._stream_cols:
            if c in row:
                out["metrics"][c] = row[c]

        # ---- 电气柜凝露风险（露点计算 + 缓变趋势预测） ----
        # 凝露条件：柜体表面温度 < 露点温度（湿空气在低温表面凝结，绝缘下降/短路风险）
        if "cabinet_temp" in row and "cabinet_humidity" in row:
            try:
                margin_now = float(dew_point_margin(row["cabinet_temp"], row["cabinet_humidity"]))
                t_dew = row["cabinet_temp"] - margin_now
                out["dew"] = {"dew_point": round(float(t_dew), 1),
                              "margin": round(margin_now, 2),
                              "at_risk": margin_now < 3.0,
                              "eta_s": None}
                if out["dew"]["at_risk"]:
                    self.dew_log.append({"timestamp": out["timestamp"], "type": "dew_now",
                                         "msg": f"电气柜凝露风险：表面温度-露点裕度 {margin_now:.1f}℃ (<3℃)"})
                # 趋势预测：实时段 ≥10min 且每 60s 评估一次
                realtime_n = len(self.stream_buf) - self.preloaded
                if realtime_n >= 600 and realtime_n % 60 == 0:
                    dff = pd.DataFrame(self.stream_buf[-120:])
                    risk = self.dew.assess(pd.to_datetime(dff["timestamp"]),
                                           dff["cabinet_temp"], dff["cabinet_humidity"])
                    if risk.predicted_risk:
                        out["dew"]["eta_s"] = risk.eta_s
                        self.dew_log.append({"timestamp": out["timestamp"], "type": "dew_predict",
                                             "msg": f"预测 {risk.eta_s/60:.0f}min 后电气柜凝露（当前裕度 {risk.margin_now:.1f}℃）"})
            except Exception:
                pass

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

        # L2 预测：统一预测引擎（GRU/PatchTST/TimesFM 可切换），每 60 条触发一次
        engine = getattr(self.pipeline, "forecast_engine", None)
        if engine is not None and engine.available() \
                and n >= engine.backend_window() and (n - self._last_l2_t) >= 60:
            df = pd.DataFrame(self.stream_buf)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            try:
                fc = engine.forecast(df["outlet_temp"])
                if fc is not None:
                    out["l2"]["forecast"] = fc
                    self._last_l2_t = n
                    if fc["max_value"] > self.cfg.detection.forecast_threshold:
                        eta = engine.exceedance_eta(df["outlet_temp"], 55.0)
                        if eta is not None:
                            out["l2"]["exceed_eta"] = round(eta, 1)
                            self.l2_log.append({"timestamp": out["timestamp"], "type": "trend",
                                                "msg": f"{fc['method']}预测出水温度 {eta/60:.1f}min 后越限 55℃"})
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
            logger.info("触发 L3 根因诊断 | ts=%s l1=%d anomaly=%.3f exceed_eta=%s",
                        out["timestamp"], len(l1), out["l2"]["anomaly_score"],
                        out["l2"].get("exceed_eta"))
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
            cond = last.get("operating_condition")
            condition = "unknown" if cond is None or (
                isinstance(cond, float) and pd.isna(cond)) else str(cond)
            try:
                diag = self.diagnose(features, condition,
                                     sensors, stats=stats, extra_candidates=extra_cands)
            except Exception:
                diag = self.pipeline._fallback_diagnose(sensors, features)
                diag.confidence = min(diag.confidence, 0.65)
            diag_dict = diag.to_dict()
            # 附加诊断上下文（与故障直接相关：L1/L2 上报 + 特征值 + 统计特征 + 维修记录）
            l1_ctx = snapshot["l1"]
            diag_dict["context"] = {
                "l1_rules": sorted({a["rule_id"] for a in l1_ctx}),
                "l1_alerts": l1_ctx[-10:],
                "l2": {"anomaly_score": snapshot["anomaly_score"],
                       "anomaly_triggered": snapshot["anomaly_score"] > (self.pipeline.detector.threshold if self.pipeline.detector else 0.6)},
                "features": {k: v for k, v in features.items() if not k.startswith("_")},
                "stats": stats,
                "maintenance_log": [
                    {"order_id": o.order_id, "date": o.date,
                     "component": o.component, "action": o.action, "note": o.note}
                    for o in self.maint.recent(days=60)],
            }
            # 线程安全：置唯一标志，防止并发重复诊断/重复工单
            self.l3_finalized = True
            self.l3_log.append({"timestamp": str(last.get("timestamp", "")), **diag_dict})
            logger.info("L3 根因诊断完成 | root_cause=%s confidence=%.2f level=%s "
                        "manual=%s retries=%d",
                        diag_dict["root_cause"], diag_dict["confidence"],
                        diag_dict["level"], diag_dict["manual_required"],
                        diag_dict["retries"])

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
                    logger.info("故障处置工单生成 | order_id=%s level=%s root_cause=%s",
                                wo["order_id"], wo["level"], wo["root_cause"])
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
            "dew": self.dew_log[-10:],
            "work_orders": self.fh_log[-5:],
            "feedback": self.co_log[-5:],
        }

    # ---------------- 时序落库（异步，失败不影响 Demo） ----------------

    def persist(self, row: dict):
        """单条落库到本地时序数据库（异步线程，避免阻塞实时流）。"""
        if not getattr(self, "tsdb_ok", False):
            return
        try:
            import threading as _t
            def _do():
                try:
                    self.tsdb.insert([dict(row)])
                except Exception:
                    pass
            _t.Thread(target=_do, daemon=True).start()
        except Exception:
            pass

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
