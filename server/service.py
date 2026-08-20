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

    def diagnose(self, features, condition="unknown", sensor_names=None,
                 l1_alerts=None, l2_forecast=None):
        """L3 根因诊断（注入完整上下文）。"""
        report = {
            "features": features,
            "condition": condition,
            "l1_alerts": l1_alerts or [],
            "l2_forecast": l2_forecast or {},
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
