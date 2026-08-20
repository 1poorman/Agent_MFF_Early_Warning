"""反馈归档与自主迭代（闭环最后一环）。

运维人员处置完成后反馈：真实根因/处理时长/效果 -> 归档训练样本库 ->
触发周期性模型微调标记，形成"运行一次、进步一点"的正向循环。
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Feedback:
    """处置反馈。"""
    order_id: str
    actual_root_cause: str        # 运维确认的真实根因
    is_true_fault: bool           # 是否真实故障（vs 误报）
    handling_time_min: float      # 处理时长
    effect: str                   # 处置效果
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))


class FeedbackStore:
    """反馈归档库（JSONL 持久化 + 训练样本 + 微调标记）。"""

    def __init__(self, store_path: str = "data/feedback/feedback.jsonl"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Feedback] = []
        self._retrain_pending = False

    def archive(self, fb: Feedback, diagnosis_snapshot: Optional[Dict] = None):
        """归档反馈 + 关联诊断快照 -> 训练样本。"""
        self.records.append(fb)
        sample = {"feedback": asdict(fb), "diagnosis": diagnosis_snapshot or {}}
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        # 每积累真实故障样本即标记待微调
        if fb.is_true_fault:
            self._retrain_pending = True

    def should_retrain(self, min_samples: int = 5) -> bool:
        """是否达到微调触发条件（真实故障样本数）。"""
        true_faults = sum(1 for r in self.records if r.is_true_fault)
        return self._retrain_pending and true_faults >= min_samples

    def mark_retrained(self):
        self._retrain_pending = False

    def stats(self) -> Dict:
        true_faults = sum(1 for r in self.records if r.is_true_fault)
        return {
            "total_feedback": len(self.records),
            "true_faults": true_faults,
            "false_alarms": len(self.records) - true_faults,
            "retrain_pending": self._retrain_pending,
        }
