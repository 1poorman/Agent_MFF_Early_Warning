"""反馈归档与自主迭代（闭环最后一环）。

运维人员处置完成后反馈：真实根因/处理时长/效果 -> 归档训练样本库 ->
触发周期性模型微调标记，形成"运行一次、进步一点"的正向循环。

归档样本 schema（DPO 前置，MS6 修复）：
    {"feedback": {...}, "diagnosis": {"prompt": <输入>, "output": <模型原始输出>,
                                      "root_cause", "confidence", ...}}
其中 diagnosis.prompt（输入提示）+ output（模型输出）+ feedback.actual_root_cause
（真值）三者齐全，才能在 MS7 构造 DPO 偏好对；缺 prompt 时显式告警。
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import get_logger

logger = get_logger("action.feedback")


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
        """归档反馈 + 关联诊断快照 -> 训练样本。

        diagnosis_snapshot 应为 DiagnosisResult.training_snapshot()（含 prompt/output）。
        真实故障样本若缺 prompt，则无法用于 DPO，显式告警但不阻断归档。
        """
        self.records.append(fb)
        snap = diagnosis_snapshot or {}
        if fb.is_true_fault and not snap.get("prompt"):
            logger.warning(
                "反馈快照缺输入 prompt，该真实故障样本不可用于 DPO 偏好对 | order_id=%s",
                fb.order_id)
        sample = {"feedback": asdict(fb), "diagnosis": snap}
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        # 每积累真实故障样本即标记待微调
        if fb.is_true_fault:
            self._retrain_pending = True

    def build_preference_pairs(self) -> List[Dict]:
        """从归档 jsonl 读取，构造 DPO 偏好对原料（MS7 前置能力）。

        仅取「真实故障 + 快照含 prompt/output + 模型判错（root_cause≠真值）」的样本：
        - prompt：模型输入（退化版提示）
        - rejected：模型原始错误输出
        - actual_root_cause：运维确认真值（chosen 的锚点，具体 chosen JSON 由 MS7 组装）
        返回列表；缺 prompt 的样本被跳过（对齐 archive 告警）。
        """
        pairs: List[Dict] = []
        if not self.store_path.is_file():
            return pairs
        for line in self.store_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fb = rec.get("feedback", {})
            diag = rec.get("diagnosis", {})
            if not fb.get("is_true_fault") or not diag.get("prompt"):
                continue
            actual = fb.get("actual_root_cause")
            model_rc = diag.get("root_cause")
            if not actual or actual == model_rc:
                continue  # 模型判对，无需偏好对
            pairs.append({
                "order_id": fb.get("order_id"),
                "prompt": diag["prompt"],
                "rejected": diag.get("output", ""),
                "actual_root_cause": actual,
                "model_root_cause": model_rc,
            })
        return pairs

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
