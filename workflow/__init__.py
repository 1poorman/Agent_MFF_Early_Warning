"""Agent 工作流编排层（MS6）。

五节点闭环（文档 5.3）：
1. 多模态数据接入（重采 ≤3 次，失败告警）
2. 边缘端特征提取（≥90% 正常数据过滤）
3. 模型根因推理（防幻觉校验，重试 ≤3 次）
4. 工具调用与工单生成（全链路记录）
5. 反馈归档与自主迭代
"""

from .pipeline import EarlyWarningPipeline, PipelineResult

__all__ = ["EarlyWarningPipeline", "PipelineResult"]
