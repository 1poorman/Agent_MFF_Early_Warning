"""时序预测与异常感知执行体。

包含 L1 规则引擎（MS2）；L2 预测（GRU/PatchTST/TimesFM 可切换）、异常检测、路由（MS3）。
"""

from .rule_engine import Alert, RuleEngine, RuleThresholds

__all__ = ["Alert", "RuleEngine", "RuleThresholds"]
