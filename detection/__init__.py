"""时序预测与异常感知执行体。

当前包含 L1 规则引擎（MS2）；L2 预测/异常检测/路由将在 MS3 加入。
"""

from .rule_engine import Alert, RuleEngine, RuleThresholds

__all__ = ["Alert", "RuleEngine", "RuleThresholds"]
