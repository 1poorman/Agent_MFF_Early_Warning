"""集中配置 + 统一日志模块。

    from config import get_settings, setup_logging, get_logger

    cfg = get_settings()               # 全局集中配置
    setup_logging()                    # 初始化日志（应用启动时）
    logger = get_logger("module")      # 业务日志
"""

from .logging_config import get_logger, log_alert, setup_logging
from .settings import (
    AppSettings,
    DatabaseSettings,
    DetectionSettings,
    LLMSettings,
    LoggingSettings,
    MCPSettings,
    PathSettings,
    QualitySettings,
    RuleSettings,
    Settings,
    SimulatorSettings,
    get_settings,
    load_settings,
)

__all__ = [
    "get_settings", "load_settings",
    "setup_logging", "get_logger", "log_alert",
    "Settings", "AppSettings", "MCPSettings", "LoggingSettings", "LLMSettings",
    "DatabaseSettings", "SimulatorSettings", "RuleSettings",
    "DetectionSettings", "QualitySettings", "PathSettings",
]
