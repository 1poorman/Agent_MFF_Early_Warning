"""统一日志配置：控制台 + 滚动文件双输出，记录运行关键日志。

用法：
    from config import get_logger, setup_logging
    setup_logging()                      # 应用启动时调用一次
    logger = get_logger("server.api")    # 业务模块取 logger
    logger.info("关键日志...")

输出：
    - 控制台：标准输出（与 uvicorn 日志同流）
    - 文件：logs/app.log（滚动，10MB×10）+ logs/error.log（ERROR 及以上）
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .settings import get_settings

_DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: Optional[str] = None,
    log_dir: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """初始化根日志（幂等：重复调用会重建 handler，避免重复输出）。

    Args:
        level: 日志级别（覆盖配置文件，如 "DEBUG"）
        log_dir: 日志目录（覆盖配置文件）
        console: 是否输出控制台
    """
    cfg = get_settings().logging
    level_name = (level or cfg.level).upper()
    level_num = getattr(logging, level_name, logging.INFO)
    directory = log_dir or cfg.log_dir
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(cfg.format or _DEFAULT_FMT, datefmt=_DATE_FMT)

    root = logging.getLogger()
    root.setLevel(level_num)
    # 清空既有 handlers，保证幂等
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level_num)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    file_name = cfg.file or "app.log"
    fh = RotatingFileHandler(directory / file_name, maxBytes=cfg.max_bytes,
                             backupCount=cfg.backup_count, encoding="utf-8")
    fh.setLevel(level_num)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if cfg.error_file:
        efh = RotatingFileHandler(directory / cfg.error_file, maxBytes=cfg.max_bytes,
                                  backupCount=cfg.backup_count, encoding="utf-8")
        efh.setLevel(logging.ERROR)
        efh.setFormatter(fmt)
        root.addHandler(efh)

    root.info("日志系统初始化完成：level=%s file=%s/%s",
              level_name, directory, file_name)
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取业务模块 logger（未初始化时自动按默认配置初始化）。"""
    # 根 logger 无 handler 时先初始化，保证直接使用也有输出
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name or "mff_agent")


def log_alert(logger: logging.Logger, level: str, message: str, **extra) -> None:
    """统一预警日志入口：level 与 message 为必填，extra 附加关键字段。"""
    tag = f"[{level.upper()}] " if level else ""
    detail = " | ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
    logger.warning("%s%s%s", tag, message, f" | {detail}" if detail else "")
