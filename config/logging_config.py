"""统一日志配置：控制台 + 按天滚动文件双输出，记录运行关键日志。

用法：
    from config import get_logger, setup_logging
    setup_logging()                      # 应用启动时调用一次
    logger = get_logger("server.api")    # 业务模块取 logger
    logger.info("关键日志...")

输出：
    - 控制台：标准输出（与 uvicorn 日志同流）
    - 文件：logs/app.log（按天滚动）+ logs/error.log（ERROR 及以上）
    - 时间戳均为北京时间（UTC+8）；进入新的北京日后首条日志触发切换，
      当天日志归档为 logs/app.log.YYYY-MM-DD，保留 backup_count 天。
    注：部署机时区可能是 UTC，时间戳与滚动均显式按东八区计算，不依赖系统时区。
"""

import glob
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .settings import get_settings

_DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
# 东八区（北京时间），部署机为 UTC 时也统一按此输出/轮转
_BJ_TZ = timezone(timedelta(hours=8))


def _bj_now() -> datetime:
    return datetime.now(_BJ_TZ)


class _BJFormatter(logging.Formatter):
    """asctime 固定输出北京时间（不依赖系统时区）。"""

    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, _BJ_TZ).timetuple()


class _BJDailyRotatingFileHandler(logging.FileHandler):
    """按北京日期滚动的文件处理器。

    进入新的一天（北京时间）后的首条日志触发切换：
      app.log -> app.log.YYYY-MM-DD（YYYY-MM-DD 为刚结束的北京日期）
    并清理超过 backup_count 天的历史归档。因日志频率高（逐条流式），
    实际切换时刻即北京时间 00:00 后的毫秒级内。
    """

    def __init__(self, filename, backupCount: int = 10,
                 encoding: Optional[str] = None):
        super().__init__(filename, encoding=encoding)
        self._backup_days = max(int(backupCount), 1)
        self._today = _bj_now().date()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if _bj_now().date() != self._today:
                self._do_rollover()
        except Exception:
            self.handleError(record)
            return
        super().emit(record)

    def _do_rollover(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None
        archived = f"{self.baseFilename}.{self._today:%Y-%m-%d}"
        if os.path.exists(archived):
            os.remove(archived)
        if os.path.exists(self.baseFilename):
            os.replace(self.baseFilename, archived)
        # 清理超过保留天数的历史归档（仅匹配本文件的 YYYY-MM-DD 归档）
        prefix = self.baseFilename + "."
        for f in glob.glob(prefix + "*"):
            stem = f[len(prefix):]
            try:
                d = datetime.strptime(stem, "%Y-%m-%d").date()
            except ValueError:
                continue
            if (_bj_now().date() - d).days > self._backup_days:
                try:
                    os.remove(f)
                except OSError:
                    pass
        self._today = _bj_now().date()
        if not self.delay:
            self.stream = self._open()


_CONFIGURED_FLAG = "_mff_logging_configured"


def setup_logging(
    level: Optional[str] = None,
    log_dir: Optional[Path] = None,
    console: bool = True,
    force: bool = False,
) -> logging.Logger:
    """初始化根日志。

    幂等：已配置过则直接返回（避免多次导入/懒加载兜底导致的重复初始化，
    例如 service.get_logger 的懒加载先于 api 显式调用触发两次 setup，
    在日志文件中留下两条相同的"日志系统初始化完成"）。需覆盖配置时传 force=True。

    Args:
        level: 日志级别（覆盖配置文件，如 "DEBUG"）
        log_dir: 日志目录（覆盖配置文件）
        console: 是否输出控制台
        force: 已配置的情况下是否仍强制重建（换目录/级别等场景）
    """
    root = logging.getLogger()
    if not force and getattr(root, _CONFIGURED_FLAG, False):
        return root
    cfg = get_settings().logging
    level_name = (level or cfg.level).upper()
    level_num = getattr(logging, level_name, logging.INFO)
    directory = log_dir or cfg.log_dir
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    fmt = _BJFormatter(cfg.format or _DEFAULT_FMT, datefmt=_DATE_FMT)

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
    fh = _BJDailyRotatingFileHandler(directory / file_name,
                                     backupCount=cfg.backup_count,
                                     encoding="utf-8")
    fh.setLevel(level_num)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if cfg.error_file:
        efh = _BJDailyRotatingFileHandler(directory / cfg.error_file,
                                          backupCount=cfg.backup_count,
                                          encoding="utf-8")
        efh.setLevel(logging.ERROR)
        efh.setFormatter(fmt)
        root.addHandler(efh)

    root.info("日志系统初始化完成：level=%s file=%s/%s 按天滚动(北京时间) "
              "保留%d天", level_name, directory, file_name, cfg.backup_count)
    setattr(root, _CONFIGURED_FLAG, True)
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
