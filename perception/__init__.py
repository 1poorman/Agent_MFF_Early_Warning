"""多模态自主感知执行体（MS1）。

- ingest:  数据接入（CSV 文件 / 实时流回放），输出统一数据帧
- quality: 数据质量管控（时间戳对齐、缺失插补、异常点剔除）
"""

from .ingest import DataIngestor, NUMERIC_COLUMNS, EXPECTED_FREQ_S
from .quality import QualityController, QualityReport

__all__ = [
    "DataIngestor",
    "NUMERIC_COLUMNS",
    "EXPECTED_FREQ_S",
    "QualityController",
    "QualityReport",
]
