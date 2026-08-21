"""时序数据存储层（本地 PostgreSQL 分区表）。

- tsdb.py: 存储读写（批量插入、区间查询、分区自动管理）
"""

from .tsdb import TimeSeriesDB, DBConfig

__all__ = ["TimeSeriesDB", "DBConfig"]
