"""本地时序数据库存储（PostgreSQL 声明式 RANGE 分区，等效时序库 hypertable）。

- 主表 sensor_ts 按时间 RANGE 分区（月度），自动建分区
- 批量插入：executemany + 单事务
- 查询：区间/工况/故障标签过滤，时序倒序
- 独立于 Demo 运行（Demo 可不落库），预留真实部署启用
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool

logger = logging.getLogger(__name__)

# 与 simulator/数据契约对齐的数值字段
NUMERIC_COLS = [
    "inlet_temp", "outlet_temp", "pressure", "flow_rate", "flow_velocity",
    "tank_level", "conductivity", "cabinet_temp", "cabinet_humidity",
    "furnace_temp", "electric_power", "electric_current",
]
TEXT_COLS = ["operating_condition", "fault_label"]


@dataclass
class DBConfig:
    dsn: str = "postgresql://postgres:postgres@localhost:5432/mff_tsdb"
    pool_min: int = 1
    pool_max: int = 5

    @classmethod
    def from_settings(cls) -> "DBConfig":
        """从集中配置（config/settings.yaml database 段）构造。"""
        from config import get_settings
        db = get_settings().database
        return cls(dsn=db.dsn, pool_min=db.pool_min, pool_max=db.pool_max)


class TimeSeriesDB:
    """中频炉时序数据存储读写。"""

    def __init__(self, cfg: Optional[DBConfig] = None):
        self.cfg = cfg or DBConfig()
        self._pool: Optional[SimpleConnectionPool] = None

    # ---------------- 连接 ----------------

    def _connect(self):
        if self._pool is None:
            self._pool = SimpleConnectionPool(
                self.cfg.pool_min, self.cfg.pool_max, self.cfg.dsn)
        return self._pool.getconn()

    def _release(self, conn):
        self._pool.putconn(conn)

    def close(self):
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None

    # ---------------- 分区管理 ----------------

    def ensure_partition(self, ts: datetime) -> None:
        """确保 ts 所在月度分区存在（含下月预建）。"""
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                for dt in (ts, ts + timedelta(days=32)):
                    month = dt.strftime("%Y_%m")
                    cur.execute("""
                        SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                        WHERE c.relname=%s AND n.nspname='public'""",
                        (f"sensor_ts_{month}",))
                    if not cur.fetchone():
                        cur.execute("""
                            SELECT create_month_partition(%s)""", (dt.strftime("%Y-%m"),))
                conn.commit()
        finally:
            self._release(conn)

    # ---------------- 写入 ----------------

    def insert(self, rows: List[Dict]) -> int:
        """批量插入传感器记录（executemany，单事务）。返回插入条数。"""
        if not rows:
            return 0
        conn = self._connect()
        try:
            cols = ["ts"] + NUMERIC_COLS + TEXT_COLS
            with conn.cursor() as cur:
                records = []
                for r in rows:
                    ts = pd.to_datetime(r.get("timestamp")).to_pydatetime()
                    self.ensure_partition(ts)
                    rec = [ts]
                    for c in NUMERIC_COLS + TEXT_COLS:
                        v = r.get(c)
                        rec.append(None if v is None else float(v) if c in NUMERIC_COLS else str(v))
                    records.append(rec)
                psycopg2.extras.execute_values(
                    cur,
                    f"INSERT INTO sensor_ts ({','.join(cols)}) VALUES %s "
                    f"ON CONFLICT DO NOTHING",
                    records,
                )
                conn.commit()
                return len(records)
        except Exception as e:
            conn.rollback()
            logger.error("insert failed: %s", e)
            raise
        finally:
            self._release(conn)

    # ---------------- 查询 ----------------

    def query(self, start: datetime, end: datetime,
              condition: Optional[str] = None,
              fault: Optional[str] = None,
              columns: Optional[List[str]] = None) -> pd.DataFrame:
        """区间查询，可选工况/故障过滤。返回 DataFrame。"""
        cols = (columns or NUMERIC_COLS + TEXT_COLS + ["ts"])
        col_sql = ", ".join(cols)
        sql = f"SELECT {col_sql} FROM sensor_ts WHERE ts >= %s AND ts <= %s"
        params: List = [start, end]
        if condition:
            sql += " AND operating_condition = %s"
            params.append(condition)
        if fault:
            sql += " AND fault_label = %s"
            params.append(fault)
        sql += " ORDER BY ts ASC"

        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                df = pd.DataFrame(rows, columns=cols)
                if not df.empty:
                    df["ts"] = pd.to_datetime(df["ts"])
                return df
        finally:
            self._release(conn)

    def latest(self, n: int = 100) -> pd.DataFrame:
        """最近 n 条记录（时序倒序）。"""
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ts, %s FROM sensor_ts ORDER BY ts DESC LIMIT %s
                """ % (", ".join(NUMERIC_COLS + TEXT_COLS), n))
                rows = cur.fetchall()
                df = pd.DataFrame(rows, columns=["ts"] + NUMERIC_COLS + TEXT_COLS)
                if not df.empty:
                    df["ts"] = pd.to_datetime(df["ts"])
                return df
        finally:
            self._release(conn)

    def count(self) -> int:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM sensor_ts")
                return cur.fetchone()[0]
        finally:
            self._release(conn)

    def stats(self) -> Dict:
        """表规模与分区统计。"""
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT c.relname, pg_size_pretty(pg_total_relation_size(c.oid))
                    FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                    WHERE n.nspname='public' AND c.relname LIKE 'sensor_ts%'
                    ORDER BY c.relname""")
                tables = cur.fetchall()
                cur.execute("SELECT count(*) FROM sensor_ts")
                total = cur.fetchone()[0]
            return {"table": "sensor_ts", "rows": total,
                    "partitions": [{"name": t[0], "size": t[1]} for t in tables]}
        finally:
            self._release(conn)
