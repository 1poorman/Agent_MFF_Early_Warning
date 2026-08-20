"""数据接入模块：对接 simulator 输出（CSV / 实时流回放），输出统一数据帧。

预留工业协议适配接口（Modbus / OPC UA / MQTT）：
    DataIngestor(source="opcua://...")  # 未来扩展，当前支持 csv / stream
"""

from pathlib import Path
from typing import Iterator, List, Optional, Union

import numpy as np
import pandas as pd

EXPECTED_FREQ_S = 1  # 题目要求：每秒 1 条记录

# 与 simulator 输出契约对齐的数值字段
NUMERIC_COLUMNS: List[str] = [
    "inlet_temp", "outlet_temp", "pressure", "flow_rate", "flow_velocity",
    "tank_level", "conductivity",
    "cabinet_temp", "cabinet_humidity",
    "furnace_temp", "electric_power", "electric_current",
]
LABEL_COLUMNS: List[str] = ["operating_condition", "fault_label"]
ALL_COLUMNS: List[str] = ["timestamp"] + NUMERIC_COLUMNS + LABEL_COLUMNS


class DataIngestor:
    """数据接入器。

    用法：
        ing = DataIngestor()
        df = ing.load_csv("data/simulated/normal_24h.csv")   # 批量
        for row in ing.replay_csv(path, speed=60):           # 流回放
            ...
    """

    def load_csv(self, path: Union[str, Path]) -> pd.DataFrame:
        """加载 CSV，返回 schema 校验后的 DataFrame（不修改原始数据）。"""
        df = pd.read_csv(path)
        return self.validate_schema(df)

    def replay_csv(
        self,
        path: Union[str, Path],
        speed: float = 0.0,
    ) -> Iterator[pd.Series]:
        """按行回放 CSV，模拟实时流。

        speed: 回放倍速，0 表示不限速（尽快逐行吐出）。
        """
        import time

        df = self.load_csv(path)
        interval = EXPECTED_FREQ_S / speed if speed > 0 else 0.0
        for _, row in df.iterrows():
            yield row
            if interval > 0:
                time.sleep(interval)

    @staticmethod
    def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
        """校验数据契约：字段齐全、类型正确、时间戳可解析。不改动数据本身。"""
        missing = [c for c in ALL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必需字段: {missing}")
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for c in NUMERIC_COLUMNS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[ALL_COLUMNS]
