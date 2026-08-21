"""数据质量管控模块（MS1）。

流水线：
    时间戳对齐（去重/排序/重采样 1Hz 网格）
    -> 缺失插补（同工况相邻片段加权均值，误差 <2%）
    -> 异常点剔除（Hampel 滤波，物理约束裁剪）
输出：规整数据帧 + QualityReport（完整度、插补数、剔除数）
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .ingest import EXPECTED_FREQ_S, LABEL_COLUMNS, NUMERIC_COLUMNS

# Hampel 滤波参数：窗口（秒）、阈值（倍 MAD）
HAMPEL_WINDOW_S = 121   # 覆盖 2 分钟，保证中位数稳健
HAMPEL_N_SIGMA = 5.0

# 物理速率限制（每秒最大变化量），传感器不可能瞬时跳变
MAX_RATE_PER_S: Dict[str, float] = {
    "inlet_temp": 0.5,
    "outlet_temp": 0.5,
    "pressure": 50.0,           # kPa/s（气蚀等真实压力震荡可达 ±14kPa@0.5Hz，需保留；尖刺由 Hampel 拦截）
    "flow_rate": 0.5,           # L/s per s
    "flow_velocity": 0.2,
    "tank_level": 5.0,          # cm/s（补水阀切换瞬间允许较大变化）
    "conductivity": 10.0,
    "cabinet_temp": 15.0,       # 功率阶跃时柜温可快速跟随
    "cabinet_humidity": 2.0,
    "furnace_temp": 20.0,       # 出炉/加冷料时炉温可快速下降
    "electric_power": 3100.0,   # 允许满功率阶跃（启炉/停炉）
    "electric_current": 2000.0, # 与功率阶跃对应
}

# 物理约束上下限（超出即为不可信测量，剔除后按缺失插补）
PHYSICAL_BOUNDS: Dict[str, tuple] = {
    "inlet_temp": (0.0, 60.0),
    "outlet_temp": (0.0, 100.0),
    "pressure": (0.0, 600.0),          # kPa
    "flow_rate": (0.0, 20.0),          # L/s
    "flow_velocity": (0.0, 10.0),      # m/s
    "tank_level": (0.0, 400.0),        # cm
    "conductivity": (0.0, 2000.0),     # µS/cm
    "cabinet_temp": (0.0, 90.0),
    "cabinet_humidity": (0.0, 100.0),
    "furnace_temp": (0.0, 2000.0),
    "electric_power": (0.0, 5000.0),   # kW
    "electric_current": (0.0, 5000.0), # A
}


@dataclass
class QualityReport:
    """质量管控报告。"""
    total_in: int = 0               # 输入记录数
    total_out: int = 0              # 输出记录数
    duplicates_removed: int = 0
    out_of_order_fixed: int = 0
    gaps_filled: int = 0            # 时间戳空洞补齐条数
    missing_filled: int = 0         # 缺失值插补个数
    outliers_removed: int = 0       # 异常点剔除个数
    completeness: float = 0.0       # 完整度 = 输出无缺失网格点数 / 期望网格点数

    def summary(self) -> str:
        return (
            f"输入 {self.total_in} 条 -> 输出 {self.total_out} 条 | "
            f"去重 {self.duplicates_removed} 乱序修正 {self.out_of_order_fixed} "
            f"补洞 {self.gaps_filled} 插补 {self.missing_filled} 剔除 {self.outliers_removed} | "
            f"完整度 {self.completeness:.2%}"
        )


class QualityController:
    """数据质量管控器。"""

    def __init__(
        self,
        hampel_window_s: int = HAMPEL_WINDOW_S,
        hampel_n_sigma: float = HAMPEL_N_SIGMA,
        physical_bounds: Optional[Dict[str, tuple]] = None,
        max_rate_per_s: Optional[Dict[str, float]] = None,
    ):
        self.hampel_window_s = hampel_window_s
        self.hampel_n_sigma = hampel_n_sigma
        self.bounds = physical_bounds or PHYSICAL_BOUNDS
        self.max_rate_per_s = max_rate_per_s or MAX_RATE_PER_S

    @classmethod
    def from_settings(cls) -> "QualityController":
        """从集中配置（config/settings.yaml quality 段）构造。"""
        from config import get_settings
        q = get_settings().quality
        bounds = {k: tuple(v) for k, v in q.physical_bounds.items()}
        return cls(hampel_window_s=q.hampel_window_s,
                   hampel_n_sigma=q.hampel_n_sigma,
                   physical_bounds=bounds or None,
                   max_rate_per_s=q.max_rate_per_s or None)

    # ---------------- 主流程 ----------------

    def process(self, df: pd.DataFrame) -> tuple:
        """执行全链路质量管控，返回 (规整 DataFrame, QualityReport)。"""
        report = QualityReport(total_in=len(df))
        df = df.copy()

        df = self._align_timestamps(df, report)
        df = self._remove_outliers(df, report)
        df = self._impute_missing(df, report)

        expected = len(df)
        report.total_out = len(df)
        report.completeness = (
            float((~df[NUMERIC_COLUMNS].isna().any(axis=1)).sum()) / expected if expected else 0.0
        )
        return df, report

    # ---------------- 时间戳对齐 ----------------

    def _align_timestamps(self, df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
        ts = df["timestamp"]
        report.out_of_order_fixed = int((ts.diff().dropna() < pd.Timedelta(0)).sum())
        dup = int(ts.duplicated().sum())
        report.duplicates_removed = dup
        if dup:
            df = df.drop_duplicates(subset="timestamp", keep="first")
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 重采样到严格 1Hz 网格：起止之间所有秒点必须存在
        full_index = pd.date_range(
            start=df["timestamp"].iloc[0],
            end=df["timestamp"].iloc[-1],
            freq=f"{EXPECTED_FREQ_S}s",
        )
        report.gaps_filled = int(len(full_index) - len(df))
        df = df.set_index("timestamp").reindex(full_index)
        df.index.name = "timestamp"
        df = df.reset_index()

        # 标签列前向填充（工况/故障标签随最近已知值）
        for c in LABEL_COLUMNS:
            df[c] = df[c].ffill().bfill()
        return df

    # ---------------- 异常点剔除 ----------------

    def _remove_outliers(self, df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
        """物理约束裁剪 + Hampel 滤波：异常点置 NaN，交由插补阶段处理。"""
        for col in NUMERIC_COLUMNS:
            series = df[col]

            # 1) 物理约束裁剪
            lo, hi = self.bounds[col]
            bad_physical = (series < lo) | (series > hi)
            n_bad = int(bad_physical.sum())
            if n_bad:
                series = series.mask(bad_physical)

            # 2) Hampel 滤波（滚动中位数 ± n·MAD）
            #    MAD 过小时（传感器噪声低于量化步长），阈值以量化步长兜底，避免误报
            med = series.rolling(self.hampel_window_s, center=True, min_periods=5).median()
            mad = (series - med).abs().rolling(
                self.hampel_window_s, center=True, min_periods=5
            ).median()
            sigma = 1.4826 * mad
            # 对含 0 工况的字段（功率/电流），用非零段量程计算 min_sigma
            nonzero = series.dropna()
            nonzero = nonzero[nonzero.abs() > 1e-6]
            quant = nonzero.abs().median() if len(nonzero) else 1.0
            min_sigma = max(0.2, quant * 0.02)  # 至少 0.2 或典型量程 2%
            sigma = sigma.clip(lower=min_sigma)
            outliers = (series - med).abs() > self.hampel_n_sigma * sigma.replace(0, np.nan)

            # 3) 物理速率限制：单秒跳变超过物理上限即判为异常点
            rate = series.diff().abs()
            rate_limit = self.max_rate_per_s.get(col)
            if rate_limit is not None:
                rate_outliers = rate > rate_limit
                # 连续跳变段整体标记（避免只标记首点）
                rate_outliers = rate_outliers | rate_outliers.shift(-1, fill_value=False)
                outliers = outliers | rate_outliers

            n_out = int(outliers.fillna(False).sum())
            series = series.mask(outliers.fillna(False))

            report.outliers_removed += n_bad + n_out
            df[col] = series
        return df

    # ---------------- 缺失插补 ----------------

    def _impute_missing(self, df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
        """同工况相邻片段加权均值插补；退化策略：线性插值。"""
        for col in NUMERIC_COLUMNS:
            series = df[col]
            missing = series.isna()
            n_missing = int(missing.sum())
            if not n_missing:
                continue
            filled = self._condition_weighted_fill(series, df["operating_condition"])
            # 兜底：线性插值 + 前后填充，保证无残留缺失
            filled = filled.interpolate(method="linear", limit_direction="both")
            filled = filled.ffill().bfill()
            df[col] = filled.round(1)
            report.missing_filled += n_missing
        return df

    @staticmethod
    def _condition_weighted_fill(series: pd.Series, condition: pd.Series) -> pd.Series:
        """同工况相邻片段加权均值插补。

        对每个缺失点，取同工况下最近的前/后有效值，按距离倒数加权。
        """
        out = series.copy()
        cond_arr = condition.to_numpy()
        values = series.to_numpy(dtype=float)
        idx_all = np.arange(len(values))

        for i in idx_all[np.isnan(values)]:
            same_cond = cond_arr == cond_arr[i]
            valid = ~np.isnan(values) & same_cond
            prev_idx = idx_all[valid & (idx_all < i)]
            next_idx = idx_all[valid & (idx_all > i)]
            cands, weights = [], []
            if len(prev_idx):
                p = prev_idx[-1]
                cands.append(values[p]); weights.append(1.0 / (i - p))
            if len(next_idx):
                q = next_idx[0]
                cands.append(values[q]); weights.append(1.0 / (q - i))
            if cands:
                out.iloc[i] = np.average(cands, weights=weights)
        return out
