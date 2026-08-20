"""MS1 感知层验收测试。

验收标准（design/MILESTONES.md）：
- 时间戳完整度 100%（无丢包/乱序）
- 插补误差 <2%
- 数据精度 1 位小数
- 吞吐 ≥1Hz 实时处理
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception import DataIngestor, QualityController
from perception.ingest import NUMERIC_COLUMNS

DATA = Path(__file__).resolve().parent.parent / "data" / "simulated" / "normal_24h.csv"


def load_ground_truth() -> pd.DataFrame:
    return DataIngestor().load_csv(DATA)


def test_schema_and_baseline():
    df = load_ground_truth()
    assert len(df) == 86400, f"基准数据条数异常: {len(df)}"
    ts = df["timestamp"]
    assert ts.is_monotonic_increasing, "基准数据时间戳非递增"
    qc = QualityController()
    clean, report = qc.process(df)
    assert report.completeness == 1.0, f"完整度 {report.completeness}"
    assert clean[NUMERIC_COLUMNS].isna().sum().sum() == 0, "处理后仍有缺失"
    diffs = clean["timestamp"].diff().dropna().dt.total_seconds()
    assert (diffs == 1.0).all(), "输出时间戳非 1s 连续"
    print(f"[PASS] 基准数据处理: {report.summary()}")
    return clean


def test_missing_imputation(clean: pd.DataFrame):
    """注入 5% 随机缺失 + 时间戳空洞，校验插补误差 <2%。"""
    rng = np.random.default_rng(0)
    dirty = clean.copy()

    # 1) 数值缺失 5%
    mask = rng.random((len(dirty), len(NUMERIC_COLUMNS))) < 0.05
    for j, col in enumerate(NUMERIC_COLUMNS):
        dirty.loc[mask[:, j], col] = np.nan

    # 2) 时间戳空洞：删除 3 段各 10 分钟
    drop_idx = []
    for start in (10000, 40000, 70000):
        drop_idx.extend(range(start, start + 600))
    holes_truth = clean.iloc[drop_idx].copy()
    dirty = dirty.drop(index=dirty.index[drop_idx]).reset_index(drop=True)

    qc = QualityController()
    fixed, report = qc.process(dirty)

    assert report.completeness == 1.0, f"完整度 {report.completeness}"
    diffs = fixed["timestamp"].diff().dropna().dt.total_seconds()
    assert (diffs == 1.0).all(), "补洞后时间戳非 1s 连续"
    assert len(fixed) == len(clean), "补洞后条数与基准不一致"

    # 插补误差：单点缺失（非空洞段）
    merged = fixed.set_index("timestamp")[NUMERIC_COLUMNS]
    truth = clean.set_index("timestamp")[NUMERIC_COLUMNS]
    single_mask = pd.DataFrame(mask, columns=NUMERIC_COLUMNS, index=truth.index)
    # 排除空洞段（真实值被删除的点不参与单点误差统计）
    hole_ts = set(holes_truth["timestamp"])
    rel_errs = []
    for col in NUMERIC_COLUMNS:
        m = single_mask[col]
        m[[t in hole_ts for t in m.index]] = False
        err = (merged.loc[m, col] - truth.loc[m, col]).abs() / truth.loc[m, col].abs().clip(lower=1e-6)
        rel_errs.append(err.median())
    med_err = float(np.median(rel_errs))
    assert med_err < 0.02, f"插补中位相对误差 {med_err:.2%} >= 2%"
    print(f"[PASS] 缺失插补: {report.summary()} | 中位相对误差 {med_err:.3%} (<2%)")


def test_outlier_removal(clean: pd.DataFrame):
    """注入尖刺与超物理值，校验全部剔除且插补回合理范围。"""
    rng = np.random.default_rng(1)
    dirty = clean.copy()
    n_spike = 50
    idx = rng.choice(len(dirty), n_spike, replace=False)
    dirty.loc[idx, "outlet_temp"] = dirty.loc[idx, "outlet_temp"] + 40.0  # 尖刺
    dirty.loc[idx[:10], "pressure"] = 999.0                                # 超物理值

    qc = QualityController()
    fixed, report = qc.process(dirty)
    assert report.outliers_removed >= n_spike, f"剔除数不足: {report.outliers_removed}"
    assert fixed["pressure"].max() < 600, "超物理压力未剔除"
    # 尖刺点修复后应回到邻域
    repaired = fixed.loc[idx, "outlet_temp"].to_numpy()
    truth = clean.loc[idx, "outlet_temp"].to_numpy()
    assert np.abs(repaired - truth).max() < 5.0, "尖刺修复后偏离真实值过大"
    print(f"[PASS] 异常点剔除: 剔除 {report.outliers_removed} 个, 修复后最大偏差 "
          f"{np.abs(repaired - truth).max():.2f}℃ (<5℃)")


def test_precision_and_throughput(clean: pd.DataFrame):
    """精度 1 位小数 + 1h 数据回放吞吐 ≥1Hz。"""
    qc = QualityController()
    fixed, _ = qc.process(clean)
    for col in NUMERIC_COLUMNS:
        decimals = (fixed[col] * 10).round(6) % 1
        assert (decimals < 1e-4).all(), f"{col} 精度超过 1 位小数"

    t0 = time.perf_counter()
    qc.process(clean.iloc[:3600])
    elapsed = time.perf_counter() - t0
    assert elapsed < 3600, f"处理 1h 数据耗时 {elapsed:.1f}s，低于实时"
    print(f"[PASS] 精度与吞吐: 全字段 1 位小数 | 1h 数据处理 {elapsed:.2f}s (实时倍率 {3600/elapsed:.0f}x)")


def main():
    clean = test_schema_and_baseline()
    test_missing_imputation(clean)
    test_outlier_removal(clean)
    test_precision_and_throughput(clean)
    print("\nMS1 全部验收通过 ✔")


if __name__ == "__main__":
    main()
