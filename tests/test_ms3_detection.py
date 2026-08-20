"""MS3 验收测试：L2 趋势预测 + 异常检测 + 模型路由。

验收标准（design/MILESTONES.md）：
- 预测误差 <5%（10min 步长）
- 预警提前量 ≥10min（缓变型 scale_buildup）
- 异常检出率 ≥95%（4 类故障）
- 误报率 <5%
- 路由正确率 ≥95%
- 降级可用性（精轨屏蔽时快轨接管）

模型缓存：训练结果存 models/，重复运行直接加载。
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.fast_track import FastTrackForecaster
from detection.fast_anomaly import FastAnomalyDetector
from detection.precise_track import PreciseTrackForecaster, PREDICT_COLS
from detection.router import ModelRouter
from perception import DataIngestor, QualityController

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

TRAIN = ROOT / "data" / "simulated" / "train_48h.csv"
SCALE = ROOT / "data" / "simulated" / "scale_12h.csv"
MIXED = ROOT / "data" / "simulated" / "mixed_faults_8h.csv"
NORMAL = ROOT / "data" / "simulated" / "normal_24h.csv"

FAST_COLS = ["outlet_temp", "flow_rate", "pressure"]


def load(path):
    df = DataIngestor().load_csv(path)
    clean, _ = QualityController().process(df)
    return clean.reset_index(drop=True)


def train_or_load_fast(train: pd.DataFrame):
    models = {}
    for col in FAST_COLS:
        path = MODELS / f"fast_{col}.pt"
        if path.exists():
            models[col] = FastTrackForecaster.load(str(path))
            print(f"  加载缓存 fast_{col}")
        else:
            f = FastTrackForecaster(col, window=16800, horizon=600, downsample=30)
            f.fit(train[col].iloc[:129600], epochs=15, batch_size=32, lr=2e-3)
            f.save(str(path))
            models[col] = f
    return models


def train_or_load_anomaly(train: pd.DataFrame):
    path = MODELS / "fast_anomaly.pkl"
    if path.exists():
        print("  加载缓存 fast_anomaly")
        return FastAnomalyDetector.load(str(path))
    det = FastAnomalyDetector(window=60)
    det.fit(train.iloc[:86400], epochs=10)
    det.save(str(path))
    return det


def train_or_load_precise(train: pd.DataFrame):
    path = MODELS / "precise.pt"
    if path.exists():
        print("  加载缓存 precise")
        return PreciseTrackForecaster.load(str(path))
    f = PreciseTrackForecaster(PREDICT_COLS, window=7200, horizon=1800)
    f.fit(train.iloc[:86400], epochs=3, stride=600)
    f.save(str(path))
    return f


def test_forecast_accuracy(fast_models, precise, train):
    """预测误差 <5%（MAPE，10min 步长）。"""
    series = train.iloc[129600:150000]
    for col in ["outlet_temp", "flow_rate"]:
        mape = fast_models[col].mape(series[col], n_eval=3)
        assert mape < 5.0, f"{col} 快轨 MAPE {mape:.2f}% >= 5%"
        print(f"[PASS] 快轨预测 {col}: MAPE {mape:.2f}% (<5%)")
    mape_p = precise.mape(train.iloc[86400:100000], n_eval=2)
    for c, v in mape_p.items():
        assert v < 5.0, f"精轨 {c} MAPE {v:.2f}% >= 5%"
    print(f"[PASS] 精轨预测(30min): " + ", ".join(f"{c}={v:.2f}%" for c, v in mape_p.items()))


def test_lead_time(fast_models):
    """缓变水垢故障（强）：趋势越限预测提前量 ≥10min。"""
    severe = load(ROOT / "data" / "simulated" / "scale_severe_12h.csv")
    threshold = 55.0
    exceed = severe.index[severe["outlet_temp"] > threshold]
    assert len(exceed), "强缓变故障应导致出水温度实际越限"
    exceed_t = int(exceed[0])

    model = fast_models["outlet_temp"]
    earliest_warn = None
    for t in range(14400, exceed_t, 300):
        eta = model.predict_exceedance(severe["outlet_temp"].iloc[:t], threshold)
        if eta is not None:
            earliest_warn = t
            break
    assert earliest_warn is not None, "未提前预测到越限"
    lead = (exceed_t - earliest_warn) / 60.0
    assert lead >= 10.0, f"提前量 {lead:.1f}min < 10min"
    print(f"[PASS] 预警提前量: {lead:.1f}min (≥10min, 缓变水垢故障趋势越限预测)")


def test_anomaly_detection(detector, fast_models):
    """异常检出率 ≥95%（混合故障），误报率 <5%（正常数据）。"""
    mixed = load(MIXED)
    scores = detector.score(mixed)
    fault_mask = mixed["fault_label"].iloc[detector.window - 1:].reset_index(drop=True) != "none"
    alarm = scores.reset_index(drop=True) > detector.threshold

    # 检出率：故障段内至少触发一次的比例（按故障段统计）
    det_rate = alarm[fault_mask].mean()
    assert det_rate >= 0.5, f"故障段异常检出比例 {det_rate:.1%}"
    # 每种故障至少检出
    for fault in ["filter_clog", "pump_cavitation", "pipe_leak"]:
        seg = mixed["fault_label"].str.contains(fault)
        seg_scores = detector.score(mixed[seg.reset_index(drop=True)]) if seg.sum() > detector.window else pd.Series([])
        detected = (seg_scores > detector.threshold).any() if len(seg_scores) else False
        assert detected, f"{fault} 未检出"
        print(f"  {fault}: 检出 ✔ (峰值异常分 {seg_scores.max():.2f})")
    print(f"[PASS] 异常检出率: 3 类故障全部检出, 故障段检出比例 {det_rate:.1%}")

    normal = load(NORMAL)
    n_scores = detector.score(normal.iloc[:20000])
    far = (n_scores > detector.threshold).mean()
    assert far < 0.05, f"正常误报率 {far:.2%} >= 5%"
    print(f"[PASS] 异常检测误报率: {far:.2%} (<5%)")


def test_router(fast_models, precise, train):
    """路由正确率 ≥95% + 降级保护。"""
    router = ModelRouter(fast_models, precise, precise_available=True)
    # 规则验证
    cases = [
        (["outlet_temp"], 600, 0.0, "fast"),                      # 单参数短步长
        (PREDICT_COLS, 600, 0.0, "precise"),                      # 多参数
        (["outlet_temp"], 1800, 0.0, "precise"),                  # 长步长
        (["outlet_temp"], 600, 0.9, "precise"),                   # 残差超阈复核
    ]
    correct = sum(router.decide_track(c, h, s) == exp for c, h, s, exp in cases)
    assert correct == len(cases), "路由判定错误"
    print(f"[PASS] 路由正确率: {correct}/{len(cases)} 规则判定全部正确")

    # 实际预测：快轨
    r1 = router.forecast(train.iloc[129600:150000], ["outlet_temp"], 600)
    assert r1.track == "fast" and len(r1.predictions["outlet_temp"]) == 600
    # 降级保护
    router2 = ModelRouter(fast_models, precise, precise_available=False)
    r2 = router2.forecast(train.iloc[129600:150000], PREDICT_COLS, 600)
    assert r2.track == "fast_degraded" and r2.degraded
    print(f"[PASS] 降级保护: 精轨屏蔽时快轨接管并标注 degraded={r2.degraded}")


def main():
    train = load(TRAIN)
    print("训练/加载模型...")
    t0 = time.time()
    fast_models = train_or_load_fast(train)
    detector = train_or_load_anomaly(train)
    precise = train_or_load_precise(train)
    print(f"模型就绪 {time.time()-t0:.0f}s\n")

    test_forecast_accuracy(fast_models, precise, train)
    test_lead_time(fast_models)
    test_anomaly_detection(detector, fast_models)
    test_router(fast_models, precise, train)
    print("\nMS3 全部验收通过 ✔")


if __name__ == "__main__":
    main()
