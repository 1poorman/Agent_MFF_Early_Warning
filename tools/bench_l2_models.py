"""L2 模型基准测试：预测任务与检测任务横向对比。

对比维度（预测轨）：gru / patchtst / timesfm / chronos
  - 准确性：MAPE（出水温度，600s 步长，正常段 + 缓变故障段）
  - 模型大小：参数量（M）+ 权重磁盘占用（MB）
  - 推理显存：加载增量 + 单次推理峰值（MB，cuda）
  - 推理速度：单次预测耗时（ms，含 warmup）
  - 输入窗口：所需最小上下文（条，决定实时流预加载时长）

对比维度（检测轨）：vae / anomaly_transformer
  - 准确性：故障段检出比例、3 类故障峰值异常分、正常误报率
  - 模型大小：磁盘占用
  - 推理显存：同上
  - 推理速度：单窗口（120 条，stream_step 口径）评分耗时 ms

用法：
    conda run -n mff_agent python -u tools/bench_l2_models.py                # 全部
    conda run -n mff_agent python -u tools/bench_l2_models.py --task forecast
    conda run -n mff_agent python -u tools/bench_l2_models.py --task detect
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
DEV = torch.device(DEVICE)
HORIZON = 600
EVAL_WINDOWS = 3          # 每个数据集的评估窗口数
EVAL_STRIDE = 1200        # 评估窗口间隔（条）
SPEED_RUNS = 5            # 预测速度测量次数
SCORE_RUNS = 20           # 检测速度测量次数（120 条窗口）
STREAM_WINDOW = 120       # stream_step 实际评分窗口

MFORECAST = ["gru", "patchtst", "timesfm", "chronos"]
MDETECT = ["vae", "anomaly_transformer"]


def load_clean(rel: str) -> pd.DataFrame:
    from perception import DataIngestor, QualityController
    df = DataIngestor().load_csv(str(ROOT / rel))
    clean, _ = QualityController().process(df)
    return clean.reset_index(drop=True)


def gpu_mem(fn):
    """返回 fn() 前后的显存增量（MB）与执行结果。"""
    torch.cuda.synchronize(DEV)
    base = torch.cuda.memory_allocated(DEV)
    torch.cuda.reset_peak_memory_stats(DEV)
    ret = fn()
    torch.cuda.synchronize(DEV)
    load_mb = (torch.cuda.memory_allocated(DEV) - base) / 1e6
    peak_mb = (torch.cuda.max_memory_allocated(DEV) - base) / 1e6
    return load_mb, peak_mb, ret


def disk_mb(*paths) -> float:
    total = 0
    for p in paths:
        fp = ROOT / p
        if fp.is_dir():
            total += sum(f.stat().st_size for f in fp.rglob("*") if f.is_file())
        elif fp.is_file():
            total += fp.stat().st_size
    return total / 1e6


def dir_params(*paths) -> float:
    """统计 safetensors 权重目录参数量（M）。"""
    n = 0
    for p in paths:
        fp = ROOT / p
        if not fp.is_dir():
            continue
        for f in sorted(fp.rglob("*.safetensors")):
            from safetensors import safe_open
            with safe_open(str(f), framework="pt") as sf:
                for k in sf.keys():
                    t = sf.get_slice(k)
                    shape = t.get_shape()
                    n += int(np.prod(shape)) if shape else 0
    return n / 1e6


def count_params(obj) -> float:
    """从加载后的对象统计 nn.Module 参数量（M），递归遍历常见属性路径（去重）。"""
    import torch.nn as nn
    seen, total = set(), 0

    def walk(o, depth=0):
        nonlocal total
        if o is None or depth > 3 or id(o) in seen:
            return
        if isinstance(o, nn.Module):
            seen.add(id(o))
            total += sum(p.numel() for p in o.parameters())
            return
        for attr in ("model", "pipeline", "net", "vae", "inner_model"):
            walk(getattr(o, attr, None), depth + 1)

    walk(obj)
    return total / 1e6


# ---------------- 预测轨 ----------------

def load_forecast_backend(name: str):
    """返回 (predict_fn, window, weights_paths, model_obj, label)。"""
    if name == "gru":
        from detection.fast_track import FastTrackForecaster
        m = FastTrackForecaster.load(str(ROOT / "models/fast_outlet_temp.pt"))

        def fn(recent):
            return np.asarray(m.predict(recent.iloc[-m.window:]))[:HORIZON]
        return fn, m.window, ["models/fast_outlet_temp.pt"], m, "GRU+attention 快轨"
    if name == "patchtst":
        from detection.precise_track import PreciseTrackForecaster
        m = PreciseTrackForecaster.load(str(ROOT / "models/precise.pt"))

        def fn(recent):
            df = pd.DataFrame({m.columns[0]: recent.iloc[-m.window:]})
            for c in m.columns[1:]:
                if c not in df.columns:
                    df[c] = 0.0
            preds = m.predict(df)
            return np.asarray(preds.get("outlet_temp", preds.get(m.columns[0])))[:HORIZON]
        return fn, m.window, ["models/precise.pt"], m, "PatchTST 精轨(多参数)"
    if name == "timesfm":
        from detection.forecaster import TimesFMBackend
        b = TimesFMBackend(weights_dir=str(ROOT / "timesfm/weights"), device=DEVICE)
        return (lambda recent: b.predict(recent, HORIZON)), b.context_len, \
            ["timesfm/weights"], b, "TimesFM-2.5 基础模型(200M)"
    if name == "chronos":
        from detection.forecaster import ChronosBackend
        b = ChronosBackend(weights_dir=str(ROOT / "models/chronos-2"),
                           context_len=1024, device=DEVICE)
        return (lambda recent: b.predict(recent, HORIZON)), b.context_len, \
            ["models/chronos-2"], b, "Chronos-2 基础模型(120M)"
    raise ValueError(name)


def mape(pred: np.ndarray, actual: np.ndarray) -> float:
    mask = np.abs(actual) > 1e-6
    return float(np.mean(np.abs((pred - actual)[mask]) / np.abs(actual[mask])) * 100)


def eval_accuracy(fn, window, series: pd.Series) -> float:
    """末端评估段内取 EVAL_WINDOWS 个窗口，600s 步长 MAPE 均值。"""
    ma = []
    n = len(series)
    for i in range(EVAL_WINDOWS):
        t = n - (EVAL_WINDOWS - i) * EVAL_STRIDE
        if t - window < 0 or t + HORIZON > n:
            continue
        ctx = series.iloc[t - window: t]
        actual = series.iloc[t: t + HORIZON].to_numpy(dtype=float)
        ma.append(mape(np.asarray(fn(ctx), dtype=float), actual))
    return float(np.mean(ma)) if ma else float("nan")


def measure_speed(fn, series: pd.Series, window: int, runs: int = SPEED_RUNS) -> float:
    ctx = series.iloc[-window:]
    fn(ctx)                                   # warmup
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(ctx)
        torch.cuda.synchronize(DEV)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts))


def bench_forecast(names):
    print("=" * 88)
    print("预测任务对比（出水温度 outlet_temp，步长 600s，device=%s）" % DEVICE)
    print("=" * 88)
    normal = load_clean("data/simulated/normal_24h.csv")
    scale = load_clean("data/simulated/scale_12h.csv")     # 缓变故障
    train = load_clean("data/simulated/train_48h.csv")
    rows = []
    for name in names:
        print(f"\n>>> {name}")
        torch.cuda.empty_cache()
        load_mb, _, (fn, window, wpaths, mobj, label) = gpu_mem(lambda: load_forecast_backend(name))
        _, peak_mb, _ = gpu_mem(lambda: fn(normal["outlet_temp"].iloc[-window:]))
        print(f"    加载显存增量 {load_mb:.0f}MB | 单次推理峰值 {peak_mb:.0f}MB")
        r = {"model": name, "label": label, "window": window,
             "params_M": count_params(mobj), "disk_MB": disk_mb(*wpaths),
             "load_vram_MB": load_mb, "infer_vram_MB": peak_mb}
        r["mape_normal"] = eval_accuracy(fn, window, normal["outlet_temp"])
        r["mape_train"] = eval_accuracy(fn, window, train["outlet_temp"])
        r["mape_scale"] = eval_accuracy(fn, window, scale["outlet_temp"])
        r["latency_ms"] = measure_speed(fn, normal["outlet_temp"], window)
        print(f"    MAPE 正常段={r['mape_normal']:.2f}% 训练段={r['mape_train']:.2f}% "
              f"缓变故障段={r['mape_scale']:.2f}% | 延迟={r['latency_ms']:.0f}ms")
        rows.append(r)
        del fn
        torch.cuda.empty_cache()
    print("\n" + "-" * 88)
    hdr = f"{'模型':<12}{'窗口':>7}{'参数M':>9}{'磁盘MB':>9}{'加载显存':>10}{'推理显存':>10}{'延迟ms':>9}{'MAPE正常':>10}{'MAPE故障':>10}"
    print(hdr)
    for r in rows:
        print(f"{r['model']:<12}{r['window']:>7}{r['params_M']:>9.2f}{r['disk_MB']:>9.1f}"
              f"{r['load_vram_MB']:>10.0f}{r['infer_vram_MB']:>10.0f}{r['latency_ms']:>9.0f}"
              f"{r['mape_normal']:>10.2f}{r['mape_scale']:>10.2f}")
    return rows


# ---------------- 检测轨 ----------------

def load_detector(name: str):
    if name == "vae":
        from detection.fast_anomaly import FastAnomalyDetector
        det = FastAnomalyDetector.load(str(ROOT / "models/fast_anomaly.pkl"))
        det.vae = det.vae.to(DEV)
        return det, ["models/fast_anomaly.pkl"], det, "VAE+孤立森林"
    if name == "anomaly_transformer":
        from detection.anomaly_transformer import AnomalyTransformerDetector
        det = AnomalyTransformerDetector.load(str(ROOT / "models/at_anomaly.pt"), device=DEVICE)
        return det, ["models/at_anomaly.pt"], det, "Anomaly-Transformer"
    raise ValueError(name)


def bench_detect(names):
    print("=" * 88)
    print("检测任务对比（异常检测轨，device=%s）" % DEVICE)
    print("=" * 88)
    mixed = load_clean("data/simulated/mixed_faults_8h.csv")
    normal = load_clean("data/simulated/normal_24h.csv")
    rows = []
    for name in names:
        print(f"\n>>> {name}")
        torch.cuda.empty_cache()
        load_mb, _, (det, wpaths, mobj, label) = gpu_mem(lambda: load_detector(name))
        _, peak_mb, _ = gpu_mem(lambda: det.score(normal.iloc[:2000]))
        thr = det.threshold
        # 检出：混合故障段
        scores = det.score(mixed)
        mask = mixed["fault_label"].iloc[det.window - 1:].reset_index(drop=True) != "none"
        det_rate = float((scores.reset_index(drop=True) > thr)[mask].mean())
        peaks = {}
        for fault in ["filter_clog", "pump_cavitation", "pipe_leak"]:
            seg = mixed[mixed["fault_label"].str.contains(fault)]
            peaks[fault] = float(det.score(seg).max())
        # 误报：正常数据 + 稳定工况段
        n_scores = det.score(normal)
        far = float((n_scores > thr).mean())
        if "operating_condition" in normal.columns:
            trans = normal["operating_condition"].ne(
                normal["operating_condition"].shift()).rolling(600, min_periods=1).max()
            stable = trans.iloc[det.window - 1:].to_numpy() == 0
            far_stable = float((n_scores.to_numpy() > thr)[stable].mean())
        else:
            far_stable = far
        # 速度：单窗口（120 条，stream_step 口径）+ 批量（2000 条）
        w = normal.iloc[:STREAM_WINDOW]
        det.score(w)
        ts = []
        for _ in range(SCORE_RUNS):
            t0 = time.perf_counter()
            det.score(w)
            torch.cuda.synchronize(DEV)
            ts.append((time.perf_counter() - t0) * 1000)
        lat_stream = float(np.mean(ts))
        t0 = time.perf_counter()
        det.score(normal.iloc[:2000])
        torch.cuda.synchronize(DEV)
        lat_batch = (time.perf_counter() - t0) * 1000

        r = {"model": name, "label": label, "window": det.window,
             "params_M": count_params(mobj), "disk_MB": disk_mb(*wpaths),
             "load_vram_MB": load_mb, "infer_vram_MB": peak_mb,
             "detection_rate": det_rate, "far": far, "far_stable": far_stable,
             "peaks": peaks, "lat_stream_ms": lat_stream, "lat_batch_ms": lat_batch}
        print(f"    检出 {det_rate:.1%} | FAR {far:.2%}(稳定段 {far_stable:.2f}) | "
              f"单窗口 {lat_stream:.1f}ms 批量2000条 {lat_batch:.0f}ms")
        rows.append(r)
        del det
        torch.cuda.empty_cache()
    print("\n" + "-" * 88)
    print(f"{'模型':<22}{'窗口':>6}{'磁盘MB':>9}{'加载显存':>10}{'推理显存':>10}"
          f"{'检出':>8}{'FAR':>8}{'单窗口ms':>10}{'批量ms':>9}")
    for r in rows:
        print(f"{r['model']:<22}{r['window']:>6}{r['disk_MB']:>9.1f}{r['load_vram_MB']:>10.0f}"
              f"{r['infer_vram_MB']:>10.0f}{r['detection_rate']:>8.1%}{r['far']:>8.2%}"
              f"{r['lat_stream_ms']:>10.1f}{r['lat_batch_ms']:>9.0f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["forecast", "detect", "all"], default="all")
    ap.add_argument("--models", default="", help="逗号分隔子集，如 gru,chronos")
    ap.add_argument("--out", default="logs/bench_l2.json")
    args = ap.parse_args()
    result = {}
    if args.task in ("forecast", "all"):
        names = args.models.split(",") if args.models else MFORECAST
        result["forecast"] = bench_forecast([n.strip() for n in names])
    if args.task in ("detect", "all"):
        names = args.models.split(",") if args.models and args.task == "detect" else MDETECT
        result["detect"] = bench_detect([n.strip() for n in names])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
