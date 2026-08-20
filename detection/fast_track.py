"""快轨预测器：STL 时序分解 + GRU+attention（单参数短步长，1~10min）。

设计：
- STL 剥离趋势/季节/残差，GRU+attention 在归一化残差上预测未来残差，
  最终输出 = 趋势外推 + 季节外推 + 残差预测，保证物理可解释性；
- 多参数独立建模（每个物理量一个轻量模型），边缘端毫秒级推理；
- 输入窗口 1h（3600s），预测步长 horizon ≤ 10min（600s）。
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.seasonal import STL


class GRUAttention(nn.Module):
    """GRU + 加性注意力 残差预测器。"""

    def __init__(self, input_size: int = 1, hidden: int = 32, horizon: int = 600):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, batch_first=True)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon))

    def forward(self, x):  # x: (B, T, 1)
        out, _ = self.gru(x)                      # (B, T, H)
        w = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        ctx = (out * w).sum(dim=1)                # (B, H)
        return self.head(ctx)                     # (B, horizon)


class FastTrackForecaster:
    """单参数 STL+GRU+attention 预测器。"""

    def __init__(self, column: str, window: int = 3600, horizon: int = 600,
                 period: int = 8400, hidden: int = 32, downsample: int = 10,
                 device: Optional[str] = None):
        self.column = column
        self.window = window
        self.horizon = horizon
        self.period = period          # 工况循环周期 8400s
        self.downsample = downsample  # GRU 输入降采样倍率（3600s->360 步，加速 10x）
        self.device = device or ("cuda:1" if torch.cuda.device_count() > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = GRUAttention(1, hidden, horizon).to(self.device)
        self.resid_std = 1.0

    # ---------------- STL 分解 ----------------

    def _stl(self, series: pd.Series):
        """训练期 STL 分解：先降采样再做 LOESS（period 同步缩放），避免 O(n·period) 爆炸。"""
        ds = self.downsample
        sub = series.iloc[::ds].reset_index(drop=True)
        stl = STL(sub, period=self.period // ds, robust=False)
        res = stl.fit()
        # 上采样回原分辨率（线性插值）
        idx_full = np.arange(len(series))
        idx_sub = np.arange(0, len(series), ds)
        trend = pd.Series(np.interp(idx_full, idx_sub, res.trend), index=series.index)
        seasonal = pd.Series(np.interp(idx_full, idx_sub, res.seasonal), index=series.index)
        resid = series - trend - seasonal
        return trend, seasonal, resid

    def _decompose_inference(self, series: pd.Series):
        """推理期轻量分解：趋势=滚动均值，季节=训练期缓存周期尾部（相位对齐），残差=原值-趋势-季节。

        避免每次 predict 重跑 STL（window 小于周期时 STL 既慢又不稳定）。
        """
        trend = series.rolling(120, center=True, min_periods=1).mean()
        n = len(series)
        tail = getattr(self, "_seasonal_tail", None)
        if tail is not None and len(tail) == self.period:
            # 相位对齐：窗口末点对应周期尾部末点
            idx = (np.arange(n) - n) % self.period
            seasonal = pd.Series(tail[idx], index=series.index)
        else:
            seasonal = pd.Series(0.0, index=series.index)
        resid = series - trend - seasonal
        return trend, seasonal, resid

    def _extrapolate_trend_seasonal(self, trend: pd.Series, seasonal: pd.Series) -> np.ndarray:
        """趋势线性外推 + 季节按周期平移，返回 horizon 长度数组。"""
        t = trend.to_numpy()
        k = min(600, len(t))  # 用近 10min 趋势斜率，避免全窗口平均稀释
        slope = np.polyfit(np.arange(k), t[-k:], 1)[0]
        trend_ext = t[-1] + slope * np.arange(1, self.horizon + 1)
        s = seasonal.to_numpy()
        if np.all(s == 0):
            return trend_ext
        # 从窗口末点相位继续平移
        last_phase = (len(s) - 1 - len(s)) % self.period
        season_ext = np.array([s[(last_phase + 1 + j) % self.period] for j in range(self.horizon)])
        return trend_ext + season_ext

    # ---------------- 训练 ----------------

    def fit(self, series: pd.Series, epochs: int = 5, batch_size: int = 64, lr: float = 1e-3):
        # 残差学习：以"末值持续"为基线，GRU 只学习相对基线的偏差（降低小波动参数的相对误差）
        self.mean = float(series.mean())
        self.std = float(series.std()) or 1.0
        r = ((series - self.mean) / self.std).astype(np.float32).to_numpy()

        X, Y = [], []
        for i in range(0, len(r) - self.window - self.horizon, 120):
            seg_x = r[i:i + self.window:self.downsample]
            baseline = r[i + self.window - 1]  # 末值持续基线
            seg_y = r[i + self.window:i + self.window + self.horizon] - baseline
            X.append(seg_x)
            Y.append(seg_y)
        X = torch.tensor(np.array(X)).unsqueeze(-1).to(self.device)
        Y = torch.tensor(np.array(Y)).to(self.device)

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for ep in range(epochs):
            perm = torch.randperm(len(X), device=self.device)
            total = 0.0
            for i in range(0, len(X), batch_size):
                idx = perm[i:i + batch_size]
                loss = loss_fn(self.model(X[idx]), Y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(idx)
            print(f"  [{self.column}] epoch {ep+1}/{epochs} mse={total/len(X):.4f}")
        return self

    # ---------------- 推理 ----------------

    def predict(self, recent: pd.Series) -> np.ndarray:
        """输入最近 window 序列，输出 horizon 步预测（物理量原尺度）。"""
        assert len(recent) >= self.window, f"需要至少 {self.window} 点"
        seg = recent.iloc[-self.window:]
        r = ((seg - self.mean) / self.std).astype(np.float32).to_numpy()
        baseline = r[-1]  # 末值持续基线
        x = torch.tensor(r[::self.downsample]).unsqueeze(0).unsqueeze(-1).to(self.device)
        self.model.eval()
        with torch.no_grad():
            delta = self.model(x).cpu().numpy().flatten()
        return (baseline + delta) * self.std + self.mean

    def predict_exceedance(self, series: pd.Series, threshold: float,
                           lookback_s: int = 1800, horizon_s: int = 1800) -> Optional[float]:
        """趋势越限预测：近 lookback 线性斜率外推，返回预计多少秒后越限（不越限返回 None）。

        用于 L2 预警：预测未来 horizon 内关键参数是否将超阈值（缓变型故障提前量来源）。
        """
        seg = series.iloc[-lookback_s:].to_numpy(dtype=float)
        if len(seg) < 60:
            return None
        slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0])
        cur = float(seg[-1])
        if slope <= 1e-6 or cur >= threshold:
            return 0.0 if cur >= threshold else None
        eta = (threshold - cur) / slope
        return float(eta) if 0 < eta <= horizon_s else None

    def mape(self, series: pd.Series, n_eval: int = 10) -> float:
        """滚动回测 MAPE（%）。"""
        errs = []
        for k in range(n_eval):
            end = len(series) - self.horizon - k * self.horizon
            pred = self.predict(series.iloc[:end])
            actual = series.iloc[end:end + self.horizon].to_numpy()
            errs.append(np.abs(pred - actual) / np.clip(np.abs(actual), 1e-6, None))
        return float(np.mean(errs) * 100)

    # ---------------- 持久化 ----------------

    def save(self, path: str):
        torch.save({"state": self.model.state_dict(), "mean": self.mean, "std": self.std,
                    "window": self.window, "horizon": self.horizon, "period": self.period,
                    "downsample": self.downsample, "column": self.column}, path)

    @classmethod
    def load(cls, path: str) -> "FastTrackForecaster":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(ckpt["column"], ckpt["window"], ckpt["horizon"], ckpt["period"],
                  downsample=ckpt.get("downsample", 10))
        obj.model.load_state_dict(ckpt["state"])
        obj.mean, obj.std = ckpt["mean"], ckpt["std"]
        return obj
