"""多变量快轨预测器：STL 残差学习 + 物理关联辅助通道（GRU+attention）。

相对单变量快轨（fast_track.py）的增强：
- 目标通道（outlet_temp）保持残差学习（末值持续基线，稳定低 MAPE）；
- 辅助通道（inlet_temp / flow_rate / electric_power / furnace_temp / pressure）
  按各自 z-score 归一后作为 GRU 额外输入通道，使模型可感知「热源→换热→水力」
  链路的联动状态（如功率升+流量降预示温升）；
- GRUAttention 天然支持 input_size>1，网络结构不变。

输入契约：pd.DataFrame，需含 target + aux_cols 全部列（detection.forecaster.MV_FEATURES）。
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from .fast_track import GRUAttention

TARGET = "outlet_temp"
# 辅助通道：物理关联见 detection/forecaster.py MV_FEATURES 注释
DEFAULT_AUX = ["inlet_temp", "flow_rate", "electric_power", "furnace_temp", "pressure"]


class FastTrackMultivarForecaster:
    """多变量 STL+GRU+attention 快轨预测器。"""

    def __init__(self, target: str = TARGET, aux_cols: Optional[List[str]] = None,
                 window: int = 16800, horizon: int = 600, period: int = 8400,
                 hidden: int = 32, downsample: int = 30, device: Optional[str] = None):
        self.target = target
        self.aux_cols = list(aux_cols) if aux_cols else list(DEFAULT_AUX)
        self.channels = [target] + self.aux_cols
        self.window = window
        self.horizon = horizon
        self.period = period
        self.downsample = downsample
        self.device = device or ("cuda:1" if torch.cuda.device_count() > 1
                                 else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = GRUAttention(len(self.channels), hidden, horizon).to(self.device)
        self.mean: Optional[np.ndarray] = None      # 各通道 z-score 均值
        self.std: Optional[np.ndarray] = None

    # ---------------- 归一 ----------------

    def _z(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.channels].to_numpy(dtype=np.float32)
        return (X - self.mean) / self.std

    # ---------------- 训练 ----------------

    def fit(self, df: pd.DataFrame, epochs: int = 15, batch_size: int = 32,
            lr: float = 2e-3, stride: int = 120):
        """残差学习：目标通道以末值持续为基线，GRU 预测相对基线的偏差。"""
        self.mean = df[self.channels].mean().to_numpy(dtype=np.float32)
        self.std = np.maximum(df[self.channels].std().to_numpy(dtype=np.float32), 1e-6)
        Z = self._z(df)

        X, Y = [], []
        for i in range(0, len(Z) - self.window - self.horizon, stride):
            seg_x = Z[i:i + self.window:self.downsample]                  # (T, C)
            baseline = Z[i + self.window - 1, 0]                          # 目标通道末值
            seg_y = Z[i + self.window:i + self.window + self.horizon, 0] - baseline
            X.append(seg_x)
            Y.append(seg_y)
        X = torch.tensor(np.array(X)).to(self.device)
        Y = torch.tensor(np.array(Y)).to(self.device)

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        self.model.train()
        for ep in range(epochs):
            perm = torch.randperm(len(X), device=self.device)
            total = 0.0
            for i in range(0, len(X), batch_size):
                idx = perm[i:i + batch_size]
                loss = loss_fn(self.model(X[idx]), Y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(idx)
            print(f"  [MV-GRU] epoch {ep+1}/{epochs} mse={total/len(X):.4f}")
        return self

    # ---------------- 推理 ----------------

    def predict(self, recent: pd.DataFrame) -> np.ndarray:
        """输入最近 window 条多变量数据，输出目标列 horizon 步预测（原尺度）。"""
        assert len(recent) >= self.window, f"需要至少 {self.window} 点"
        seg = self._z(recent.iloc[-self.window:])
        baseline = seg[-1, 0]
        x = torch.tensor(seg[::self.downsample]).unsqueeze(0).to(self.device)   # (1, T, C)
        self.model.eval()
        with torch.no_grad():
            delta = self.model(x).cpu().numpy().flatten()
        return (baseline + delta) * self.std[0] + self.mean[0]

    def predict_exceedance(self, series: pd.Series, threshold: float,
                           lookback_s: int = 1800, horizon_s: int = 1800) -> Optional[float]:
        """趋势越限预测（与单变量快轨一致：近 lookback 线性斜率外推）。"""
        seg = series.iloc[-lookback_s:].to_numpy(dtype=float)
        if len(seg) < 60:
            return None
        slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0])
        cur = float(seg[-1])
        if slope <= 1e-6 or cur >= threshold:
            return 0.0 if cur >= threshold else None
        eta = (threshold - cur) / slope
        return float(eta) if 0 < eta <= horizon_s else None

    def mape(self, df: pd.DataFrame, n_eval: int = 3, stride: int = 1200) -> float:
        """滚动回测 MAPE（%），末端 n_eval 个窗口。"""
        errs = []
        for k in range(n_eval):
            end = len(df) - self.horizon - k * stride
            if end < self.window:
                continue
            pred = self.predict(df.iloc[:end])
            actual = df[self.target].iloc[end:end + self.horizon].to_numpy(dtype=float)
            errs.append(np.abs(pred - actual) / np.clip(np.abs(actual), 1e-6, None))
        return float(np.mean(errs) * 100)

    # ---------------- 持久化 ----------------

    def save(self, path: str):
        torch.save({"state": self.model.state_dict(), "mean": self.mean, "std": self.std,
                    "target": self.target, "aux_cols": self.aux_cols,
                    "window": self.window, "horizon": self.horizon, "period": self.period,
                    "downsample": self.downsample}, path)

    @classmethod
    def load(cls, path: str) -> "FastTrackMultivarForecaster":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(ckpt["target"], ckpt["aux_cols"], ckpt["window"], ckpt["horizon"],
                  ckpt["period"], downsample=ckpt.get("downsample", 30))
        obj.model.load_state_dict(ckpt["state"])
        obj.mean, obj.std = ckpt["mean"], ckpt["std"]
        return obj
