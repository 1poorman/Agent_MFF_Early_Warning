"""精轨预测器：多参数联合多步长预测（10~30min）。

主力：通道独立 PatchTST 风格 Transformer（对齐题目 TCN/Transformer 类时序技术），
多参数共享编码器，输出头分别预测各通道；
增强：预留 Anomaly-Transformer / timesfm 适配接口（enhance_backend）。
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PREDICT_COLS = ["outlet_temp", "inlet_temp", "pressure", "flow_rate"]


class PatchTST(nn.Module):
    """通道独立 PatchTST：patch 切分 + Transformer 编码 + 线性输出头。"""

    def __init__(self, n_channels: int, window: int, horizon: int,
                 patch_len: int = 120, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = window // patch_len
        self.embed = nn.Linear(patch_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(self.n_patches * d_model, horizon)
        self.n_channels = n_channels
        self.horizon = horizon

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        x = x.reshape(B * C, T // self.patch_len, self.patch_len)
        z = self.encoder(self.embed(x))          # (B*C, N, D)
        z = z.reshape(B * C, -1)
        out = self.head(z)                       # (B*C, horizon)
        return out.reshape(B, C, self.horizon)


class PreciseTrackForecaster:
    """多参数联合预测器（精轨）。"""

    def __init__(self, columns: Optional[List[str]] = None, window: int = 7200,
                 horizon: int = 1800, device: Optional[str] = None):
        self.columns = columns or PREDICT_COLS
        self.window = window        # 2h 历史
        self.horizon = horizon      # 30min 预测
        self.device = device or ("cuda:1" if torch.cuda.device_count() > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PatchTST(len(self.columns), window, horizon).to(self.device)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, epochs: int = 3, batch_size: int = 32, lr: float = 1e-3, stride: int = 300):
        data = df[self.columns].to_numpy(dtype=np.float32)
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + 1e-6
        d = (data - self.mean) / self.std

        X, Y = [], []
        for i in range(0, len(d) - self.window - self.horizon, stride):
            X.append(d[i:i + self.window].T)
            Y.append(d[i + self.window:i + self.window + self.horizon].T)
        X = torch.tensor(np.array(X)).to(self.device)
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
            print(f"  [PatchTST] epoch {ep+1}/{epochs} mse={total/len(X):.4f}")
        return self

    def predict(self, recent: pd.DataFrame) -> Dict[str, np.ndarray]:
        """输入最近 window 多参数序列，输出各通道 horizon 步预测。"""
        assert len(recent) >= self.window
        d = (recent[self.columns].iloc[-self.window:].to_numpy(dtype=np.float32) - self.mean) / self.std
        x = torch.tensor(d.T).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x).cpu().numpy()[0]   # (C, horizon)
        out = out * self.std[:, None] + self.mean[:, None]
        return {c: out[i] for i, c in enumerate(self.columns)}

    def mape(self, df: pd.DataFrame, n_eval: int = 3) -> Dict[str, float]:
        errs = {c: [] for c in self.columns}
        for k in range(n_eval):
            end = len(df) - self.horizon - k * self.horizon
            pred = self.predict(df.iloc[:end])
            for c in self.columns:
                actual = df[c].iloc[end:end + self.horizon].to_numpy()
                errs[c].append(np.abs(pred[c] - actual) / np.clip(np.abs(actual), 1e-6, None))
        return {c: float(np.mean(v) * 100) for c, v in errs.items()}

    def save(self, path: str):
        torch.save({"state": self.model.state_dict(), "mean": self.mean, "std": self.std,
                    "columns": self.columns, "window": self.window, "horizon": self.horizon}, path)

    @classmethod
    def load(cls, path: str) -> "PreciseTrackForecaster":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(ckpt["columns"], ckpt["window"], ckpt["horizon"])
        obj.model.load_state_dict(ckpt["state"])
        obj.mean, obj.std = ckpt["mean"], ckpt["std"]
        return obj
