"""快速异常检测轨：VAE 自编码器 + 孤立森林融合。

- VAE 仅用正常工况数据训练，学习正常模式的低维流形；
  异常样本的重构误差显著增大；
- 孤立森林在 [重构误差, 一阶差分能量] 残差特征空间中切割孤立点；
- 融合分 = 0.5·VAE归一化残差 + 0.5·IF异常分，>0.6 判为异常候选。
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

FEATURE_COLS = [
    "inlet_temp", "outlet_temp", "pressure", "flow_rate",
    "tank_level", "conductivity", "cabinet_temp", "cabinet_humidity",
    "furnace_temp", "electric_power",
]


class VAE(nn.Module):
    def __init__(self, in_dim: int, latent: int = 8, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.mu = nn.Linear(hidden // 2, latent)
        self.logvar = nn.Linear(hidden // 2, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(z), mu, logvar


class FastAnomalyDetector:
    """VAE + 孤立森林 融合异常检测器。"""

    def __init__(self, window: int = 60, latent: int = 8, device: Optional[str] = None,
                 vae_weight: float = 0.5, threshold: float = 0.6):
        self.window = window
        self.device = device or ("cuda:1" if torch.cuda.device_count() > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vae = VAE(len(FEATURE_COLS) * window, latent).to(self.device)
        self.iforest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
        self.vae_weight = vae_weight
        self.threshold = threshold
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.vae_err_scale = 1.0

    # ---------------- 特征 ----------------

    def _windows(self, df: pd.DataFrame) -> np.ndarray:
        """滑动窗口样本: (N, window, n_feat) -> flatten。返回最后点索引对齐。"""
        X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        X = (X - self.mean) / self.std
        wins = np.lib.stride_tricks.sliding_window_view(X, (self.window, X.shape[1]))
        wins = wins[:, 0]                       # (N, window, n_feat)
        return wins.reshape(len(wins), -1)

    def _vae_score(self, X: torch.Tensor) -> np.ndarray:
        self.vae.eval()
        with torch.no_grad():
            recon, _, _ = self.vae(X)
            err = ((recon - X) ** 2).mean(dim=1)
        return err.cpu().numpy()

    # ---------------- 训练（仅正常数据） ----------------

    def fit(self, df_normal: pd.DataFrame, epochs: int = 10, batch_size: int = 128, lr: float = 1e-3):
        self.mean = df_normal[FEATURE_COLS].mean().to_numpy(dtype=np.float32)
        self.std = df_normal[FEATURE_COLS].std().replace(0, 1).to_numpy(dtype=np.float32)
        X = torch.tensor(self._windows(df_normal)).to(self.device)

        opt = torch.optim.Adam(self.vae.parameters(), lr=lr)
        self.vae.train()
        for ep in range(epochs):
            perm = torch.randperm(len(X), device=self.device)
            total = 0.0
            for i in range(0, len(X), batch_size):
                xb = X[perm[i:i + batch_size]]
                recon, mu, logvar = self.vae(xb)
                loss = ((recon - xb) ** 2).mean() + 1e-3 * (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean())
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(xb)
            print(f"  [VAE] epoch {ep+1}/{epochs} loss={total/len(X):.4f}")

        # 正常数据残差尺度，用于归一化
        self.vae_err_scale = float(self._vae_score(X).mean()) or 1.0
        # 孤立森林拟合残差特征
        self.iforest.fit(self._if_features(df_normal))
        return self

    def _if_features(self, df: pd.DataFrame) -> np.ndarray:
        """IF 特征: [VAE归一化残差, 各通道差分能量均值]。"""
        X = torch.tensor(self._windows(df)).to(self.device)
        vae_s = self._vae_score(X) / self.vae_err_scale
        raw = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        d_energy = np.abs(np.diff(raw, axis=0)).mean(axis=1)
        d_energy = np.concatenate([[d_energy[0]], d_energy])
        d_win = np.array([d_energy[i:i + self.window].mean() for i in range(len(d_energy) - self.window + 1)])
        return np.stack([vae_s, d_win], axis=1)

    # ---------------- 推理 ----------------

    def score(self, df: pd.DataFrame) -> pd.Series:
        """输出融合异常分（0~1+，>threshold 判异常），索引对齐 df[window-1:]。"""
        feats = self._if_features(df)
        vae_part = np.clip(feats[:, 0] / (3.0), 0, 1)          # 正常均值≈1，3σ 截断归一
        if_part = np.clip(-self.iforest.score_samples(feats), 0, 1)
        fused = self.vae_weight * vae_part + (1 - self.vae_weight) * if_part
        idx = df.index[self.window - 1:]
        return pd.Series(fused, index=idx, name="anomaly_score")

    def is_anomaly(self, df: pd.DataFrame) -> pd.Series:
        return self.score(df) > self.threshold

    # ---------------- 持久化 ----------------

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "vae": self.vae.state_dict(), "iforest": self.iforest,
                "mean": self.mean, "std": self.std, "window": self.window,
                "vae_err_scale": self.vae_err_scale,
            }, f)

    @classmethod
    def load(cls, path: str) -> "FastAnomalyDetector":
        import pickle
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        obj = cls(window=ckpt["window"])
        obj.vae.load_state_dict(ckpt["vae"])
        obj.iforest = ckpt["iforest"]
        obj.mean, obj.std = ckpt["mean"], ckpt["std"]
        obj.vae_err_scale = ckpt["vae_err_scale"]
        return obj
