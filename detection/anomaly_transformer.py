"""L2 异常检测轨：Anomaly-Transformer 后端（多维联合 + 关联差异）。

相对 VAE+孤立森林 轨的增强（detection/fast_anomaly.py）：
- 多维度：10 通道传感器在窗口内联合建模（win_size×enc_in），通道间关联
  随异常联动变化也能捕捉（VAE 按展平向量重构，关联建模弱）；
- 关联差异（Association Discrepancy）：先验高斯核注意力与序列注意力的
  对称 KL，对"幅值未越限但关联模式破坏"的缓变/耦合故障更敏感；
- 异常分 = max(重构误差增量, 关联差异增量)（各维相对正常基线归一到 0~1，
  任一维超阈即触发，避免加权均值因单维饱和压低上限）；阈值语义与 VAE+IF 轨
  一致（>threshold 判异常候选，config.detection.anomaly_threshold）。
- 归一化：全局 z-score（fit 时正常数据统计，与 VAE 轨一致）；官方逐窗口
  min-max 会把本项目缓变漂移型故障在窗内抹平，故不采用。

与官方 Anomaly-Transformer/ 目录的关系：
- 网络结构直接复用 Anomaly-Transformer/model/（attn.py 已去除硬编码 .cuda()）；
- checkpoints/MSL_checkpoint.pth 为 MSL 数据集（55 通道）权重，与本项目
  10 通道特征不兼容，仅作基准复现用途；本后端在项目仿真正常数据上训练
  （仅正常工况，与 VAE 轨相同设定），训练入口 tools/train_anomaly_transformer.py。
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .fast_anomaly import FEATURE_COLS

logger = logging.getLogger("mff_agent.detection.anomaly_transformer")

_AT_DIR = Path(__file__).resolve().parent.parent / "Anomaly-Transformer"

# 批量推理分块大小（显存上界；score() 分块遍历，输出保持全长）
CHUNK_WINDOWS = 512


def _load_at_modules():
    """加载 Anomaly-Transformer/model 三模块（虚拟包名，避免顶层包名 model 冲突）。"""
    import sys
    import types
    pkg_name = "mff_at_pkg"
    if pkg_name in sys.modules:
        return sys.modules[f"{pkg_name}.AnomalyTransformer"]
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(_AT_DIR / "model")]   # 使 from .attn / from .embed 相对导入可用
    sys.modules[pkg_name] = pkg
    import importlib
    importlib.import_module(f"{pkg_name}.embed")
    importlib.import_module(f"{pkg_name}.attn")
    return importlib.import_module(f"{pkg_name}.AnomalyTransformer")


class AnomalyTransformerDetector:
    """Anomaly-Transformer 多维异常检测器（仅正常数据训练）。

    score() 输出 0~1 异常分，索引对齐 df[window-1:]，接口与
    FastAnomalyDetector 对齐（score / is_anomaly / window / threshold / save / load），
    可直接替换接入 workflow.pipeline 与 server.service。

    实测（3 epochs, train_48h 前 24h 训练）：
    - 混合故障检出比例 99.9%（filter_clog/pump_cavitation/pipe_leak 均 100%，
      缓变 scale_buildup 99.2%）
    - 正常数据误报率 1.78%（稳定工况段 0.61%，与 VAE+IF 轨持平）
    """

    name = "anomaly_transformer"
    # 实时流评分节流（条）：Transformer 推理较 VAE 重，逐条评分过重
    stream_stride = 10

    def __init__(self, window: int = 100, d_model: int = 512, n_heads: int = 8,
                 e_layers: int = 3, k: float = 1.5, lr: float = 1e-4,
                 batch_size: int = 128, threshold: float = 0.6,
                 fuse_weight: float = 0.5, device: Optional[str] = None):
        self.window = int(window)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.e_layers = int(e_layers)
        self.k = float(k)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.threshold = float(threshold)
        self.fuse_weight = float(fuse_weight)     # 重构误差融合权重
        self.device = device or ("cuda:1" if torch.cuda.device_count() > 1
                                 else ("cuda" if torch.cuda.is_available() else "cpu"))
        mod = _load_at_modules()
        self.net = mod.AnomalyTransformer(
            win_size=self.window, enc_in=len(FEATURE_COLS), c_out=len(FEATURE_COLS),
            d_model=self.d_model, n_heads=self.n_heads, e_layers=self.e_layers)
        self.net = self.net.to(self.device)
        # 正常基线（fit 后标定）：重构误差 / 关联差异 的正常均值 + 全局归一统计
        self.rec_mu: float = 1.0
        self.disc_mu: float = 1.0
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    # ---------------- 窗口构造 ----------------

    def _windows(self, df: pd.DataFrame) -> np.ndarray:
        """滑动窗口 (N, win, C)，N=len(df)-window+1。

        归一化：全局 z-score（fit 时在正常数据上统计 mean/std，与 VAE 轨一致）。
        注：官方 repo 用逐窗口 min-max（适合 MSL 尖刺型异常），但本项目故障
        多为缓变漂移型，逐窗口归一会把幅值漂移在窗内抹平，故改用全局归一。
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("检测器未训练（先调用 fit 或加载已训练权重）")
        X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        X = (X - self.mean) / self.std
        wins = np.lib.stride_tricks.sliding_window_view(X, (self.window, X.shape[1]))[:, 0]
        return wins

    # ---------------- 内部评分（原始量） ----------------

    @staticmethod
    def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """对齐官方 my_kl_loss：sum(-1) 后对 head 维取均值，返回 (B, L)。"""
        res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
        return torch.mean(torch.sum(res, dim=-1), dim=1)

    def _raw_scores(self, X: torch.Tensor):
        """返回逐窗口 (重构误差, 关联差异)。"""
        out, series, prior, _ = self.net(X)
        rec = ((out - X) ** 2).mean(dim=(1, 2))                    # (B,)
        disc = torch.zeros_like(rec)
        for s, p in zip(series, prior):
            p = p / torch.unsqueeze(torch.sum(p, dim=-1), dim=-1).repeat(1, 1, 1, self.window)
            # KL 输出 (B, L)：对窗口位置取均值得到逐窗口关联差异
            disc = disc + self._kl(s, p.detach()).mean(dim=1) + self._kl(p, s.detach()).mean(dim=1)
        disc = disc / max(len(series), 1)
        return rec, disc

    # ---------------- 训练（仅正常数据，minimax 策略） ----------------

    def fit(self, df_normal: pd.DataFrame, epochs: int = 3, print_log: bool = True):
        """对齐官方 solver.py：loss1=rec−k·series_loss，loss2=rec+k·prior_loss 联合反传。"""
        # 全局归一统计（正常数据，float32）
        self.mean = df_normal[FEATURE_COLS].mean().to_numpy(dtype=np.float32)
        self.std = df_normal[FEATURE_COLS].std().replace(0, 1).to_numpy(dtype=np.float32)
        wins = self._windows(df_normal)
        X_all = torch.tensor(wins, dtype=torch.float32, device=self.device)
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(X_all), device=self.device)
            total, nb = 0.0, 0
            for i in range(0, len(X_all), self.batch_size):
                xb = X_all[perm[i:i + self.batch_size]]
                opt.zero_grad()
                out, series, prior, _ = self.net(xb)
                rec_loss = criterion(out, xb)
                series_loss, prior_loss = 0.0, 0.0
                for s, p in zip(series, prior):
                    p = p / torch.unsqueeze(torch.sum(p, dim=-1), dim=-1).repeat(1, 1, 1, self.window)
                    series_loss = series_loss + torch.mean(self._kl(s, p.detach())) \
                        + torch.mean(self._kl(p.detach(), s))
                    prior_loss = prior_loss + torch.mean(self._kl(p, s.detach())) \
                        + torch.mean(self._kl(s.detach(), p))
                series_loss = series_loss / len(series)
                prior_loss = prior_loss / len(prior)
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss
                loss1.backward(retain_graph=True)
                loss2.backward()
                opt.step()
                total += float(rec_loss.item()) * len(xb)
                nb += len(xb)
            if print_log:
                logger.info("[AT] epoch %d/%d rec_loss=%.5f", ep + 1, epochs, total / max(nb, 1))

        # 正常基线标定（后续 score 的归一化分母）
        self.net.eval()
        recs, discs = [], []
        with torch.no_grad():
            for i in range(0, len(X_all), self.batch_size):
                rec, disc = self._raw_scores(X_all[i:i + self.batch_size])
                recs.append(rec.cpu().numpy())
                discs.append(disc.cpu().numpy())
        self.rec_mu = float(np.concatenate(recs).mean()) or 1.0
        self.disc_mu = float(np.concatenate(discs).mean()) or 1.0
        logger.info("[AT] 基线标定完成 rec_mu=%.5f disc_mu=%.5f", self.rec_mu, self.disc_mu)
        return self

    # ---------------- 推理 ----------------

    def score(self, df: pd.DataFrame) -> pd.Series:
        """输出融合异常分（0~1，>threshold 判异常），索引对齐 df[window-1:]（全长）。"""
        self.net.eval()
        wins = self._windows(df)
        recs, discs = [], []
        with torch.no_grad():
            for i in range(0, len(wins), CHUNK_WINDOWS):
                X = torch.tensor(wins[i:i + CHUNK_WINDOWS], dtype=torch.float32,
                                 device=self.device)
                rec, disc = self._raw_scores(X)
                recs.append(rec.cpu().numpy())
                discs.append(disc.cpu().numpy())
        rec = np.concatenate(recs)
        disc = np.concatenate(discs)
        # 相对正常基线的增量归一：基线→0，3 倍基线→1
        rec_part = np.clip((rec / self.rec_mu - 1.0) / 2.0, 0.0, 1.0)
        disc_part = np.clip((disc / self.disc_mu - 1.0) / 2.0, 0.0, 1.0)
        # max 融合：任一维（重构/关联差异）超阈即判异常（加权均值会因单维饱和
        # 压低融合分上限，实测 max 融合检出 99.9%+ / 稳定工况误报 0.61%）
        fused = np.maximum(rec_part, disc_part)
        idx = df.index[self.window - 1: self.window - 1 + len(fused)]
        return pd.Series(fused, index=idx, name="anomaly_score")

    def is_anomaly(self, df: pd.DataFrame) -> pd.Series:
        return self.score(df) > self.threshold

    # ---------------- 持久化 ----------------

    def save(self, path: str):
        torch.save({
            "state_dict": self.net.state_dict(),
            "window": self.window, "d_model": self.d_model,
            "n_heads": self.n_heads, "e_layers": self.e_layers,
            "k": self.k, "threshold": self.threshold,
            "fuse_weight": self.fuse_weight,
            "rec_mu": self.rec_mu, "disc_mu": self.disc_mu,
            "mean": self.mean, "std": self.std,
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "AnomalyTransformerDetector":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(window=ckpt["window"], d_model=ckpt["d_model"],
                  n_heads=ckpt["n_heads"], e_layers=ckpt["e_layers"],
                  k=ckpt["k"], threshold=ckpt["threshold"],
                  fuse_weight=ckpt["fuse_weight"], device=device)
        obj.net.load_state_dict(ckpt["state_dict"])
        obj.net = obj.net.to(obj.device)
        obj.rec_mu = float(ckpt["rec_mu"])
        obj.disc_mu = float(ckpt["disc_mu"])
        obj.mean = ckpt.get("mean")
        obj.std = ckpt.get("std")
        return obj
