"""训练 PatchTST 通道数消融变体（1/4/6 通道），回答通道选择问题。

与 MS3 精轨协议完全一致（window=7200, horizon=1800, epochs=3, stride=600,
训练数据 train_48h 前 24h），仅通道组合不同：
    - models/precise_uni.pt : [outlet_temp]                 单通道
    - models/precise.pt     : PREDICT_COLS (4 通道)         既有基线
    - models/precise_mv6.pt : MV_FEATURES (6 通道)          多变量版
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from perception import DataIngestor, QualityController
from detection.precise_track import PreciseTrackForecaster, PREDICT_COLS
from detection.forecaster import MV_FEATURES

VARIANTS = {
    "precise_uni.pt": ["outlet_temp"],
    "precise_mv6.pt": MV_FEATURES,
}


def main():
    df = DataIngestor().load_csv(str(ROOT / "data/simulated/train_48h.csv"))
    clean, _ = QualityController().process(df)
    train = clean.reset_index(drop=True)
    print(f"数据 {len(train)} 行")

    for fname, cols in VARIANTS.items():
        path = ROOT / "models" / fname
        if path.exists():
            print(f"跳过 {fname}（已存在）")
            continue
        print(f"\n>>> 训练 {fname} 通道={cols}")
        m = PreciseTrackForecaster(cols, window=7200, horizon=1800)
        t0 = time.time()
        m.fit(train.iloc[:86400], epochs=3, stride=600)
        m.save(str(path))
        print(f"完成 {time.time()-t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
