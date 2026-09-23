"""训练多变量快轨预测器（STL 残差学习 + 物理关联辅助通道）。

输入通道（物理关联，见 detection/forecaster.py MV_FEATURES）：
    outlet_temp(目标) + inlet_temp / flow_rate / electric_power / furnace_temp / pressure

用法：
    conda run -n mff_agent python -u tools/train_fast_mv.py            # 全量（对齐 MS3 协议）
    conda run -n mff_agent python -u tools/train_fast_mv.py --epochs 2 # 快速冒烟

产出 models/fast_outlet_temp_mv.pt，服务启动时 GRU 后端自动优先加载。
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from perception import DataIngestor, QualityController


def load(path: str):
    df = DataIngestor().load_csv(path)
    clean, _ = QualityController().process(df)
    return clean.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/simulated/train_48h.csv")
    ap.add_argument("--rows", type=int, default=129600, help="训练行数（MS3 协议 36h）")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--window", type=int, default=16800)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--downsample", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--save", default="models/fast_outlet_temp_mv.pt")
    args = ap.parse_args()

    from detection.fast_track_mv import FastTrackMultivarForecaster
    from detection.forecaster import MV_FEATURES, MV_TARGET

    print(f"加载数据 {args.train} ...")
    train = load(str(ROOT / args.train))
    print(f"通道 {MV_FEATURES} | 训练行数 {min(args.rows, len(train))}")

    m = FastTrackMultivarForecaster(MV_TARGET, MV_FEATURES[1:], window=args.window,
                                    horizon=args.horizon, downsample=args.downsample,
                                    hidden=args.hidden)
    t0 = time.time()
    m.fit(train.iloc[:args.rows], epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    print(f"训练完成 {time.time()-t0:.0f}s")

    save = ROOT / args.save
    m.save(str(save))
    print(f"权重已保存: {save}")

    # 回测（与 MS3 同口径：评估段 train.iloc[129600:150000]）
    ev = train.iloc[129600:150000]
    if len(ev) > m.window + m.horizon:
        print(f"回测 MAPE: {m.mape(ev, n_eval=3):.2f}%（单变量快轨基线 1.05%）")


if __name__ == "__main__":
    main()
