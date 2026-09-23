"""训练 L2 异常检测轨 Anomaly-Transformer（仅正常工况数据）。

用法：
    conda run -n mff_agent python -u tools/train_anomaly_transformer.py            # 全量（86k 窗口, 3 epochs）
    conda run -n mff_agent python -u tools/train_anomaly_transformer.py --rows 20000 --epochs 1   # 快速冒烟

产出 models/at_anomaly.pt（config/settings.yaml detection.anomaly_at_weights 指向），
server 启动时按 detection.anomaly_model=anomaly_transformer 自动加载；
训练后自动在混合故障/正常数据上评测（检出率 + 误报率）。
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from perception import DataIngestor, QualityController


def load(path: Path) -> "pd.DataFrame":
    df = DataIngestor().load_csv(str(path))
    clean, _ = QualityController().process(df)
    return clean.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="训练 Anomaly-Transformer 异常检测器")
    ap.add_argument("--train", default="data/simulated/train_48h.csv")
    ap.add_argument("--rows", type=int, default=86400, help="训练使用的正常数据行数")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default=None, help="cuda:1 / cuda / cpu（默认自动）")
    ap.add_argument("--save", default="models/at_anomaly.pt")
    ap.add_argument("--mixed", default="data/simulated/mixed_faults_8h.csv")
    ap.add_argument("--normal", default="data/simulated/normal_24h.csv")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    import torch
    from detection.anomaly_transformer import AnomalyTransformerDetector

    device = args.device or ("cuda:1" if torch.cuda.device_count() > 1
                             else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"加载数据 {args.train} ...")
    train = load(ROOT / args.train)
    print(f"训练行数 {min(args.rows, len(train))}（共 {len(train)}）| device={device}")

    det = AnomalyTransformerDetector(
        window=args.window, d_model=args.d_model, n_heads=args.heads,
        e_layers=args.layers, batch_size=args.batch_size, lr=args.lr,
        threshold=0.6, device=device)
    t0 = time.time()
    det.fit(train.iloc[:args.rows], epochs=args.epochs)
    print(f"训练完成 {time.time() - t0:.0f}s")

    save_path = ROOT / args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    det.save(str(save_path))
    print(f"权重已保存: {save_path}")

    if args.no_eval:
        return
    thr = det.threshold
    try:
        mixed = load(ROOT / args.mixed)
        scores = det.score(mixed)
        mask = mixed["fault_label"].iloc[det.window - 1:].reset_index(drop=True) != "none"
        rate = (scores.reset_index(drop=True) > thr)[mask].mean()
        print(f"[评测] 混合故障检出比例: {rate:.1%}（故障段异常分>阈值占比）")
        for fault in ["filter_clog", "pump_cavitation", "pipe_leak", "scale_buildup"]:
            seg = mixed["fault_label"].str.contains(fault)
            if int(seg.sum()) > det.window:
                s = det.score(mixed[seg.reset_index(drop=True)])
                print(f"  {fault}: 峰值异常分 {s.max():.2f} "
                      f"{'检出 ✔' if (s > thr).any() else '未检出 ✘'}")
    except Exception as e:
        print(f"[评测] 混合故障数据不可用: {e}")
    try:
        normal = load(ROOT / args.normal)
        n_scores = det.score(normal.iloc[:20000])
        far = (n_scores > thr).mean()
        print(f"[评测] 正常数据误报率: {far:.2%}（要求 <5%）")
    except Exception as e:
        print(f"[评测] 正常数据不可用: {e}")


if __name__ == "__main__":
    main()
