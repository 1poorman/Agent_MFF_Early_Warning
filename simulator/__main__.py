"""CLI 入口。

示例：
    # 生成 24h 正常数据
    python -m simulator --duration 86400 --out data/simulated/normal_24h.csv

    # 生成 4h 数据，第 1 小时注入过滤器堵塞故障（10 分钟爬升到 80% 程度）
    python -m simulator --duration 14400 \
        --faults "filter_clog@3600:600:0.8" \
        --out data/simulated/filter_clog_4h.csv
"""

import argparse

from .faults import FAULT_REGISTRY, parse_fault_spec
from .generator import DataSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="中频炉水冷系统数据模拟智能体")
    parser.add_argument("--duration", type=float, required=True, help="仿真时长（秒）")
    parser.add_argument("--out", required=True, help="输出文件路径")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv", help="输出格式")
    parser.add_argument("--start", default="2026-08-20 00:00:00", help="起始时间戳")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--faults", nargs="*", default=[],
        help=f"故障表达式 名称@起始秒:爬升秒:程度(0~1)，可选: {list(FAULT_REGISTRY)}",
    )
    args = parser.parse_args()

    from .config import SimConfig
    cfg = SimConfig(start_time=args.start, seed=args.seed)
    faults = [parse_fault_spec(f) for f in args.faults]
    sim = DataSimulator(config=cfg, faults=faults)
    out = sim.run_to_file(args.out, args.duration, fmt=args.format)
    print(f"已生成 {args.duration:.0f}s ({args.duration/3600:.1f}h) 数据 -> {out}")
    if faults:
        print("注入故障:", [f"{f.name}@{f.start}s" for f in faults])


if __name__ == "__main__":
    main()
