#!/usr/bin/env python3
"""Export observed training curves and behavioral results as PNG and SVG."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="logs/rl_minicpm5_20260928")
    ap.add_argument("--out-prefix", default="design/assets/minicpm5_rl_comparison")
    args = ap.parse_args()
    root = Path(args.run_dir)
    result = json.loads((root / "comparison.json").read_text())
    histories = {}
    for method in ("dpo", "grpo"):
        directory = root / method / "adapter"
        if json.loads((directory / "run_manifest.json").read_text())["status"] != "completed":
            raise ValueError(f"{method} training is incomplete")
        histories[method] = [json.loads(s) for s in (directory / "progress.jsonl").read_text().splitlines()]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    dpo = [r for r in histories["dpo"] if "loss" in r]
    axes[0, 0].plot([r["step"] for r in dpo], [r["loss"] for r in dpo], color="#2b6cb0")
    axes[0, 0].set(title="DPO: training preference loss", xlabel="Optimizer step", ylabel="Loss")
    grpo = [r for r in histories["grpo"] if "reward" in r]
    for key, label, color in (("reward", "Mean reward", "#2b6cb0"),
                              ("frac_reward_zero_std", "Groups with zero reward variance", "#c05621")):
        axes[0, 1].plot([r["step"] for r in grpo], [r[key] for r in grpo], label=label, color=color)
    axes[0, 1].set(title="GRPO: online reward and learning signal", xlabel="Optimizer step", ylim=(0, 1.04))
    axes[0, 1].legend(fontsize=8)
    names = list(result["arms"])
    colors = ["#718096", "#2b6cb0", "#c05621", "#2f855a"]
    for ax, key, title, scale in ((axes[1, 0], "refusal_rate", "Frozen eval: correct data-quality refusal", 100),
                                  (axes[1, 1], "brier", "Frozen eval: Brier score (lower is better)", 1)):
        values = [result["arms"][name]["mean"][key] * scale for name in names]
        bars = ax.bar(names, values, color=colors[:len(names)])
        ax.bar_label(bars, fmt="%.1f" if scale == 100 else "%.4f", padding=3)
        ax.set_title(title)
        ax.set_ylim(0, 110 if scale == 100 else max(values) * 1.25)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("MiniCPM5-2B post-training comparison\nOne training seed; three repeated inference runs; unequal training budgets", fontsize=13)
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(out.with_suffix(suffix), dpi=180)
    plt.close(fig)
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
