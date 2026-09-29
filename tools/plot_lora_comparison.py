#!/usr/bin/env python3
"""Plot measured training loss and GPU memory for the MiniCPM LoRA comparison."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-dir", default="saves/mff-sft-full")
    parser.add_argument("--lora-dir", default="saves/mff-lora-minicpm5-2b-20260923")
    parser.add_argument("--measurement-dir", default="logs/lora_minicpm5_20260923/train")
    parser.add_argument("--out-prefix", default="design/assets/minicpm5_lora_training")
    args = parser.parse_args()
    full = read_json(Path(args.full_dir) / "trainer_state.json")["log_history"]
    lora_dir = Path(args.lora_dir)
    progress = read_jsonl(lora_dir / "progress.jsonl")
    memory = read_jsonl(Path(args.measurement_dir) / "gpu_samples.jsonl")
    resources = read_json(Path(args.measurement_dir) / "resource_summary.json")
    metrics = read_json(lora_dir / "lora_train_metrics.json")
    if resources["status"] != "completed" or metrics["global_step"] != 441:
        raise RuntimeError("plot requires the completed 441-step experiment")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for rows, label, color in [(full, "Full SFT v3", "#3366a8"),
                               (progress, "LoRA r=32", "#da7339")]:
        rows = [row for row in rows if "loss" in row]
        axes[0].plot([row["step"] for row in rows], [row["loss"] for row in rows],
                     label=label, color=color, linewidth=1.7)
    validation = {}
    for row in progress:
        if "eval_loss" in row:
            validation[row["step"]] = row["eval_loss"]
    axes[0].scatter(list(validation), list(validation.values()), label="LoRA validation",
                    marker="D", color="#293f33", s=24, zorder=3)
    axes[0].set(yscale="log", xlabel="Optimizer step", ylabel="Cross-entropy loss",
                title="Training and validation loss", xlim=(0, 450))
    axes[0].grid(alpha=0.18, which="both")
    axes[0].legend(frameon=False)

    elapsed = [row["elapsed_seconds"] / 60 for row in memory]
    axes[1].plot(elapsed, [row["gpu_used_mib"] / 1024 for row in memory],
                 color="#8b9bb0", label="Whole GPU (includes other jobs)", linewidth=1.2)
    axes[1].plot(elapsed, [(row["child_gpu_memory_mib"] or 0) / 1024 for row in memory],
                 color="#da7339", label="Training process (NVIDIA)", linewidth=1.6)
    active_peak = metrics["cuda_memory_train_including_periodic_eval"]["peak_allocated_mib"] / 1024
    axes[1].axhline(active_peak, color="#293f33", linestyle="--", linewidth=1.2,
                    label=f"PyTorch peak allocated: {active_peak:.2f} GiB")
    started = datetime.fromisoformat(resources["started_utc"])
    for i, row in enumerate(row for row in progress if "eval_runtime" in row):
        end = (datetime.fromisoformat(row["time_utc"]) - started).total_seconds() / 60
        start = end - row["eval_runtime"] / 60
        axes[1].axvspan(start, end, color="#526556", alpha=0.2,
                        label="Validation (batch 4)" if i == 0 else None)
    axes[1].set(xlabel="Minutes since process launch", ylabel="GPU memory (GiB)",
                title="LoRA memory across the complete run", ylim=(0, 24),
                xlim=(0, resources["wall_seconds"] / 60))
    axes[1].grid(alpha=0.18)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("MiniCPM5-2B | 2,337 train / 259 validation | RTX 3090 | 3 epochs", fontsize=12)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = prefix.with_suffix("." + extension)
        fig.savefig(path, dpi=180, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
