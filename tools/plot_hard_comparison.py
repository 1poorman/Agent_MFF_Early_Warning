#!/usr/bin/env python3
"""Plot completed eight-fault training and frozen-evaluation results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('logs/hard_minicpm5_20260929'))
    parser.add_argument('--analysis', type=Path, help='analysis.json from a separate audit directory')
    parser.add_argument('--out-prefix', type=Path, default=Path('design/assets/minicpm5_hard_comparison'))
    args = parser.parse_args()
    analysis = json.loads((args.analysis or args.run_dir / 'analysis.json').read_text())['methods']
    comparison = json.loads((args.run_dir / 'comparison.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {'full': '#64748b', 'lora': '#2563eb', 'dpo': '#c2410c', 'grpo': '#15803d'}
    for method in ('full', 'lora'):
        history = [r for r in analysis[method]['training_progress'] if 'loss' in r]
        axes[0, 0].plot([r['step'] for r in history], [r['loss'] for r in history],
                        label=method.upper(), color=colors[method])
    axes[0, 0].set(title='SFT: supervised training loss', xlabel='Optimizer step', ylabel='Loss')
    axes[0, 0].legend()
    history = [r for r in analysis['dpo']['training_progress'] if 'loss' in r]
    axes[0, 1].plot([r['step'] for r in history], [r['loss'] for r in history], color=colors['dpo'])
    axes[0, 1].set(title='DPO: training preference loss', xlabel='Optimizer step', ylabel='Loss')
    history = [r for r in analysis['grpo']['training_progress'] if 'reward' in r]
    for key, label, color in (('reward', 'Mean reward', colors['grpo']),
                              ('frac_reward_zero_std', 'Groups with zero reward variance', colors['dpo'])):
        axes[1, 0].plot([r['step'] for r in history], [r[key] for r in history], label=label, color=color)
    axes[1, 0].set(title='GRPO: online reward and learning signal', xlabel='Optimizer step', ylim=(0, 1.04))
    axes[1, 0].legend(fontsize=8)
    methods = list(colors)
    for index, (metric, label, offset, alpha) in enumerate((
            ('set_exact', 'Root-cause set exact match', -.2, 1),
            ('joint_accuracy', 'Set + uncertainty correct', .2, .5))):
        values = [comparison[m]['mean'][metric] * 100 for m in methods]
        low = [values[i] - analysis[m]['run_range'][metric][0] * 100 for i, m in enumerate(methods)]
        high = [analysis[m]['run_range'][metric][1] * 100 - values[i] for i, m in enumerate(methods)]
        bars = axes[1, 1].bar([i + offset for i in range(4)], values, width=.38,
                             color=list(colors.values()), alpha=alpha, label=label,
                             yerr=[low, high], capsize=3)
        axes[1, 1].bar_label(bars, fmt='%.1f', fontsize=8, padding=4)
    axes[1, 1].set(title='Frozen eval: 268 inputs, three runs', ylabel='Accuracy (%)', ylim=(0, 110),
                   xticks=range(4), xticklabels=[m.upper() for m in methods])
    axes[1, 1].legend(fontsize=8, loc='lower left')
    fig.suptitle('MiniCPM5-2B: eight-fault post-training comparison\n'
                 'One training seed; unequal training budgets; error bars show inference min/max', fontsize=13)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.svg'):
        fig.savefig(args.out_prefix.with_suffix(suffix), dpi=180)
    plt.close(fig)
    print(args.out_prefix.with_suffix('.png'))


if __name__ == '__main__':
    main()
