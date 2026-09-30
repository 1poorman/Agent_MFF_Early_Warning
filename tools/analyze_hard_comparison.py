#!/usr/bin/env python3
"""Audit completed hard-fault results and report subgroups, costs and paired changes."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hard_fault_metrics import HardFaultScorer

METHODS = ('full', 'lora', 'dpo', 'grpo')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(run):
    pipeline = read(run / 'pipeline.json')
    if pipeline['status'] != 'completed':
        raise ValueError('wait for the full pipeline to complete before analyzing')
    for name, expected in pipeline['source_sha256'].items():
        if digest(ROOT / name) != expected or digest(run / 'source' / name) != expected:
            raise ValueError(f'source mismatch: {name}')
    dataset_path = ROOT / 'data/sft_hard_v1/eval.jsonl'
    dataset = rows(dataset_path)
    dataset_hash = digest(dataset_path)
    if dataset_hash != pipeline['dataset_sha256']['eval']:
        raise ValueError('frozen evaluation dataset changed')
    scorer = HardFaultScorer(dataset_path.parent / 'ontology.json')
    comparison = read(run / 'comparison.json')
    all_rows, analysis, inference = {}, {}, None
    for method in METHODS:
        directory = run / method / 'eval'
        manifest = read(directory / 'evaluation_manifest.json')
        if (manifest['status'] != 'completed' or not manifest['temporary_service_stopped']
                or manifest['runs'] != 3 or manifest['dataset_sha256'] != dataset_hash):
            raise ValueError(f'incomplete evaluation: {method}')
        if inference is not None and inference != manifest['inference']:
            raise ValueError(f'inference settings differ: {method}')
        inference = manifest['inference']
        repetitions = []
        for repeat in (1, 2, 3):
            result = rows(directory / f'run{repeat}.jsonl')
            if len(result) != len(dataset):
                raise ValueError(f'incomplete result: {method}/{repeat}')
            for index, (sample, row) in enumerate(zip(dataset, result)):
                expected_input = hashlib.sha256(json.dumps(
                    sample['messages'][:-1], ensure_ascii=False).encode()).hexdigest()
                answer = json.loads(sample['messages'][-1]['content'])
                gold = {key: answer[key] for key in ('root_causes', 'needs_more_data')}
                if (row['index'] != index or row['input_sha256'] != expected_input
                        or row['gold'] != gold or row['meta'] != sample['meta'] or row['error']):
                    raise ValueError(f'failed or misaligned result: {method}/{repeat}/{index}')
                score = asdict(scorer.score(sample['messages'][1]['content'], row['raw_output'], gold))
                if score != row['score']:
                    raise ValueError(f'non-reproducible score: {method}/{repeat}/{index}')
            repetitions.append(result)
        all_rows[method] = repetitions
        summaries = comparison[method]['summaries']
        group_means = {
            group: {'n': summaries[0]['groups'][group]['n'], **{
                key: statistics.fmean(s['groups'][group][key] for s in summaries)
                for key in ('set_exact', 'joint_accuracy')}}
            for group in summaries[0]['groups']}
        model = ROOT / pipeline['plan']['models'][method]
        metrics = read(model / f'{method}_train_metrics.json')
        progress = rows(model / 'progress.jsonl')
        analysis[method] = {
            'groups': group_means,
            'train_metrics': metrics,
            'run_range': {key: [min(s[key] for s in summaries), max(s[key] for s in summaries)]
                          for key in ('set_exact', 'joint_accuracy')},
            'truncated': sum(s['truncated'] for s in summaries),
            'training_progress': progress,
        }
        monitor = model / 'reward_monitor.jsonl'
        if monitor.exists():
            analysis[method]['reward_monitor'] = rows(monitor)
    for method in ('full', 'dpo', 'grpo'):
        pairs = [(base, candidate) for baseline, candidates in zip(all_rows['lora'], all_rows[method])
                 for base, candidate in zip(baseline, candidates)]
        analysis[method]['paired_vs_lora'] = {
            'paired_predictions': len(pairs),
            'identical_raw_outputs': sum(a['raw_output'] == b['raw_output'] for a, b in pairs),
            **{key: {
                'improved': sum(not a['score'][key] and b['score'][key] for a, b in pairs),
                'regressed': sum(a['score'][key] and not b['score'][key] for a, b in pairs),
            } for key in ('set_exact', 'correct')},
        }
    return {'dataset_sha256': dataset_hash, 'n_per_run': len(dataset), 'runs': 3,
            'analysis_script_sha256': digest(__file__), 'methods': analysis}


def markdown(result, comparison):
    lines = ['# 八类故障后训练：分层结果与训练成本', '',
             '每个模型在同一冻结的 268 条输入上评估三次；以下准确率为三次推理均值。',
             '逐行输入、金标、元数据及原始输出重新评分均通过核验；各临时评估服务已停止。', '',
             '| 方法 | 根因集合匹配 | 集合＋不确定性正确 | Macro-F1 | Brier ↓ |',
             '|---|---:|---:|---:|---:|']
    for method in METHODS:
        m = comparison[method]['mean']
        brier = '—' if m['brier_joint'] is None else f"{m['brier_joint']:.4f}"
        lines.append(f"| {method} | {m['set_exact']:.2%} | {m['joint_accuracy']:.2%} | {m['macro_f1']:.4f} | {brier} |")
    lines += ['', '## 分层根因集合匹配率', '',
              '| 子集 | 样本数 | Full | LoRA | DPO | GRPO |', '|---|---:|---:|---:|---:|---:|']
    for key, label in (('compound', '复合故障'), ('missing_sensors', '缺失通道'),
                       ('shifted', '偏移工况'), ('unshifted', '原范围工况'), ('weak_signal', '弱信号')):
        groups = [result['methods'][m]['groups'][key] for m in METHODS]
        lines.append(f"| {label} | {groups[0]['n']} | " + ' | '.join(f"{g['set_exact']:.2%}" for g in groups) + ' |')
    lines += ['', '各子集可以重叠，不能相加；每个炉次的两个窗口同属一个集合。', '',
              '## 配对变化（相对 LoRA）', '',
              '| 方法 | 集合由错变对 | 集合由对变错 | 集合＋不确定性由错变对 | 集合＋不确定性由对变错 |',
              '|---|---:|---:|---:|---:|']
    for method in ('full', 'dpo', 'grpo'):
        p = result['methods'][method]['paired_vs_lora']
        lines.append(f"| {method} | {p['set_exact']['improved']} | {p['set_exact']['regressed']} | {p['correct']['improved']} | {p['correct']['regressed']} |")
    lines += ['', '计数来自 268 × 3 个配对预测，重复推理与同炉次窗口并非独立样本。', '',
              '## 训练成本', '',
              '| 方法 | 优化步数 | 完整命令耗时（分钟） | 采样训练进程显存峰值（MiB） |',
              '|---|---:|---:|---:|']
    for method in METHODS:
        a = result['methods'][method]
        r = comparison[method]['train_resources']
        lines.append(f"| {method} | {a['train_metrics']['global_step']} | {r['wall_seconds']/60:.2f} | {r['peak_child_gpu_memory_mib']:.0f} |")
    lines += ['', 'DPO、GRPO 的成本是 LoRA-SFT 之后的额外训练成本，不含合并和行为评估。',
              '显存每秒采样，可能遗漏短时峰值；PyTorch allocated/reserved 峰值另存于 analysis.json。', '',
              '训练种子均为 42；三次推理不代表三个训练种子，也不是等算力算法消融。',
              'DPO 使用合成规则偏好，GRPO 使用在线规则奖励；本实验没有人工或 AI 裁判反馈。',
              '置信度不参与八类奖励；Brier 以集合和不确定性同时判对为目标，仅使用格式可解析输出。',
              '训练置信度是合成教学值，不能将其拟合程度解释成真实设备上的概率校准。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'logs/hard_minicpm5_20260929')
    args = parser.parse_args()
    result = analyze(args.run_dir)
    (args.run_dir / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    report = markdown(result, read(args.run_dir / 'comparison.json'))
    (args.run_dir / 'analysis.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
