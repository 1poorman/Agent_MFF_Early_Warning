#!/usr/bin/env python3
"""Audit completed hard-fault results and report subgroups, costs and paired changes."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hard_fault_metrics import HardFaultScorer, summarize

METHODS = ('full', 'lora', 'dpo', 'grpo')
METRICS = ('set_exact', 'joint_accuracy', 'macro_f1', 'json_valid', 'evidence_traceable',
           'sop_recall', 'uncertainty_accuracy', 'uncertainty_precision', 'uncertainty_recall',
           'normal_false_positive', 'brier_joint')
GROUPS = {
    'compound': lambda r: len(r['meta']['injected_faults']) > 1,
    'missing_sensors': lambda r: bool(r['meta']['hidden_channels']),
    'shifted': lambda r: r['meta']['shifted_envelope'],
    'unshifted': lambda r: not r['meta']['shifted_envelope'],
    'weak_signal': lambda r: bool(r['meta']['injected_faults']) and
        max(f['severity'] for f in r['meta']['injected_faults']) < 0.3,
}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_equal(actual, expected, location):
    """Compare recomputed metrics recursively, including denominators and nulls."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f'summary mismatch: {location}')
        for key, value in expected.items():
            require_equal(actual[key], value, f'{location}/{key}')
    elif isinstance(expected, float):
        if type(actual) not in (int, float) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'summary mismatch: {location}')
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f'summary mismatch: {location}')


def audit_summary(result, names, saved, location):
    expected = summarize(result, names)
    expected['groups'] = {name: summarize([r for r in result if select(r)], names)
                          for name, select in GROUPS.items()}
    expected['truncated'] = sum(r['finish_reason'] == 'length' for r in result)
    if not expected.keys() <= saved.keys():
        raise ValueError(f'summary mismatch: {location}/missing fields')
    require_equal({key: saved[key] for key in expected}, expected, location)


def cluster_intervals(all_rows, *, resamples=10000, seed=42):
    """Paired percentile bootstrap of episodes; retain every window and repeat.

    The estimand remains the pooled per-window accuracy, including when episodes
    have unequal numbers of windows. No resampling of training/inference seeds.
    """
    import numpy as np

    if type(resamples) is not int or resamples < 1:
        raise ValueError('resamples must be a positive integer')
    reference = all_rows['lora']
    if not reference or not reference[0]:
        raise ValueError('empty evaluation')
    keys = [(r['meta']['episode_id'], r['meta']['window_end']) for r in reference[0]]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate evaluation window')
    episodes = sorted({key[0] for key in keys})
    if len(episodes) < 2:
        raise ValueError('at least two episodes are required')
    lookup = {episode: index for index, episode in enumerate(episodes)}
    positions = np.array([lookup[key[0]] for key in keys])
    weights = np.bincount(positions, minlength=len(episodes)) * len(reference)
    draws = np.random.default_rng(seed).integers(len(episodes), size=(resamples, len(episodes)))
    denominators = weights[draws].sum(axis=1)
    estimates, samples = {}, {}
    def interval(point, values):
        return {'estimate': float(point), 'ci95': np.quantile(values, [.025, .975]).tolist()}
    for method, repetitions in all_rows.items():
        if len(repetitions) != len(reference) or any(
                [(r['meta']['episode_id'], r['meta']['window_end']) for r in repeat] != keys
                for repeat in repetitions):
            raise ValueError(f'unpaired evaluation windows: {method}')
        samples[method], estimates[method] = {}, {}
        for metric, field in (('set_exact', 'set_exact'), ('joint_accuracy', 'correct')):
            counts = np.zeros(len(episodes))
            for repeat in repetitions:
                np.add.at(counts, positions, [r['score'][field] for r in repeat])
            samples[method][metric] = counts[draws].sum(axis=1) / denominators
            estimates[method][metric] = interval(counts.sum() / weights.sum(), samples[method][metric])
    paired = {}
    for method in all_rows:
        if method == 'lora':
            continue
        paired[method] = {metric: interval(
            estimates[method][metric]['estimate'] - estimates['lora'][metric]['estimate'],
            samples[method][metric] - samples['lora'][metric])
            for metric in samples[method]}
    return {'unit': 'episode_id', 'episodes': len(episodes), 'windows': len(keys),
            'runs': len(reference), 'resamples': resamples, 'seed': seed,
            'method': 'paired episode percentile bootstrap; pooled window accuracy',
            'methods': estimates, 'paired_vs_lora': paired}


def error_breakdown(repetitions):
    counts = dict.fromkeys(('correct', 'invalid', 'set_only', 'uncertainty_only', 'both'), 0)
    missed, extra = Counter(), Counter()
    uncertainty = dict.fromkeys(('tp', 'fp', 'fn', 'tn', 'invalid'), 0)
    failures = []
    for repeat, result in enumerate(repetitions, 1):
        for row in result:
            score, gold = row['score'], row['gold']
            if not score['json_ok']:
                category = 'invalid'
                uncertainty['invalid'] += 1
            else:
                exact, uncertain = score['set_exact'], score['uncertainty_correct']
                category = ('correct' if uncertain else 'uncertainty_only') if exact else (
                    'set_only' if uncertain else 'both')
                expected, predicted = set(gold['root_causes']), set(score['predicted_causes'])
                missed.update(expected - predicted)
                extra.update(predicted - expected)
                positive = score['needs_more_data']
                uncertainty[('tp' if positive else 'fn') if gold['needs_more_data'] else (
                    'fp' if positive else 'tn')] += 1
            counts[category] += 1
            if category != 'correct':
                failures.append({'repeat': repeat, 'index': row['index'],
                                 'episode_id': row['meta']['episode_id'],
                                 'window_end': row['meta']['window_end'], 'category': category,
                                 'gold': gold, 'predicted_causes': score['predicted_causes'],
                                 'needs_more_data': score['needs_more_data']})
    return {'predictions': sum(counts.values()), 'categories': counts,
            'missed_causes': dict(missed.most_common()), 'extra_causes': dict(extra.most_common()),
            'uncertainty': uncertainty, 'failures': failures}


def analyze(run, *, resamples=10000, seed=42):
    pipeline = read(run / 'pipeline.json')
    if pipeline['status'] != 'completed':
        raise ValueError('wait for the full pipeline to complete before analyzing')
    for name, expected in pipeline['source_sha256'].items():
        if digest(ROOT / name) != expected or digest(run / 'source' / name) != expected:
            raise ValueError(f'source mismatch: {name}')
    for split, expected in pipeline['dataset_sha256'].items():
        if digest(ROOT / f'data/sft_hard_v1/{split}.jsonl') != expected:
            raise ValueError(f'dataset mismatch: {split}')
    for split, expected in pipeline['preference_sha256'].items():
        if digest(run / 'rl_data' / f'prefs_{split}.jsonl') != expected:
            raise ValueError(f'preference mismatch: {split}')
    dataset_path = ROOT / 'data/sft_hard_v1/eval.jsonl'
    dataset = rows(dataset_path)
    dataset_hash = digest(dataset_path)
    if dataset_hash != pipeline['dataset_sha256']['eval']:
        raise ValueError('frozen evaluation dataset changed')
    scorer = HardFaultScorer(dataset_path.parent / 'ontology.json')
    comparison = read(run / 'comparison.json')
    all_rows, analysis, inference, input_hashes = {}, {}, None, {}
    input_hashes['pipeline.json'] = digest(run / 'pipeline.json')
    input_hashes['comparison.json'] = digest(run / 'comparison.json')
    for method in METHODS:
        directory = run / method / 'eval'
        manifest = read(directory / 'evaluation_manifest.json')
        if (manifest['status'] != 'completed' or not manifest['temporary_service_stopped']
                or manifest['runs'] != 3 or manifest['dataset_sha256'] != dataset_hash):
            raise ValueError(f'incomplete evaluation: {method}')
        if inference is not None and inference != manifest['inference']:
            raise ValueError(f'inference settings differ: {method}')
        inference = manifest['inference']
        input_hashes[f'{method}/eval/evaluation_manifest.json'] = digest(directory / 'evaluation_manifest.json')
        repetitions = []
        summaries = comparison[method]['summaries']
        if len(summaries) != 3 or comparison[method]['runs'] != 3 or comparison[method]['n_per_run'] != len(dataset):
            raise ValueError(f'invalid comparison dimensions: {method}')
        for repeat in (1, 2, 3):
            result = rows(directory / f'run{repeat}.jsonl')
            saved = read(directory / f'run{repeat}.summary.json')
            require_equal(saved, summaries[repeat - 1], f'{method}/{repeat}/comparison')
            if (saved['dataset_sha256'] != dataset_hash or saved['seed'] != repeat
                    or saved['metric_script_sha256'] != pipeline['source_sha256']['tools/hard_fault_metrics.py']):
                raise ValueError(f'summary provenance mismatch: {method}/{repeat}')
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
            audit_summary(result, scorer.names, saved, f'{method}/{repeat}')
            for suffix in ('.jsonl', '.summary.json'):
                name = f'{method}/eval/run{repeat}{suffix}'
                input_hashes[name] = digest(run / name)
            repetitions.append(result)
        all_rows[method] = repetitions
        expected_means = {key: statistics.fmean(s[key] for s in summaries if s[key] is not None)
                          if any(s[key] is not None for s in summaries) else None
                          for key in METRICS}
        require_equal(comparison[method]['mean'], expected_means, f'{method}/mean')
        group_means = {
            group: {'n': summaries[0]['groups'][group]['n'], **{
                key: statistics.fmean(s['groups'][group][key] for s in summaries)
                for key in ('set_exact', 'joint_accuracy')}}
            for group in summaries[0]['groups']}
        model = ROOT / pipeline['plan']['models'][method]
        require_equal(read(model / 'run_manifest.json'), comparison[method]['training_manifest'],
                      f'{method}/training_manifest')
        require_equal(read(run / method / 'train/resource_summary.json'), comparison[method]['train_resources'],
                      f'{method}/train_resources')
        if (comparison[method]['training_manifest']['status'] != 'completed'
                or comparison[method]['train_resources']['status'] != 'completed'):
            raise ValueError(f'incomplete training: {method}')
        metrics = read(model / f'{method}_train_metrics.json')
        progress = rows(model / 'progress.jsonl')
        analysis[method] = {
            'mean': expected_means,
            'groups': group_means,
            'errors': error_breakdown(repetitions),
            'per_class': {name: {'support_per_run': summaries[0]['per_class'][name]['tp'] + summaries[0]['per_class'][name]['fn'],
                                'mean_f1': statistics.fmean(s['per_class'][name]['f1'] for s in summaries)}
                          for name in sorted(scorer.names)},
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
            'analysis_script_sha256': digest(__file__), 'input_sha256': input_hashes,
            'inference': inference, 'methods': analysis,
            'cluster_bootstrap': cluster_intervals(all_rows, resamples=resamples, seed=seed)}


def markdown(result, comparison):
    lines = ['# 八类故障后训练：分层结果与训练成本', '',
             f"每个模型在同一冻结的 {result['n_per_run']} 条输入上评估 {result['runs']} 次；以下准确率为各次推理均值。",
             '逐行输入、金标、元数据、原始输出重新评分及全部汇总指标均通过核验；历史清单记录临时评估服务已停止。', '',
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
              '缺失通道的集合匹配率只衡量可见根因；是否正确请求补充观测需同时查看联合准确率。', '',
              '## 按炉次配对的 95% 区间', '',
              '| 方法 | 集合匹配率及区间 | 联合准确率及区间 |',
              '|---|---:|---:|']
    bootstrap = result['cluster_bootstrap']
    for method in METHODS:
        values = bootstrap['methods'][method]
        cells = [f"{values[k]['estimate']:.2%} [{values[k]['ci95'][0]:.2%}, {values[k]['ci95'][1]:.2%}]"
                 for k in ('set_exact', 'joint_accuracy')]
        lines.append(f"| {method} | " + ' | '.join(cells) + ' |')
    lines += ['', '| 相对 LoRA | 集合匹配变化（百分点） | 联合准确率变化（百分点） |',
              '|---|---:|---:|']
    for method in ('full', 'dpo', 'grpo'):
        values = bootstrap['paired_vs_lora'][method]
        cells = [f"{values[k]['estimate'] * 100:+.2f} [{values[k]['ci95'][0] * 100:+.2f}, {values[k]['ci95'][1] * 100:+.2f}]"
                 for k in ('set_exact', 'joint_accuracy')]
        lines.append(f"| {method} | " + ' | '.join(cells) + ' |')
    lines += ['', f"以 {bootstrap['episodes']} 个炉次为簇，有放回抽样 {bootstrap['resamples']:,} 次（seed={bootstrap['seed']}），取 2.5%/97.5% 分位数。",
              '每次抽样保留同一炉次的所有窗口和三次推理，各方法使用同一抽样；点估计仍按窗口计权。',
              '这些是固定训练权重下的探索性区间，未做多重比较校正，不包含训练种子或真实设备分布的不确定性。', '',
              '## 错误分解', '',
              '| 方法 | 全部预测 | 仅集合错误 | 仅不确定性错误 | 两者都错 | 格式错误 | 补测漏报 FN | 补测误报 FP |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        errors = result['methods'][method]['errors']
        c, u = errors['categories'], errors['uncertainty']
        lines.append(f"| {method} | {errors['predictions']} | {c['set_only']} | {c['uncertainty_only']} | {c['both']} | {c['invalid']} | {u['fn']} | {u['fp']} |")
    lines += ['', '前四类错误互斥；补测 FN/FP 是交叉统计，不能重复相加。计数包含重复推理，不代表独立样本数。',
              '逐项漏诊/多报根因以及失败窗口、轮次和索引保存在 analysis.json 的 errors 字段。', '',
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
    parser.add_argument('--out-dir', type=Path, help='write analysis separately from the original run')
    parser.add_argument('--bootstrap-resamples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=42)
    args = parser.parse_args()
    result = analyze(args.run_dir, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed)
    output = args.out_dir or args.run_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    report = markdown(result, read(args.run_dir / 'comparison.json'))
    (output / 'analysis.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
