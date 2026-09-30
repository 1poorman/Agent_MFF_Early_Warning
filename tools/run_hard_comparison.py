#!/usr/bin/env python3
"""Complete the four-method comparison after measured SFT jobs have started.

Independent GPU branches: SFT completion -> export -> frozen evaluation, then
DPO/GRPO training -> export -> frozen evaluation. Fails closed on any failed
training run, source change, incomplete result, or endpoint error.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PY_AGENT = str(Path(sys.executable).resolve())
PY_TRAIN = '/home/huachenghao/.conda/envs/mff_sft/bin/python'
PY_RL = '/home/huachenghao/.conda/envs/mff_rl/bin/python'
BASE = '/home/huachenghao/models/MiniCPM5-2B'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, default=Path('logs/hard_minicpm5_20260929'))
    ap.add_argument('--tag', default='20260929')
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--timeout-hours', type=float, default=12)
    args = ap.parse_args()
    os.chdir(ROOT)
    run = args.run_dir
    models = {m: Path('saves') / f'mff-{m}-hard-minicpm5-2b-{args.tag}' for m in ('full', 'lora', 'dpo', 'grpo')}
    merged = {m: Path(str(path) + '-merged') for m, path in models.items() if m != 'full'}
    source_names = ['tools/run_hard_comparison.py', 'tools/train_rl.py', 'tools/hard_fault_metrics.py',
                    'tools/train_lora.py', 'tools/measure_run.py', 'data/sft_hard_v1/ontology.json',
                    'tools/mff_full_hard_minicpm5.yaml', 'tools/mff_lora_hard_minicpm5.yaml',
                    'tools/run_adapter_eval.py', 'tools/eval_hard_model.py', 'tools/merge_lora.py',
                    'tools/build_hard_preferences.py', 'tools/rl_reward.py',
                    'tools/mff_dpo_hard_minicpm5.yaml', 'tools/mff_grpo_hard_minicpm5.yaml']
    gpu = {'full': '0', 'lora': '1', 'dpo': '1', 'grpo': '0'}
    def eval_command(method):
        return [PY_AGENT, 'tools/run_adapter_eval.py', '--model', str(models[method] if method == 'full' else merged[method]),
                '--name', f'mff-hard-{method}', '--gpu', gpu[method], '--port', '3765' if gpu[method] == '0' else '3766',
                '--schema', 'hard_faults_v1', '--dataset', 'data/sft_hard_v1/eval.jsonl',
                '--max-tokens', '640', '--runs', '3', '--out-dir', str(run / method / 'eval')]
    def train_command(method):
        return [PY_AGENT, 'tools/measure_run.py', '--gpu', gpu[method], '--out-dir', str(run / method / 'train'),
                '--', PY_RL, 'tools/train_rl.py', '--algo', method, '--config', f'tools/mff_{method}_hard_minicpm5.yaml']
    plan = {'models': {k: str(v) for k, v in models.items()},
            'evaluations': {m: eval_command(m) for m in models},
            'posttraining': {m: train_command(m) for m in ('dpo', 'grpo')}}
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    state_path = run / 'pipeline.json'
    if state_path.exists():
        raise FileExistsError(f'refusing to overwrite {state_path}')
    source_hashes = {name: sha(name) for name in source_names}
    dataset_hashes = {split: sha(f'data/sft_hard_v1/{split}.jsonl') for split in ('train', 'val', 'eval')}
    preference_hashes = {split: sha(run / 'rl_data' / f'prefs_{split}.jsonl') for split in ('train', 'val')}
    for name in source_names:
        target = run / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    state = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'stages': {},
             'plan': plan, 'source_sha256': source_hashes, 'dataset_sha256': dataset_hashes,
             'preference_sha256': preference_hashes, 'controller_pid': os.getpid()}
    lock = threading.Lock()
    def save():
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(state_path)
    def stage(label, status, **details):
        with lock:
            state['stages'][label] = {'status': status, 'time_utc': datetime.now(timezone.utc).isoformat(), **details}
            save()
        print(f'{label}: {status}', flush=True)
    def check_inputs():
        for name, digest in source_hashes.items():
            if sha(name) != digest:
                raise RuntimeError(f'source changed during experiment: {name}')
        for split, digest in dataset_hashes.items():
            if sha(f'data/sft_hard_v1/{split}.jsonl') != digest:
                raise RuntimeError(f'dataset changed: {split}')
        for split, digest in preference_hashes.items():
            if sha(run / 'rl_data' / f'prefs_{split}.jsonl') != digest:
                raise RuntimeError(f'preferences changed: {split}')
    def command(label, argv, cpu=False):
        check_inputs()
        path = run / (label.replace('/', '_') + '.log')
        env = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
        if cpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        started = time.perf_counter()
        with path.open('x') as log:
            child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            stage(label, 'running', command=argv, pid=child.pid, log=str(path))
            code = child.wait()
        stage(label, 'completed' if code == 0 else 'failed', command=argv, exit_code=code,
              wall_seconds=time.perf_counter() - started, log=str(path))
        if code:
            raise RuntimeError(f'{label} exited {code}; see {path}')
    def wait_sft(method):
        stage(f'{method}/wait_training', 'running')
        deadline = time.monotonic() + args.timeout_hours * 3600
        while True:
            measured = read(run / method / 'train/resource_summary.json')
            if measured['status'] == 'failed':
                raise RuntimeError(f'{method} training failed')
            if measured['status'] == 'completed':
                manifest = read(models[method] / 'run_manifest.json')
                if manifest.get('status') != 'completed' or manifest['train']['sha256'] != dataset_hashes['train'] or manifest['val']['sha256'] != dataset_hashes['val']:
                    raise RuntimeError(f'{method} training manifest mismatch')
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f'{method} training did not finish')
            time.sleep(10)
        stage(f'{method}/wait_training', 'completed')
    def export_and_evaluate(method):
        if method != 'full':
            base = BASE if method == 'lora' else str(merged['lora'])
            command(f'{method}/merge', [PY_RL, 'tools/merge_lora.py', '--base-model', base,
                    '--adapter', str(models[method]), '--output', str(merged[method])], cpu=True)
        command(f'{method}/eval', eval_command(method))
    def baseline(method):
        wait_sft(method)
        export_and_evaluate(method)
    def posttrain(method):
        command(f'{method}/train', train_command(method))
        manifest = read(models[method] / 'run_manifest.json')
        if manifest['status'] != 'completed' or manifest['policy_start_and_reference'] != str(merged['lora']):
            raise RuntimeError(f'{method} reference/status mismatch')
        export_and_evaluate(method)
    save()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(baseline, ('full', 'lora')))
        # Hyperparameters remain frozen; baseline test scores do not change them.
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(posttrain, ('dpo', 'grpo')))
        check_inputs()
        comparison = {}
        keys = ['set_exact', 'joint_accuracy', 'macro_f1', 'json_valid', 'evidence_traceable',
                'sop_recall', 'uncertainty_accuracy', 'uncertainty_precision', 'uncertainty_recall',
                'normal_false_positive', 'brier_joint']
        expected_indices = list(range(268))
        reference_inputs = None
        for method in models:
            summaries = []
            for repeat in (1, 2, 3):
                prefix = run / method / 'eval' / f'run{repeat}'
                summary = read(prefix.with_suffix('.summary.json'))
                rows = [json.loads(line) for line in prefix.with_suffix('.jsonl').read_text().splitlines()]
                if summary['n'] != 268 or summary['errors'] or summary['dataset_sha256'] != dataset_hashes['eval']:
                    raise RuntimeError(f'incomplete evaluation: {method}/{repeat}')
                inputs = [r['input_sha256'] for r in rows]
                if [r['index'] for r in rows] != expected_indices or (reference_inputs is not None and inputs != reference_inputs):
                    raise RuntimeError('evaluation input order mismatch')
                reference_inputs = inputs
                summaries.append(summary)
            comparison[method] = {'runs': 3, 'n_per_run': 268,
                'mean': {key: statistics.fmean(s[key] for s in summaries if s[key] is not None)
                         if any(s[key] is not None for s in summaries) else None for key in keys},
                'train_resources': read(run / method / 'train/resource_summary.json'),
                'training_manifest': read(models[method] / 'run_manifest.json'),
                'summaries': summaries}
        (run / 'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + '\n')
        lines = ['# 八类故障：全参 SFT、LoRA、DPO、GRPO 对照', '',
                 '数据版本 `hard_faults_v1`：训练 912、验证 200、冻结评估 268 条；各模型完成三次推理。', '',
                 '| 方法 | 根因集合完全匹配 | 集合＋不确定性正确 | Macro-F1 | 证据可追溯 | Brier ↓ |',
                 '|---|---:|---:|---:|---:|---:|']
        for method, values in comparison.items():
            m = values['mean']
            brier = f"{m['brier_joint']:.4f}" if m['brier_joint'] is not None else '—'
            lines.append(f"| {method} | {m['set_exact']:.2%} | {m['joint_accuracy']:.2%} | {m['macro_f1']:.4f} | {m['evidence_traceable']:.2%} | {brier} |")
        lines += ['', '全参与 LoRA 均从原始 MiniCPM5-2B 训练 3 epochs / 171 步；有效 batch=16，学习率分别为 1e-5 / 1e-4。',
                  'DPO 与 GRPO 分别从新的 LoRA-SFT 合并权重启动，保持相同 SFT 参考策略；DPO 114 步，GRPO 128 步、每组四次采样。',
                  'DPO 使用 912 对合成偏好，200 对独立验证；GRPO 在线采样，规则奖励为集合完全匹配 0.4、样本 F1 0.2、不确定性正确 0.2、证据 0.1、SOP 0.05、格式 0.05。',
                  '置信度不参与奖励；训练标签中的置信度为合成教学值，不是经验校准概率。Brier 针对集合和不确定性同时判对，分母仅为格式可解析输出。',
                  '这些实验是合成规则反馈，不是人工 RLHF 或 AI 裁判 RLAIF；三次推理不等于三个训练种子，也不是等算力算法消融。',
                  '', f'完整分层指标、各轮结果、数据与源码指纹、耗时及显存见 `{run}/comparison.json` 和 `pipeline.json`。',
                  '权重和合并模型均位于 `/models/hch/Agent_MFF_Early_Warning/saves/`。', '']
        report = '\n'.join(lines)
        (run / 'comparison.md').write_text(report)
        Path('design/HARD_POSTTRAINING_COMPARISON.md').write_text(report)
        with lock:
            state['status'] = 'completed'
            state['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save()
        print(report, flush=True)
    except BaseException as exc:
        with lock:
            state['status'] = 'failed'
            state['error'] = f'{type(exc).__name__}: {exc}'
            save()
        raise


if __name__ == '__main__':
    main()
