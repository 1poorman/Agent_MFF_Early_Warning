#!/usr/bin/env python3
"""RL 后训练单入口：--algo {dpo, grpo, rloo, reward}。

留痕口径对标 tools/train_lora.py（resolved_config.json / run_manifest.json /
progress.jsonl / cuda_peaks / 拒绝覆盖已有产物），保证 A3/A4/A5 与 A2 的训练
成本数字逐格可比。

三条 RL 线的共同起点：**A2 合并权重**，不是原始底座
---------------------------------------------------
`model_name_or_path` 指向 `saves/mff-lora-minicpm5-2b-20260923-merged`，再由
`peft_config` 在其上新挂一个 adapter。于是：

    策略  = A2 + 新 adapter
    参考  = 关掉 adapter = A2 本身

参考模型正是 SFT 模型，KL 约束把策略拴在 A2 附近 —— 这才是标准 RLHF 口径。
若改成「底座 + 加载 A2 adapter 续训」，参考会退化成**原始底座**，KL 就把模型
往没学过任务的方向拉。同时这套写法不需要第二份权重，是三条线能塞进 24 G 的原因。

两处已实测的模板陷阱（tools/check_rl_template.py 的产出）
--------------------------------------------------------
1. `add_special_tokens=True` 会改变全部 1200 条的 token（双 BOS）。TRL 在
   dpo_trainer.py:674 硬编码 `add_special_tokens=False`，与本项目一致，安全。
2. **eos_token 是 `</s>`，而模板的 assistant 收尾是 `<|im_end|>\\n`**。
   DPO 的 tokenize_row（dpo_trainer.py:734）无条件补 `eos_token_id`，若直接喂
   模型原始输出，训练学到的收尾 token 就与线上不同。故 DPO/RM 的 chosen 与
   rejected 一律**自带模板收尾串**，且该串在运行时从 tokenizer 现算（见
   `template_suffix()`），不写死，避免将来换模板后静默漂移。
   generation_config.json 的 eos_token_id = [1, 130073]，两个 token 都会停，
   因此 TRL 额外补的 `</s>` 落在 `<|im_end|>` 之后，线上永远走不到。

用法：
    TRL_EXPERIMENTAL_SILENCE=1 CUDA_VISIBLE_DEVICES=1 \\
    /home/huachenghao/.conda/envs/mff_rl/bin/python tools/train_rl.py \\
        --algo dpo --config tools/mff_dpo_minicpm5.yaml
    # 先跑 --dry-run：只建数据与模型，打印首条样本的解码结果，不动优化器
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")   # RLOO 迁移警告，须早于 import trl

import numpy as np
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.rl_reward import RewardScorer, make_reward_func  # noqa: E402

ALGOS = ("dpo", "grpo", "rloo", "reward")


# ---------------------------------------------------------------- 基础设施

def load_config(path: str) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def cuda_peaks() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    return {"device": torch.cuda.get_device_name(),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}


def load_jsonl(path: str, limit: int = 0) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ProgressLog(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.path = output_dir / "progress.jsonl"

    def on_log(self, args, state, control, logs=None, **kwargs):
        row = {"time_utc": datetime.now(timezone.utc).isoformat(),
               "step": state.global_step, "epoch": state.epoch, **(logs or {})}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------- 模板

def render_prompt(tokenizer, messages: list[dict]) -> str:
    """与线上逐 token 相同的 prompt（G0 第 2 项在 1200 条上断言过）。"""
    msgs = [m for m in messages if m["role"] != "assistant"]
    return tokenizer.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)


def template_suffix(tokenizer) -> str:
    """现算 assistant 轮的收尾串（MiniCPM5 上是 '<|im_end|>\\n'）。

    用一条哑样本渲染「有/无 assistant」两次做差得到，不写死字面量：模板一旦改版，
    这里跟着变，而写死的字面量只会静默错位。
    """
    probe = [{"role": "user", "content": "__probe__"}]
    ans = "__answer__"
    head = tokenizer.apply_chat_template(probe, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
    full = tokenizer.apply_chat_template(probe + [{"role": "assistant", "content": ans}],
                                         tokenize=False, add_generation_prompt=False,
                                         enable_thinking=False)
    if not full.startswith(head) or ans not in full[len(head):]:
        raise RuntimeError("无法从模板切出 assistant 收尾串；模板结构与预期不符")
    tail = full[len(head):]
    return tail[tail.index(ans) + len(ans):]


# ---------------------------------------------------------------- 数据

def build_preference_dataset(cfg: dict, tokenizer, suffix: str, limit: int = 0):
    """DPO / RewardTrainer 共用：prompt 预渲染，completion 自带模板收尾串。"""
    rows = load_jsonl(cfg["train_file"], limit)
    if not rows:
        raise ValueError(f"preference dataset is empty: {cfg['train_file']}")

    def to_example(r: dict) -> dict:
        prompt = render_prompt(tokenizer, r["messages"])
        return {"prompt": prompt,
                "chosen": r["chosen"].rstrip() + suffix,
                "rejected": r["rejected"].rstrip() + suffix}

    train = Dataset.from_list([to_example(r) for r in rows])
    val = None
    val_rows: list[dict] = []
    if cfg.get("eval_file") and Path(cfg["eval_file"]).exists():
        val_rows = load_jsonl(cfg["eval_file"], limit)
        if val_rows:
            val = Dataset.from_list([to_example(r) for r in val_rows])
    return train, val, rows, val_rows


def build_prompt_dataset(cfg: dict, tokenizer, limit: int = 0):
    """GRPO / RLOO：只要 prompt；gold 与 user_text 作为附加列透传给奖励函数。

    `user_text` 不可省 —— 见 tools/rl_reward.py:make_reward_func 的说明：
    拿渲染后的 prompt 打分会把 system 里的阈值数字也算成"输入出现过的数"。
    """
    rows = load_jsonl(cfg["prompt_file"], limit)
    if not rows:
        raise ValueError(f"prompt pool is empty: {cfg['prompt_file']}")
    examples = []
    for r in rows:
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        gold = r["meta"]["gold"]
        if cfg.get("reward_schema") == "hard_faults_v1":
            gold = json.dumps({"root_causes": r["meta"]["gold_causes"],
                               "needs_more_data": r["meta"]["needs_more_data"]}, ensure_ascii=False)
        examples.append({"prompt": render_prompt(tokenizer, r["messages"]),
                         "user_text": user,
                         "gold": gold,
                         "kind": r["meta"].get("kind", "")})
    return Dataset.from_list(examples), rows


def data_manifest(path: str, rows: list, extra: dict | None = None) -> dict:
    out = {"path": str(Path(path).resolve()), "sha256": sha256(path),
           "samples_used": len(rows),
           "kind_counts": dict(Counter(
               (r.get("meta") or {}).get("kind", r.get("kind")) for r in rows))}
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------- 奖励劫持监控

class RewardMonitor:
    """按方案 §3 的预案盯住奖励劫持：置信度整体平移套 Brier。

    判据：mean_correct 与 mean_wrong **同步上升**而 gap 不改善，或 conf 标准差
    塌缩到近 0（所有输出报同一个置信度）。这两项都写进 reward_monitor.jsonl，
    失败时作为负面证据留档。
    """

    def __init__(self, output_dir: Path, flush_every: int = 256):
        self.path = output_dir / "reward_monitor.jsonl"
        self.flush_every = flush_every
        self.buf: list[dict] = []
        self.n_seen = 0

    def on_score(self, score, gold, index):
        self.buf.append({"conf": score.confidence, "correct": bool(score.correct),
                         "total": score.total, "json_ok": bool(score.json_ok),
                         "hallu_ok": bool(score.hallu_ok)})
        self.n_seen += 1
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        confs = [b["conf"] for b in self.buf if b["conf"] is not None]
        cor = [b["conf"] for b in self.buf if b["correct"] and b["conf"] is not None]
        wro = [b["conf"] for b in self.buf if not b["correct"] and b["conf"] is not None]
        row = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "n_cumulative": self.n_seen, "n_window": len(self.buf),
            "reward_mean": round(statistics.fmean([b["total"] for b in self.buf]), 6),
            "accuracy": round(sum(b["correct"] for b in self.buf) / len(self.buf), 4),
            "json_ok_rate": round(sum(b["json_ok"] for b in self.buf) / len(self.buf), 4),
            "hallu_ok_rate": round(sum(b["hallu_ok"] for b in self.buf) / len(self.buf), 4),
            "conf_mean": round(statistics.fmean(confs), 4) if confs else None,
            # 标准差塌缩 = 所有输出报同一个置信度 = 典型的 Brier 劫持
            "conf_std": round(statistics.pstdev(confs), 4) if len(confs) > 1 else 0.0,
            "conf_mean_correct": round(statistics.fmean(cor), 4) if cor else None,
            "conf_mean_wrong": round(statistics.fmean(wro), 4) if wro else None,
            "conf_gap": (round(statistics.fmean(cor) - statistics.fmean(wro), 4)
                         if cor and wro else None),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.buf.clear()


# ---------------------------------------------------------------- 主流程

def main() -> int:
    started = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", required=True, choices=ALGOS)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", help="覆盖 YAML 里的 output_dir，不改实验配置文件")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-steps", type=int, help="Override training steps, e.g. 1 for a smoke run")
    ap.add_argument("--dry-run", action="store_true",
                    help="只建数据与模型并打印首条样本解码结果，不进优化器")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.max_steps is not None:
        if args.max_steps < 1:
            ap.error("--max-steps must be positive")
        cfg["max_steps"] = args.max_steps
    output_dir = Path(cfg["output_dir"])
    if (not args.dry_run and output_dir.exists() and any(output_dir.iterdir())
            and not cfg.get("resume_from_checkpoint")):
        raise FileExistsError(f"拒绝覆盖已有产物：{output_dir}；换 --output-dir")
    seed_everything(int(cfg.get("seed", 42)))
    torch.set_num_threads(int(cfg.get("cpu_threads", 8)))

    model_path = str(cfg["model_name_or_path"])
    if not Path(model_path).exists():
        raise FileNotFoundError(f"起点权重不存在：{model_path}（应为 A2 合并权重）")
    has_cuda = torch.cuda.is_available()
    print(f"[env] torch={torch.__version__} cuda={has_cuda} "
          f"devices={torch.cuda.device_count()} algo={args.algo}", flush=True)
    print(f"[model] 起点（同时也是参考策略）= {model_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=bool(cfg.get("trust_remote_code", True)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    suffix = template_suffix(tokenizer)
    if args.algo in ("grpo", "rloo"):
        # TRL uses this single EOS both for generation and for loss masking.
        # MiniCPM's chat EOS differs from its tokenizer's default </s>.
        stop_ids = tokenizer(suffix.strip(), add_special_tokens=False)["input_ids"]
        if len(stop_ids) != 1:
            raise ValueError(f"online RL requires one template stop token: {suffix!r}")
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(stop_ids[0])
    print(f"[template] assistant 收尾串={suffix!r} eos_token={tokenizer.eos_token!r}",
          flush=True)

    # ---- 数据 ----
    monitor = RewardMonitor(output_dir)
    reward_funcs = None
    eval_ds = None
    if args.algo in ("dpo", "reward"):
        train_ds, eval_ds, rows, val_rows = build_preference_dataset(
            cfg, tokenizer, suffix, args.max_samples)
        dm = {"train": data_manifest(cfg["train_file"], rows)}
        if val_rows:
            dm["eval"] = data_manifest(cfg["eval_file"], val_rows)
        dm["train"]["source_counts"] = dict(Counter(r.get("source") for r in rows))
    else:
        train_ds, rows = build_prompt_dataset(cfg, tokenizer, args.max_samples)
        dm = {"train": data_manifest(cfg["prompt_file"], rows)}
        if args.algo == "grpo":
            if cfg.get("reward_schema") == "hard_faults_v1":
                from tools.hard_fault_metrics import HardFaultScorer
                scorer = HardFaultScorer(cfg.get("ontology_file"))
            else:
                scorer = RewardScorer()
            reward_funcs = [make_reward_func({}, scorer, on_score=monitor.on_score)]

    print(f"[data] train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}", flush=True)

    # ---- prompt 长度体检：max_prompt_length 偏小会**左截断**掉 prompt 开头 ----
    prompt_tokens = [len(tokenizer(p, add_special_tokens=False)["input_ids"])
                     for p in train_ds["prompt"]]
    p_max, p_mean = max(prompt_tokens), statistics.fmean(prompt_tokens)
    limit_key = int(cfg.get("max_prompt_length", 1280))
    print(f"[len] prompt tokens max={p_max} mean={p_mean:.0f} "
          f"max_prompt_length={limit_key}", flush=True)
    if p_max > limit_key:
        raise ValueError(
            f"max_prompt_length={limit_key} < 实测最长 prompt {p_max} token。"
            f"TRL 对 prompt 做的是**左截断**（dpo_trainer.py:739），会切掉开头的"
            f"系统提示与工况说明，训练输入与线上不再是同一件事。调大后重跑。")

    if args.algo == "dpo":
        cap = int(cfg.get("max_completion_length", 640))
        longest = max(len(tokenizer(t, add_special_tokens=False)["input_ids"]) + 1
                      for ds in (train_ds, eval_ds) if ds is not None
                      for col in ("chosen", "rejected") for t in ds[col])
        if longest > cap:
            raise ValueError(f"completion requires {longest} tokens including EOS; cap is {cap}")

    if args.dry_run:
        ex = train_ds[0]
        print("\n=== dry-run 首条样本 ===")
        print(f"[prompt 尾部 240 字]\n...{ex['prompt'][-240:]}")
        if "chosen" in ex:
            print(f"\n[chosen 尾部 160 字]\n...{ex['chosen'][-160:]!r}")
            print(f"[rejected 尾部 160 字]\n...{ex['rejected'][-160:]!r}")
        else:
            print(f"\n[gold]={ex['gold']}  [kind]={ex['kind']}")
            print(f"[user_text 头部 120 字]\n{ex['user_text'][:120]}")
        print(f"\n[dry-run] 数据构建成功，未加载模型。train={len(train_ds)}")
        return 0

    # ---- 模型 ----
    dtype = torch.bfloat16 if bool(cfg.get("bf16", True)) else torch.float32
    peft_cfg = LoraConfig(
        task_type="SEQ_CLS" if args.algo == "reward" else "CAUSAL_LM",
        r=int(cfg.get("lora_rank", 32)),
        lora_alpha=int(cfg.get("lora_alpha", 64)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=cfg.get("target_modules", "all-linear"),
        bias="none",
    )

    from trl import (DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, RewardConfig,
                     RewardTrainer, RLOOConfig, RLOOTrainer)

    common = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 8)),
        "num_train_epochs": float(cfg.get("num_train_epochs", 1.0)),
        "learning_rate": float(cfg.get("learning_rate", 5e-6)),
        "lr_scheduler_type": cfg.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": float(cfg.get("warmup_ratio", 0.1)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 1.0)),
        "logging_steps": int(cfg.get("logging_steps", 5)),
        "logging_first_step": True,
        "save_strategy": cfg.get("save_strategy", "steps"),
        "save_steps": int(cfg.get("save_steps", 200)),
        "save_total_limit": int(cfg.get("save_total_limit", 2)),
        "report_to": [],
        "seed": int(cfg.get("seed", 42)),
        "bf16": bool(cfg.get("bf16", True)) and has_cuda,
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", True)),
        "disable_tqdm": bool(cfg.get("disable_tqdm", True)),
        "logging_dir": str(output_dir / "logs"),
        "dataloader_num_workers": 0,
    }
    if cfg.get("max_steps"):
        common["max_steps"] = int(cfg["max_steps"])

    model = None
    if args.algo == "dpo":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, low_cpu_mem_usage=True,
            trust_remote_code=bool(cfg.get("trust_remote_code", True)))
        model.config.use_cache = False
        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}  # TRL 0.24 expects this pre-transformers-5 bookkeeping.
        targs = DPOConfig(
            beta=float(cfg.get("beta", 0.1)),
            max_prompt_length=limit_key,
            max_completion_length=int(cfg.get("max_completion_length", 640)),
            max_length=limit_key + int(cfg.get("max_completion_length", 640)),
            **common)
        trainer = DPOTrainer(model=model, args=targs, train_dataset=train_ds,
                             eval_dataset=eval_ds, processing_class=tokenizer,
                             peft_config=peft_cfg,
                             callbacks=[ProgressLog(output_dir)])
    elif args.algo == "reward":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=1, dtype=dtype, low_cpu_mem_usage=True,
            trust_remote_code=bool(cfg.get("trust_remote_code", True)))
        model.config.use_cache = False
        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        targs = RewardConfig(
            max_length=int(cfg.get("max_length", 1920)),
            # RewardTrainer 自己补 eos（reward_trainer.py:465），告诉它模板的真收尾 token
            eos_token=cfg.get("eos_token", suffix.strip() or None),
            **common)
        trainer = RewardTrainer(model=model, args=targs, train_dataset=train_ds,
                                eval_dataset=eval_ds, processing_class=tokenizer,
                                peft_config=peft_cfg,
                                callbacks=[ProgressLog(output_dir)])
    else:   # grpo / rloo
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, low_cpu_mem_usage=True,
            trust_remote_code=bool(cfg.get("trust_remote_code", True)))
        model.config.use_cache = True
        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}
        gen = {
            "num_generations": int(cfg.get("num_generations", 8)),
            "max_prompt_length": limit_key,
            "max_completion_length": int(cfg.get("max_completion_length", 320)),
            "temperature": float(cfg.get("temperature", 1.0)),
            "top_p": float(cfg.get("top_p", 0.95)),
            "beta": float(cfg.get("beta", 0.04)),
        }
        if args.algo == "grpo":
            gen["generation_kwargs"] = {"use_cache": True}
            gen["mask_truncated_completions"] = True
            gen["loss_type"] = "grpo"
            gen["disable_dropout"] = True
            targs = GRPOConfig(**gen, **common)
            trainer = GRPOTrainer(model=model, reward_funcs=reward_funcs, args=targs,
                                  train_dataset=train_ds, processing_class=tokenizer,
                                  peft_config=peft_cfg,
                                  callbacks=[ProgressLog(output_dir)])
        else:
            rm_path = cfg.get("reward_model_path")
            if not rm_path or not Path(rm_path).exists():
                raise FileNotFoundError(
                    f"RLOO 需要已训好的奖励模型：reward_model_path={rm_path!r} 不存在。"
                    f"先跑 --algo reward。")
            from transformers import AutoModelForSequenceClassification
            rm = AutoModelForSequenceClassification.from_pretrained(
                rm_path, num_labels=1, dtype=dtype, trust_remote_code=True)
            rm_tok = AutoTokenizer.from_pretrained(rm_path, trust_remote_code=True)
            targs = RLOOConfig(**gen, **common)
            trainer = RLOOTrainer(model=model, reward_funcs=rm, args=targs,
                                  train_dataset=train_ds, processing_class=tokenizer,
                                  reward_processing_classes=rm_tok,
                                  peft_config=peft_cfg,
                                  callbacks=[ProgressLog(output_dir)])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", {"algo": args.algo, **cfg})
    write_json(output_dir / "trainer_config.json", trainer.args.to_dict())

    packages = {}
    for pkg in ("torch", "transformers", "trl", "peft", "accelerate", "datasets"):
        try:
            packages[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            packages[pkg] = None
    manifest = {
        "algo": args.algo,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(), "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "policy_start_and_reference": model_path,
        "reference_note": "peft_config 使参考策略 = 关掉 adapter 的 A2，无需第二份权重",
        "template_suffix": suffix, "tokenizer_eos": tokenizer.eos_token,
        "prompt_tokens": {"max": p_max, "mean": round(p_mean, 1),
                          "max_prompt_length": limit_key},
        "script_sha256": sha256(__file__), "config_sha256": sha256(args.config),
        "reward_schema": cfg.get("reward_schema", "legacy_four_faults"),
        "reward_script_sha256": sha256(ROOT / "tools" / (
            "hard_fault_metrics.py" if cfg.get("reward_schema") == "hard_faults_v1" else "rl_reward.py")),
        "feedback_source": cfg.get("feedback_source", "unspecified"),
        "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in trainer.model.parameters()),
        "status": "training",
        "data": dm,
    }
    write_json(output_dir / "run_manifest.json", manifest)

    if has_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    if has_cuda:
        torch.cuda.synchronize()
    train_wall = time.perf_counter() - t0
    peaks = cuda_peaks()
    monitor.flush()

    trainer.save_model(str(output_dir))
    trainer.save_state()
    tokenizer.save_pretrained(str(output_dir))

    metrics = dict(result.metrics)
    metrics.update({"algo": args.algo, "train_samples": len(train_ds),
                    "global_step": trainer.state.global_step,
                    "train_wall_seconds": train_wall, "cuda_memory_train": peaks})
    if eval_ds is not None:
        if has_cuda:
            torch.cuda.reset_peak_memory_stats()
        ev = trainer.evaluate()
        ev["cuda_memory"] = cuda_peaks()
        write_json(output_dir / "eval_metrics.json", ev)
        metrics["eval"] = ev
    write_json(output_dir / f"{args.algo}_train_metrics.json", metrics)

    manifest.update({"finished_utc": datetime.now(timezone.utc).isoformat(),
                     "main_wall_seconds": time.perf_counter() - started,
                     "train_wall_seconds": train_wall, "cuda_memory_train": peaks,
                     "status": "completed"})
    write_json(output_dir / "run_manifest.json", manifest)
    print(f"[done] {args.algo} adapter -> {output_dir}  "
          f"wall={train_wall/60:.1f}min peak={peaks.get('peak_reserved_mib', 0):.0f}MiB",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
