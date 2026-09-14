"""MS8 数据飞轮半自动闭环编排器。

闭环：should_retrain 触发 → 增量数据（真实:仿真 ≥1:3）→ 再训 → 自动评估 → 人工确认灰度。

设计原则（对齐 docs/SFT_MILESTONES.md MS8）：
- **半自动**：数据构建/训练/评估全自动，**仅最终切换端点由人工确认**（promote 只打印指令）；
- **真实:仿真 ≥1:3**：真实反馈样本稀缺，按最小 3 倍仿真稀释，避免灾难性遗忘；
- **可演练**：无真实反馈时可用 `--simulate-feedback N` 注入 N 条带 prompt 的合成真实样本，
  跑通全链路（验收要求"半自动演练一次"）。

阶段（子命令）：
    status   查看反馈统计与触发状态
    build    构建增量数据集 + 注册到 LLaMA-Factory dataset_info.json
    train    生成训练 yaml 并调用 llamafactory-cli
    eval     对指定端点跑 L2 全表 + 回归集零退化
    promote  打印人工灰度/回滚指令（不自动执行）
    all      status → build → train → eval → promote（--dry-run 只打印命令）

用法：
    conda run -n mff_agent python tools/flywheel.py status
    conda run -n mff_agent python tools/flywheel.py all --dry-run
    conda run -n mff_agent python tools/flywheel.py build --simulate-feedback 5
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from action.feedback import Feedback, FeedbackStore  # noqa: E402
from config import get_settings  # noqa: E402
from reasoning.knowledge_graph import KnowledgeGraph  # noqa: E402

# 与 data/sft/train.jsonl 保持严格一致的 system 消息（防分布漂移）
SYS_PROMPT = ("你是中频炉水冷系统的故障诊断专家。"
              "输出先给推理过程（引用统计特征数值做鉴别），"
              "最后输出唯一一个JSON对象，不要输出多余内容。")

DEFAULT_FB = "data/feedback/feedback.jsonl"
DEFAULT_SIM = "data/sft/train.jsonl"
DEFAULT_OUT_DIR = "data/sft/incremental"
LF_DATASET_INFO = Path.home() / "codes/LLaMA-Factory/data/dataset_info.json"
LF_TRAIN_YAML = "tools/mff_full_sft.yaml"

SIM_RATIO = 3          # 真实:仿真 = 1:3（仿真至少 3 倍）


# ---------------- 数据构建 ----------------

def feedback_to_sample(rec: Dict, kg: KnowledgeGraph) -> Optional[Dict]:
    """单条反馈记录 -> SFT 训练样本（人工校正后金标签）。

    仅接受「真实故障 + 含输入 prompt + 运维确认根因」的记录；
    其余（误报/缺 prompt/缺根因）跳过，返回 None。
    """
    fb = rec.get("feedback", {}) or {}
    diag = rec.get("diagnosis", {}) or {}
    prompt = diag.get("prompt")
    actual = fb.get("actual_root_cause")
    if not fb.get("is_true_fault") or not prompt or not actual:
        return None
    sop = kg.actions_for_fault(actual) or []
    out = {
        "root_cause": actual,
        "confidence": 0.85,
        "evidence": ["人工复核确认根因（反馈校正样本）"],
        "sop": sop,
    }
    assistant = (f"（人工反馈校正）经运维复核，实际根因为{actual}，"
                 f"依据现场处置确认。\n" + json.dumps(out, ensure_ascii=False))
    return {
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {"kind": "feedback", "gold": actual,
                 "order_id": fb.get("order_id"), "source": "feedback"},
    }


def load_feedback_samples(fb_path: Path, kg: KnowledgeGraph) -> List[Dict]:
    if not fb_path.is_file():
        return []
    samples = []
    for line in fb_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = feedback_to_sample(rec, kg)
        if s:
            samples.append(s)
    return samples


def load_sim_samples(sim_path: Path, n: int, seed: int = 42) -> List[Dict]:
    if not sim_path.is_file() or n <= 0:
        return []
    rows = [json.loads(l) for l in sim_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    if n >= len(rows):
        return rows
    return rng.sample(rows, n)


def simulate_feedback(n: int, kg: KnowledgeGraph, fb_path: Path) -> None:
    """注入 N 条合成真实反馈（带 prompt + 校正根因），用于无真实数据时演练全链路。

    幂等：合成 order_id 固定为 WO-SIMFLY-xxxx，已存在的跳过，重复运行不重复注入。
    """
    existing = set()
    if fb_path.is_file():
        for line in fb_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                existing.add(json.loads(line).get("feedback", {}).get("order_id"))
            except json.JSONDecodeError:
                continue
    store = FeedbackStore(str(fb_path))
    cases = [
        ("管道泄漏", "【实时异常特征】{\"湿度\": 74.2, \"压力\": 140.1}\n【统计鉴别特征】\n- 湿度均值_pctRH: 74.2\n- 湿度上升量_pctRH: 5.6"),
        ("水泵气蚀", "【实时异常特征】{\"压力\": 182.0, \"流量\": 6.3}\n【统计鉴别特征】\n- 压力波动幅度_std_kPa: 6.1\n- 湿度均值_pctRH: 55.0"),
        ("过滤器堵塞", "【实时异常特征】{\"流量\": 5.4, \"压力\": 150.0}\n【统计鉴别特征】\n- 流量_L/s: 5.4\n- PQ特性偏移_pct: 2.1"),
        ("线圈结垢", "【实时异常特征】{\"出水温度\": 54.0, \"流量\": 7.6}\n【统计鉴别特征】\n- 进出水温差_℃: 23.5\n- PQ特性偏移_pct: 12.4"),
        ("管道泄漏", "【实时异常特征】{\"湿度\": 71.5, \"水箱液位\": 190.0}\n【统计鉴别特征】\n- 湿度均值_pctRH: 71.5\n- 湿度上升量_pctRH: 4.8"),
    ]
    for i in range(n):
        gold, prompt = cases[i % len(cases)]
        order_id = f"WO-SIMFLY-{i:04d}"
        if order_id in existing:
            continue
        diag = {"prompt": prompt,
                "output": json.dumps({"root_cause": "水泵气蚀", "confidence": 0.9}, ensure_ascii=False),
                "root_cause": "水泵气蚀"}
        store.archive(Feedback(order_id, gold, True, 25.0, "演练合成反馈"),
                      diagnosis_snapshot=diag)
    print(f"[simulate] 已注入（幂等）合成真实反馈 -> {fb_path}")


def mix_incremental(real: List[Dict], sim: List[Dict]) -> List[Dict]:
    """真实 + 仿真混合，保证仿真 ≥ SIM_RATIO × 真实（真实:仿真 ≤1:3）。"""
    data = list(real) + list(sim)
    random.Random(42).shuffle(data)
    return data


def build(args, kg: KnowledgeGraph) -> Dict:
    fb_path = Path(args.feedback)
    if args.simulate_feedback:
        simulate_feedback(args.simulate_feedback, kg, fb_path)

    real = load_feedback_samples(fb_path, kg)
    if not real and not args.allow_empty:
        return {"ok": False, "reason": "无可用真实反馈样本（需 is_true_fault+prompt+根因）；"
                                      "可用 --simulate-feedback N 演练"}
    n_sim = max(SIM_RATIO * len(real), args.min_sim)
    sim = load_sim_samples(Path(args.sim), n_sim)
    rows = mix_incremental(real, sim)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_p = out_dir / "train.jsonl"
    with open(train_p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 增量集较小，验证集直接复用主 val（防泄漏口径不变）
    val_src = Path("data/sft/val.jsonl")
    val_p = out_dir / "val.jsonl"
    if val_src.is_file():
        shutil.copyfile(val_src, val_p)

    meta = {"real": len(real), "sim": len(sim), "total": len(rows),
            "ratio": (f"1:{len(sim) / len(real):.1f}" if real else "n/a"),
            "train": str(train_p), "val": str(val_p)}
    print(json.dumps({"ok": True, **meta}, ensure_ascii=False, indent=2))
    return {"ok": True, **meta}


def register_datasets(train_p: str, val_p: str) -> bool:
    """把增量数据集注册进 LLaMA-Factory dataset_info.json（幂等）。"""
    if not LF_DATASET_INFO.is_file():
        print(f"[warn] 未找到 {LF_DATASET_INFO}（训练时需手动注册）")
        return False
    info = json.loads(LF_DATASET_INFO.read_text(encoding="utf-8"))
    tags = {"role_tag": "role", "content_tag": "content", "user_tag": "user",
            "assistant_tag": "assistant", "observation_tag": "observation",
            "function_tag": "function_call", "system_tag": "system"}
    for name, path in (("mff_diag_incr_train", train_p), ("mff_diag_incr_val", val_p)):
        info[name] = {"file_name": str(Path(path).resolve()), "formatting": "sharegpt",
                      "columns": {"messages": "messages"}, "tags": dict(tags)}
    LF_DATASET_INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[register] mff_diag_incr_train / mff_diag_incr_val -> {LF_DATASET_INFO}")
    return True


# ---------------- 阶段 ----------------

def stage_status(args, kg) -> int:
    store = FeedbackStore(args.feedback)
    stats = store.stats()
    # 重新加载 jsonl 以获得持久化计数（FeedbackStore.records 仅内存）
    fb_path = Path(args.feedback)
    n_lines = sum(1 for l in fb_path.read_text(encoding="utf-8").splitlines() if l.strip()) if fb_path.is_file() else 0
    samples = load_feedback_samples(fb_path, kg)
    pairs = store.build_preference_pairs()
    print(json.dumps({
        "feedback_total_lines": n_lines,
        "usable_real_samples": len(samples),
        "dpo_preference_pairs": len(pairs),
        "should_retrain(min=5)": n_lines >= 5,
        "ms7_threshold(>=500)": len(pairs) >= 500,
    }, ensure_ascii=False, indent=2))
    return 0


def stage_build(args, kg) -> int:
    r = build(args, kg)
    if not r.get("ok"):
        print(f"[build] 跳过: {r['reason']}")
        return 1
    if not args.no_register:
        register_datasets(r["train"], r["val"])
    return 0


def _gen_train_yaml(args, incr: Dict) -> str:
    """基于 tools/mff_full_sft.yaml 生成增量训练 yaml（数据/输出目录替换）。"""
    src = Path(LF_TRAIN_YAML).read_text(encoding="utf-8")
    src = src.replace("dataset: mff_diag_train", "dataset: mff_diag_incr_train")
    src = src.replace("eval_dataset: mff_diag_val", "eval_dataset: mff_diag_incr_val")
    # dataset_dir 默认相对 CWD；显式指向 LLaMA-Factory 的 data 目录（含 dataset_info.json）
    src = src.replace("dataset: mff_diag_incr_train",
                      f"dataset: mff_diag_incr_train\ndataset_dir: {LF_DATASET_INFO.parent}")
    src = src.replace("output_dir: /home/huachenghao/codes/Agent_MFF_Early_Warning/saves/mff-sft-full",
                      f"output_dir: {Path(args.out_dir).resolve().parent / 'mff-sft-incr'}")
    if args.epochs:
        src = src.replace("num_train_epochs: 3.0", f"num_train_epochs: {args.epochs}")
    if args.max_steps:
        src = src.replace("num_train_epochs: 3.0", "max_steps: %d" % args.max_steps)
    dst = Path(args.out_dir) / "mff_incr_sft.yaml"
    dst.write_text(src, encoding="utf-8")
    return str(dst)


def stage_train(args, kg) -> int:
    incr = Path(args.out_dir)
    train_p, val_p = incr / "train.jsonl", incr / "val.jsonl"
    if not train_p.is_file():
        print("[train] 缺少增量数据，请先 build")
        return 1
    if not args.dry_run:
        register_datasets(str(train_p), str(val_p))
    yaml_path = _gen_train_yaml(args, {"train": str(train_p), "val": str(val_p)})
    cmd = (f"CUDA_VISIBLE_DEVICES={args.gpu} "
           f"$HOME/.conda/envs/mff_sft/bin/llamafactory-cli train {yaml_path}")
    if args.dry_run:
        print(f"[dry-run] {cmd}")
        return 0
    print(f"[train] {cmd}")
    return subprocess.call(cmd, shell=True)


def stage_eval(args, kg) -> int:
    eval_cmd = (f"conda run -n mff_agent python tests/eval_sft_model.py "
                f"--url {args.eval_url} --model {args.eval_model} --arbitration off")
    reg_cmd = (f"conda run -n mff_agent python tests/run_regression.py "
               f"--url {args.eval_url} --model {args.eval_model}")
    if args.dry_run:
        print(f"[dry-run] {eval_cmd}")
        print(f"[dry-run] {reg_cmd}")
        return 0
    rc1 = subprocess.call(eval_cmd, shell=True)
    rc2 = subprocess.call(reg_cmd, shell=True)
    return 0 if (rc1 == 0 and rc2 == 0) else 1


def stage_promote(args, kg) -> int:
    print("""
===== 人工灰度 / 回滚（本脚本不自动执行）=====
1. 确认 L2 全表硬门槛全绿、回归集零退化；
2. 起新端点（示例 3762）：
     CUDA_VISIBLE_DEVICES=0 $HOME/.conda/envs/mff_sft_serve/bin/vllm serve \\
       <新产物目录> --served-model-name mff-sft-incr --max-model-len 32768 --port 3762
3. 改 .env 指向新端点后重启主服务；观察升级率与根因；
4. 回滚：.env 改回原端点并重启（30s 内），或 vLLM 换端口。
=============================================""")
    return 0


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="MS8 数据飞轮半自动闭环")
    ap.add_argument("--feedback", default=DEFAULT_FB)
    ap.add_argument("--sim", default=DEFAULT_SIM)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--min-sim", type=int, default=30, help="仿真样本下限")
    ap.add_argument("--simulate-feedback", type=int, default=0, help="注入 N 条合成真实反馈（演练）")
    ap.add_argument("--allow-empty", action="store_true", help="无真实样本也构建（仅仿真）")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=float, default=0.0, help="覆盖训练轮数")
    ap.add_argument("--max-steps", type=int, default=0, help="覆盖为固定步数（冒烟）")
    ap.add_argument("--eval-url", default="http://localhost:3762/v1")
    ap.add_argument("--eval-model", default="mff-sft-incr")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("stage", choices=["status", "build", "train", "eval", "promote", "all"])
    args = ap.parse_args()

    kg = KnowledgeGraph()
    if args.stage == "status":
        return stage_status(args, kg)
    if args.stage == "build":
        return stage_build(args, kg)
    if args.stage == "train":
        return stage_train(args, kg)
    if args.stage == "eval":
        return stage_eval(args, kg)
    if args.stage == "promote":
        return stage_promote(args, kg)
    # all
    rc = stage_status(args, kg)
    if rc:
        return rc
    rc = stage_build(args, kg)
    if rc:
        return rc
    rc = stage_train(args, kg)
    if rc:
        return rc
    rc = stage_eval(args, kg)
    if rc:
        return rc
    return stage_promote(args, kg)


if __name__ == "__main__":
    sys.exit(main())
