# LLM SFT（监督微调）后训练实施指南

> 目标：对 L3 根因诊断所用的 LLM 进行 SFT 及后续 DPO/GRPO 后训练，将中频炉水冷系统专业知识（故障签名、物理鉴别规则、图谱因果、SOP）**内化到模型权重**，从而：
>
> 1. 缩短/简化 `reasoning/root_cause.py` 中日益膨胀的 CoT 提示（当前提示已内置 6 条硬判定规则，属"提示工程承载专业知识"，脆弱且占 token）；
> 2. 在 `enable_thinking=false`（3~5s 时延要求）下提升非思考模式的多跳推理正确率，减少对"统计硬校验仲裁"的依赖（当前仲裁逻辑兜底了 LLM 的误判）；
> 3. 让置信度输出更校准（当前 `confidence` 需代码侧 `+0.1/-0.1` 修正）。

**结论先行**：推荐 **LoRA SFT（首选，4×RTX 3090 可行）→ 离线评估 → vLLM 部署 A/B →（可选）DPO/GRPO 强化** 的路线。不推荐对 27B 全参微调（显存不够），也不建议先上 RLHF（数据量不足，见第 8 节）。

> 当前双端点：big 模型实为 **Qwen3.8-27B**（`.env` 中 `big_model_name` 写作 `Qwen3.6-27B`，实际服务的是 Qwen3.8-27B，负责 L3 根因推理），small 模型为 Qwen3.6-27B-INT4。SFT 针对的是 **big 模型链路**（`RootCauseReasoner` 的实际调用对象）；训练底座需取原始 bf16 权重，两个推理端点均不能直接用于训练，详见 1.3。

---

## 目录

1. [可行性分析与路线选择](#1-可行性分析与路线选择)
2. [训练数据构造（本项目的最大优势）](#2-训练数据构造本项目的最大优势)
3. [数据格式化与质量审计](#3-数据格式化与质量审计)
4. [LoRA SFT 训练（LLaMA-Factory）](#4-lora-sft-训练llama-factory)
5. [离线评估与验收标准](#5-离线评估与验收标准)
6. [部署、切换与回滚](#6-部署切换与回滚)
7. [进阶后训练：DPO / GRPO](#7-进阶后训练dpo--grpo)
8. [风险与注意事项](#8-风险与注意事项)
9. [与项目闭环（持续优化智能体）的衔接](#9-与项目闭环持续优化智能体的衔接)
10. [附录：命令速查](#10-附录命令速查)

---

## 1. 可行性分析与路线选择

### 1.1 现状诊断

| 现状 | 证据（代码位置） | 问题 |
|---|---|---|
| 专业知识靠提示注入 | `reasoning/root_cause.py:65-121` `_build_prompt`，含湿度>70%RH 判泄漏、压力std>3kPa 判气蚀等硬规则 | 每次推理消耗 ~2000 token 提示；规则改动需改代码；模型本身未学会这些知识 |
| LLM 误判靠代码仲裁兜底 | `root_cause.py:178-219`：图谱 Top1 仲裁、统计强先验回退、堵塞/结垢 PQ 偏移改判 | 说明模型在非思考模式下确实会混淆（如把气蚀判成泄漏、凝露误判）——正是 SFT 的靶点 |
| 置信度不可信 | `root_cause.py:179,183` 代码侧 `+0.1`/`-0.1` 修正 | SFT 可用带校准置信度标签的数据直接训练 |
| 反馈闭环已预留微调钩子 | `action/feedback.py:45-51` `should_retrain(min_samples=5)` | 天然的数据飞轮入口，但 `diagnosis` 快照目前为空（见 9.1） |

### 1.2 三种知识注入方式对比

| 方式 | 成本 | 效果 | 适用 |
|---|---|---|---|
| 提示工程（现状） | 低 | 边际递减，已接近上限 | 已用尽 |
| RAG（知识库检索增强） | 中 | 对"事实查询"有效，对"多跳鉴别推理"帮助有限；本项目判定规则已在提示中，检索不会更优 | 跳过 |
| **SFT 后训练** | 中高 | 将鉴别规则、物理直觉内化，可缩短提示、提升非思考模式推理 | **本方案** |

### 1.3 硬件与底座选型（4× RTX 3090，24GB×4）

| 方案 | 可行性 | 说明 |
|---|---|---|
| 27B 全参 SFT | ❌ | bf16 权重 54GB + 优化器状态，24GB×4 远不够（ZeRO-3+offload 勉强但极慢） |
| **27B LoRA SFT** | ✅ 首选 | 冻结底座，仅训 rank 16~64 适配器；单卡可跑，4 卡数据并行更快 |
| 14B/8B LoRA SFT | ✅ 备选 | 若后续想缩小部署体积（INT4 量化后单卡即可服务） |
| 27B QLoRA | ✅ 兜底 | 显存最省（底座 4bit 常驻），速度最慢，仅当 LoRA 显存不足时用 |

**当前双端点实况**（`.env`，注意 `big_model_name` 配置值与实际模型不完全一致）：

| 端点 | 配置名（`.env`） | 实际模型 | 用途 |
|---|---|---|---|
| `url`（172.25.67.140:9997） | `Qwen3.6-27B` | **Qwen3.8-27B** | big 模型，L3 根因推理（`LLMClient` 默认 `prefer="big"`） |
| `base_url`（172.25.67.120:9997） | `Qwen3.6-27B-INT4` | Qwen3.6-27B-INT4 | small 模型，轻量任务 |

> 两个注意点：
> 1. **SFT 的对象是 big 模型（Qwen3.8-27B）所在的链路**——这是 `RootCauseReasoner` 实际调用的模型（`llm_client.py:59` 默认走 `url`/`big_model_name` 端点）。
> 2. SFT 需要**底座模型的原始 bf16 权重**（HuggingFace 官方权重），当前两个端点都是 vLLM 推理服务、拿不到权重。若无法获得 Qwen3.8-27B 原始权重，则退而选择可公开获取的底座（如 Qwen3-14B/Qwen3-8B）做 SFT，并在评估阶段证明"小模型 SFT 后 ≥ 27B 提示工程"——这本身也是很好的技术叙事（大赛加分点）。INT4 量化版不适合直接做训练底座（量化权重不可微调），但训完的模型可以再量化成 INT4 部署，保持与现有服务相同的体积与时延（见 6.5）。

### 1.4 路线图

```
阶段0 底座与工具准备 ──► 阶段1 数据构造(2~5k样本) ──► 阶段2 LoRA SFT
                                                        │
        阶段5 闭环迭代 ◄── 阶段4 部署A/B ◄── 阶段3 离线评估(不达标→回阶段1补数据)
                              │
                              └─(数据量够后)──► 阶段6 DPO/GRPO（可选）
```

---

## 2. 训练数据构造（本项目的最大优势）

本项目拥有完整的数据自生成能力，**无需人工标注**即可构造高质量 SFT 数据对：

| 数据源 | 用途 |
|---|---|
| `simulator/`（物理机理仿真，4 类故障注入） | 生成海量带真值标签的故障场景：正常 / 过滤器堵塞 / 水泵气蚀 / 管道泄漏 / 线圈结垢（含 `scale_severe`、`mixed_faults` 复合故障） |
| `detection/`（L1 规则 + L2 预测 + 统计鉴别特征） | 生成与线上完全一致的输入上下文（L1 告警、L2 预测、stats 窗口特征） |
| `reasoning/knowledge_graph.py`（34 节点 39 边） | 提供 SOP 标签（`actions_for_fault`）与候选根因先验 |
| `context/`（维修工单 / 工况运行表） | 充实输入上下文，训练模型正确使用辅助证据（工单仅辅助、不得单独定论） |
| `data/feedback/feedback.jsonl` | 真实（当前为演示）反馈归档：`actual_root_cause` 即金标签 |
| 大模型自身（thinking 开启） | CoT 蒸馏：让 27B 开思考模式生成高质量推理链，作为 SFT 的思维链监督（见 2.3） |

### 2.1 输入-输出对的设计原则

**SFT 样本的输入必须与线上 `_build_prompt` 的实际输入分布一致**（否则训了不生效）：

- **输入（user message）**：复用 `root_cause._build_prompt` 的模板，但**做两个刻意的退化**，以逼模型自己补足知识：
  1. 删除模板中"判定规则（严格遵循）"整节及 3 条"注意"——这些是要内化的知识；
  2. 保留 L1/L2/stats/工况/维修工单/图谱候选等**客观数据**（这是要学的证据利用能力）。
- **输出（assistant message）**：包含两段——先自然语言推理链（引用 stats 数值做鉴别，训练非思考模式下的隐性 CoT），后严格 JSON（`root_cause / confidence / evidence / sop`），JSON 格式与 `DiagnosisResult.to_dict()` 对齐。

### 2.2 样本配比（关键：覆盖易混淆对与负样本）

| 样本类别 | 数量建议 | 设计意图 |
|---|---|---|
| 4 类故障 × 各严重度梯度（轻/中/重） | 每类 ≥400 | 主任务 |
| **混淆对**：气蚀 vs 泄漏（湿度边界 65~70%RH）、堵塞 vs 结垢（PQ 偏移 8~12% 边界） | ≥600 | 当前代码仲裁专门兜底的场景，SFT 的核心靶点 |
| 工况切换正常波动（`condition_transition=True`，L2 异常分升高但无故障） | ≥300 | 防误报：训练"判无故障"输出（`root_cause: "无故障"`，见 2.5） |
| 复合故障（`mixed_faults`：两故障同现） | ≥200 | 训练主次根因排序 |
| 维修工单误导样本（工单记录 A 故障，但实时 stats 指向 B） | ≥200 | 训练"工单仅辅助证据"原则 |
| 防幻觉负样本：证据含越界数值、图谱外根因的拒答/纠正 | ≥100 | 训练自我校验意识 |

**总量建议 2500~5000 条**。质量优先于数量：每类边界样本必须有明确的金标签（由仿真注入的故障类型直接给出，零标注成本）。

### 2.3 推理链（CoT）三种来源，按优先级

1. **模板化手写**（最可控，推荐主力）：按根因类别各写 2~3 条推理链骨架，程序填入实际 stats 数值，例：

   > 进水压力尾段去趋势 std=4.8kPa > 3kPa，存在压力震荡；湿度均值 58%RH ≤ 65%RH 且上升量 1.2%RH < 4%RH，无水汽逸散证据；排除泄漏。流量轻微下降但 PQ 特性偏移 +6% < +10%，排除线圈结垢。压力震荡 + 湿度正常符合水泵气蚀的 NPSH 不足签名……置信度 0.87。

2. **大模型蒸馏**（补充多样性）：对同一输入，用现有 27B 端点开 `enable_thinking=True` 生成推理链，**仅当最终结论与仿真金标签一致时才收录**（自动过滤错误思维链）。

3. **代码仲裁日志回放**：历史上被 `root_cause.py:202-219` 纠正过的错误推理，构造成"正确版"样本——这些恰是模型最弱的点。

### 2.4 置信度标签校准

给 `confidence` 标签时不要一律 0.95，而按证据强度分级，例如：

- 证据全部命中且无矛盾：0.90~0.95
- 边界样本（如湿度 68%RH 判泄漏）：0.75~0.85
- 复合故障主次判断：主根因 0.85+，并要求 evidence 中说明为何非次根因

这样训练后模型的 confidence 天然分层，下游 `ConfidenceGate`（≥90% 直出 / 70~90% Top3 / <70% 人工）的分流才有意义，也可退役代码侧的 `±0.1` 修正。

### 2.5 输出 schema 的一个扩展决策

现有 `DiagnosisResult.root_cause` 隐含"必须有故障"。建议 SFT 数据显式支持 `root_cause: "无故障（工况切换正常波动）"`，并同步在 `root_cause.py` 的仲裁逻辑中放行该值（需小改：图谱候选为空且 LLM 判无故障时不强行回退 kg_top1）。这是消除"工况切换误报"的根本手段。

### 2.6 数据生成脚本骨架

新建 `tools/gen_sft_data.py`（示例骨架，正式实现放该文件）：

```python
"""SFT 数据生成：仿真场景 → 线上同构 report → (退化提示, 金标签推理链+JSON)。"""
from simulator import run_scenario          # 4类故障注入 + 正常/工况切换
from detection import build_report          # 复用 L1/L2/stats 组包逻辑（与 server 实时流同构）
from reasoning.knowledge_graph import KnowledgeGraph

TEMPLATE_COT = {...}   # 2.3 的手写推理链模板，按根因类别

def make_sample(scenario, seed):
    ts = run_scenario(scenario, seed=seed)          # 带故障真值标签
    report = build_report(ts)                       # l1_alerts / l2_forecast / stats / diag_context
    gold = scenario.fault_name                      # 仿真注入的根因 = 金标签
    prompt = build_degraded_prompt(report)          # 2.1 的退化版 _build_prompt（去判定规则节）
    cot = fill_template(TEMPLATE_COT[gold], report) # 推理链 + 校准置信度(2.4)
    answer = cot + "\n" + json.dumps({...}, ensure_ascii=False)
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer}]}

# 按第 2.2 节配比循环生成 → data/sft/train.jsonl（+ val.jsonl 留 10%）
```

关键正确性要求：
- `build_report` **必须与 `server` 实时流上报给 `diagnose(report=...)` 的字段完全同构**（含 stats 的中文键名如 `压力波动幅度_std_kPa`），否则训练/推理分布漂移；
- 每条样本记录 `meta: {fault, severity, seed}`，便于评估时分层统计。

---

## 3. 数据格式化与质量审计

### 3.1 格式（LLaMA-Factory / ms-swift 通用 sharegpt 格式）

```json
{
  "messages": [
    {"role": "system", "content": "你是中频炉水冷系统的故障诊断专家。输出先给推理过程，最后输出唯一一个JSON对象。"},
    {"role": "user", "content": "<退化版诊断提示，含L1/L2/stats/工况/工单/图谱候选>"},
    {"role": "assistant", "content": "<推理链>\n{\"root_cause\": \"水泵气蚀\", \"confidence\": 0.87, \"evidence\": [\"...\"], \"sop\": [\"...\"]}"}
  ]
}
```

> **模板一致性警告**：训练时的 chat template 必须与推理时一致。Qwen3 系列自带 `enable_thinking` 开关的 chat template——SFT 时把 assistant 消息按**非思考格式**渲染（thinking 关闭时 template 会自动附加空 think 标签或直接输出），确保与 vLLM 部署后 `chat_template_kwargs={"enable_thinking": false}` 的行为一致。LLaMA-Factory 对 Qwen3 有现成处理，用 `template: qwen3` 即可。

### 3.2 人工抽检清单（训练前必做，抽 5%约 100~250 条）

- [ ] JSON 可被 `json.loads` 解析（复用 `root_cause._parse` 的正则逻辑验证）；
- [ ] `root_cause` 与仿真金标签一致；
- [ ] `evidence` 中引用的数值与输入 stats 一致（**杜绝编造数值**——这是防幻觉的底线，宁可 evidence 短）；
- [ ] `sop` 与知识图谱 `actions_for_fault(金标签)` 一致或为其合理子集；
- [ ] 推理链中的每步鉴别都引用了真实存在的特征。

写一个 `tools/audit_sft_data.py` 自动执行前 4 项，人工只看第 5 项。

---

## 4. LoRA SFT 训练（LLaMA-Factory）

工具选 **LLaMA-Factory**（对 Qwen3 + sharegpt 格式 + LoRA + 多卡支持最成熟，命令行/Web 板皆可；`ms-swift` 为备选）。

### 4.1 环境准备（独立 conda 环境，勿污染 `mff_agent`）

```bash
conda create -n sft python=3.10 -y
conda activate sft
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]" deepspeed

# 底座权重：HuggingFace 下载（Qwen3 系列原始 bf16 权重，非推理端点的 INT4 版）
huggingface-cli download <底座模型repo> --local-dir /data/models/<底座>
```

### 4.2 注册数据集

```bash
# LLaMA-Factory/data/dataset_info.json 追加：
"mff_diag": {
  "file_name": "/abs/path/to/Agent_MFF_Early_Warning/data/sft/train.jsonl",
  "formatting": "sharegpt",
  "columns": {"messages": "messages"}
}
```

### 4.3 LoRA 训练配置（`mff_lora_sft.yaml`）

```yaml
model_name_or_path: /data/models/<底座>
stage: sft
do_train: true
finetuning_type: lora
lora_target: all            # 对所有线性层挂 LoRA
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05

template: qwen3
dataset: mff_diag
cutoff_len: 4096             # 诊断提示~2k token + 推理链输出
preprocessing_num_workers: 8

output_dir: saves/mff-qwen-lora
per_device_train_batch_size: 2
gradient_accumulation_steps: 8   # 有效 batch=16
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
logging_steps: 10
save_steps: 200
eval_steps: 200
val_size: 0.1                # 从训练集切 10% 做验证
per_device_eval_batch_size: 4
```

### 4.4 启动（4×3090 数据并行）

```bash
FORCE_TORCHRUN=1 llamafactory-cli train mff_lora_sft.yaml
# 显存不足时：加 deepspeed: ds2_zero_offload.yaml（优化器状态卸载到内存）
```

观察要点：
- loss 应在 0.2~1.0 区间内稳定下降，epoch 2 后趋平（过拟合信号：eval loss 回升即早停）；
- 3 epochs、5k 样本、14B LoRA 在 4×3090 上约数小时量级；
- 若 loss 降但评估不涨（第 5 节），通常是数据问题不是超参问题——**回第 2 节修数据，不要盲目调参**。

### 4.5 合并权重（部署用）

```bash
llamafactory-cli export \
  --model_name_or_path /data/models/<底座> \
  --adapter_name_or_path saves/mff-qwen-lora \
  --template qwen3 \
  --export_dir /data/models/mff-qwen-sft-merged
```

---

## 5. 离线评估与验收标准

**训练完成 ≠ 可用。必须先离线评估达标，再动线上端点。**

### 5.1 评估集

- 独立于训练集重新生成（不同 seed），约 200 条，按第 2.2 节同配比；
- 外加**回归集**：把线上历史 case（`logs/` 中 L3 诊断日志、`data/feedback/feedback.jsonl` 的快照）整理为固定评估集，防止"新能力上来、老 case 崩"。

### 5.2 评估脚本（新建 `tests/eval_sft_model.py`）

复用现有测试基建，直接实例化 `LLMClient` 指向新模型：

```python
from reasoning.llm_client import LLMClient
from reasoning.root_cause import RootCauseReasoner

llm = LLMClient(config={
    "url": "http://localhost:3761/v1",   # 新模型 vLLM 服务
    "key": "empty",
    "big_model_name": "mff-qwen-sft-merged",
    "enable_thinking": False,             # 必须与线上一致！
})
reasoner = RootCauseReasoner(llm=llm)

# 对评估集逐条 diagnose()，与金标签比对
```

### 5.3 指标与验收门槛

| 指标 | 计算方式 | 门槛 |
|---|---|---|
| 根因准确率（总体） | `root_cause == gold` 比例 | ≥ 提示工程基线（当前线上 100% 是**含代码仲裁**的成绩；SFT 目标：**关闭仲裁逻辑后** ≥95%，含仲裁 100%） |
| 混淆对准确率 | 仅气蚀/泄漏、堵塞/结垢边界子集 | ≥90%（当前无仲裁时这是重灾区） |
| JSON 合法率 | `root_cause._parse` 成功比例 | ≥99% |
| 证据忠实率 | evidence 引用数值存在于输入中 | ≥98%（防幻觉底线） |
| 防幻觉校验通过率 | `AntiHallucinationChecker.check().passed` 比例 | 100%（物理违规零容忍） |
| 置信度校准 | 判对样本平均 conf − 判错样本平均 conf | > 0.15（区分度）；判错样本 conf < 0.8（安全） |
| 推理时延 | vLLM 部署后端到端 | ≤5s（维持 `enable_thinking=false` 契约） |
| 无故障样本误报率 | 工况切换子集判"有故障"比例 | ≤5% |

**任一硬指标（JSON 合法率 / 证据忠实率 / 防幻觉通过率）不达标 → 数据回炉，不上线。**

---

## 6. 部署、切换与回滚

### 6.1 vLLM 服务新模型

```bash
conda activate sft
vllm serve /data/models/mff-qwen-sft-merged \
  --served-model-name mff-qwen-sft \
  --port 3761 --max-model-len 8192
# 探活：curl http://localhost:3761/v1/models
```

注意：本项目 `LLMClient` 通过 `chat_template_kwargs: {"enable_thinking": false}` 关思考（`llm_client.py:78-79`），vLLM 原生支持该透传，合并后的权重保留了 Qwen3 chat template，行为一致。

### 6.2 灰度切换（利用现有配置体系，零代码改动）

`.env` / 环境变量即可切换（`config/settings.yaml` llm 段 → `.env` → 环境变量的既有优先级链）：

```bash
# 灰度：先把 SFT 模型配到 small_model（base_url 端点），big 保持不动
# 全量：确认后替换 url/big_model_name 指向新端点
```

推荐**双端点并行**一段时间：保留旧 27B 端点不动，新开 3761 端口服务 SFT 模型，通过 `.env` 切换，回归有问题 30 秒内改回。

### 6.3 提示词的渐进退役（重要，勿一步到位）

SFT 模型已内化判定规则，但**第一阶段保持 `_build_prompt` 原样**（含完整判定规则）——先验证"模型 + 提示"不劣于基线；达标后进入第二阶段再删除"判定规则"节做 A/B，观察指标后决定是否永久瘦身提示（省 token、降时延）。每一步都要跑第 5.3 节回归集。

### 6.4 回滚预案

- 权重、数据、评估报告全部版本化（`data/sft/`、`saves/`、`design/` 记录）；
- `.env` 切回旧端点（big 指回 172.25.67.140:9997）即完成回滚；
- vLLM 服务进程独立于 uvicorn API，重启互不影响。

### 6.5 INT4 量化部署（对齐线上小模型的体积与时延）

线上 small 端点已是 Qwen3.6-27B-INT4（vLLM 量化版）。SFT 产物如需长期替代 27B 端点、复用同等推理资源，走同样的量化路线：

```bash
# 1. 合并 LoRA 得到 bf16 全量权重（见 4.5）
# 2. 量化为 INT4（GPTQ/AWQ 任选，与现有 INT4 服务同规格）
llamafactory-cli export \
  --model_name_or_path /data/models/mff-qwen-sft-merged \
  --export_quantization_bit 4 \
  --export_quantization_block 128 \
  --export_dir /data/models/mff-qwen-sft-int4

# 3. vLLM 服务量化版
vllm serve /data/models/mff-qwen-sft-int4 \
  --served-model-name mff-qwen-sft-int4 --port 3761 --max-model-len 8192
```

**量化前后都要跑一遍第 5.3 节评估集**：量化对边界样本（湿度 65~70%RH、PQ 偏移 8~12%）的鉴别可能有可测的精度损失，若混淆对准确率量化后掉 >3 个百分点，则放弃量化、直接部署 bf16 版（27B bf16 单卡放不下，可 2 卡张量并行：`vllm serve ... --tensor-parallel-size 2`）。

---

## 7. 进阶后训练：DPO / GRPO

SFT 达标且反馈数据积累到位后（**建议 ≥500 条带真实反馈的偏好对**，当前 `feedback.jsonl` 的 `diagnosis` 快照为空，先补 9.1），可继续强化：

### 7.1 DPO（偏好对齐，首选，稳定）

- **偏好对来源**：同一输入下"正确诊断(chosen) vs 历史误判(rejected)"。历史误判样本天然存在：被 `root_cause.py` 仲裁逻辑纠正过的 LLM 原始输出（误判气蚀为泄漏等），`raw` 字段已保存于 `DiagnosisResult`。
- 偏好目标不仅是结论正确，还包括：置信度校准（误判时低置信）、证据忠实（不编数值）、JSON 合法。
- LLaMA-Factory 直接支持 `stage: dpo`，在 SFT 后的模型上继续训练（`pref_beta: 0.1`，lr 降一个量级 `5e-6`）。

### 7.2 GRPO / 规则奖励 RL（本项目特性红利，可作为技术亮点）

`AntiHallucinationChecker` + 金标签本质是一个**免费的程序化奖励函数**：

```
reward = 1.0×(根因==金标签) + 0.3×(JSON合法) + 0.3×(防幻觉校验通过) + 0.2×(置信度校准项) − 0.5×(编造数值)
```

用 TRL 的 `GRPOTrainer` 以仿真数据无限生成带奖励的 rollout，无需人工偏好标注。27B 在 4×3090 上做 GRPO 偏重，建议仅对 14B 以下底座尝试；若走此路线，先在 SFT 模型基础上小步验证。

### 7.3 何时停手

SFT 解决"知识内化"，DPO/GRPO 解决"偏好校准"。若 SFT 后混淆对准确率已达 95%+ 且校准合格，RL 的边际收益低于复杂度成本——**停**。RL 不是必经之路，是可选项。

---

## 8. 风险与注意事项

| 风险 | 缓解 |
|---|---|
| **灾难性遗忘**（通用能力退化→JSON 格式漂移、指令跟随变差） | LoRA rank 不要过大（32 够用）；训练数据中混入 5~10% 通用指令数据（如 alpaca 采样）；每轮跑第 5.3 节 JSON 合法率 |
| **数据泄漏**（评估集与训练集同 seed 生成 → 虚高） | 评估集用**不相交的 seed 区间**生成，脚本中显式断言 |
| **训练/推理分布漂移**（提示模板改了忘改数据，或 stats 键名变动） | 数据生成直接 import 线上 `build_report`/`_build_prompt`，不手抄模板；线上模板改动时重新生成数据 |
| **过拟合仿真分布**（真实工厂数据分布不同） | 当前全量数据来自物理仿真——大赛演示完全够用；部署到真实产线时需按 9.2 用真实反馈持续混合训练 |
| **过度自信**（SFT 后 confidence 普遍 0.95+，门控失效） | 数据侧按 2.4 分层打标；评估侧监控判错样本平均置信度 |
| **证书/密钥安全** | 训练与部署涉及的 `.env`、HF token 均不入库（`.gitignore` 已排除 `.env`，保持） |
| **框架版本地狱** | SFT 全部在独立 `sft` conda 环境进行，`mff_agent` 环境零改动 |

---

## 9. 与项目闭环（持续优化智能体）的衔接

### 9.1 先修两个小缺口（SFT 前置工作）

1. **`feedback.py` 的 `diagnosis` 快照当前为空**（`feedback.jsonl` 中 `"diagnosis": {}`）——确认调用 `archive()` 处传入了 `result.to_dict()` + LLM 原始输出 `raw`。没有快照就没有 DPO 偏好对来源。
2. `should_retrain(min_samples=5)` 目前只置标记。建议接到本方案：标记触发时调用 `tools/gen_sft_data.py` 增量生成 + 真实样本混合 → 重训 LoRA → 跑 5.3 回归 → 达标才切端点，形成"运行一次、进步一点"的完整叙事（正是四大智能体中 ④ 的设计意图）。

### 9.2 数据飞轮

```
线上诊断 ─► 运维反馈(actual_root_cause) ─► feedback.jsonl(含诊断快照)
     ▲                                          │
     │                                   真实样本 + 仿真样本混合
SFT模型定期再训 ◄── 离线评估门控 ◄── 增量SFT数据 ◄──┘
```

真实样本入库后**优先级高于仿真样本**（真实分布），混合比例建议真实:仿真 ≥ 1:3 起步。

---

## 10. 附录：命令速查

```bash
# ── 数据 ──
conda run -n mff_agent python tools/gen_sft_data.py     # 生成 data/sft/{train,val}.jsonl
conda run -n mff_agent python tools/audit_sft_data.py   # 自动审计 5% 抽检前置项

# ── 训练（sft 环境，LLaMA-Factory 目录下）──
FORCE_TORCHRUN=1 llamafactory-cli train mff_lora_sft.yaml
llamafactory-cli export --model_name_or_path <底座> \
  --adapter_name_or_path saves/mff-qwen-lora --template qwen3 \
  --export_dir /data/models/mff-qwen-sft-merged

# ── 评估（先起新模型服务）──
vllm serve /data/models/mff-qwen-sft-merged --served-model-name mff-qwen-sft --port 3761 --max-model-len 8192
conda run -n mff_agent python tests/eval_sft_model.py --url http://localhost:3761/v1

# ── （可选）INT4 量化部署，对齐线上小模型规格，量化后必须复跑评估 ──
# 见 6.5：export --export_quantization_bit 4 → vllm serve mff-qwen-sft-int4

# ── 上线切换（改 .env 后重启 uvicorn；big 链路指向新端点）──
# url: http://<host>:3761/v1
# big_model_name: mff-qwen-sft   # .env 的模型名需与 --served-model-name 一致
uvicorn server.api:app --host 0.0.0.0 --port 8000
```

**验收一览**：混淆对 ≥90% / JSON 合法率 ≥99% / 证据忠实 ≥98% / 防幻觉通过率 100% / 判错置信度 <0.8 / 时延 ≤5s —— 全绿再切端点。
