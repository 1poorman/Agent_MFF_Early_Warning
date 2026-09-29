# MiniCPM5-2B 中频炉根因诊断后训练 · 全参 SFT 与 LoRA 完整记录

> 本文档记录本次后训练的**完整过程**：初始规划、环境与硬件、分阶段执行
> （含**显存/耗时/指标**）、遇到的问题与解决对策、最终结论。
> 配套文档：`docs/SFT_BLUEPRINT.md`（蓝图）· `docs/SFT_MILESTONES.md`（里程碑）·
> `docs/SFT_TEST_PLAN.md`（验收）· `design/SFT_BASELINE_REPORT.md`（MS1）·
> `design/SFT_EVAL_REPORT.md`（MS5/L3）。
> 全参历史记录：2026-09-09 ~ 2026-09-13；LoRA 补充训练：2026-09-23；修订：2026-09-24。
> 本次 LoRA 的数据集、参数、命令、时长、显存与对照评估详见
> [LoRA 与全参 SFT 对照报告](SFT_LORA_COMPARISON.md)；原始历史记录保留在下文。

---

## 一、项目目标与初始规划

### 1.1 目标

对 L3 根因诊断链路引入 **MiniCPM5-2B 后训练**，实现三重目标：

1. **知识内化**：把水冷系统故障鉴别规则从 `_build_prompt` 提示工程迁入模型权重；
2. **推理增强**：在非思考模式（低时延）下提升多跳鉴别正确率，减少对代码仲裁的依赖；
3. **自主可控**：摆脱内网 27B 依赖，模型-数据-训练全链路本地闭环。

### 1.2 核心指标与验收口径

| 指标 | 目标 |
|---|---|
| 级联根因准确率 | 关仲裁 ≥90%（含仲裁 100%，以 27B 为基准） |
| 混淆对准确率（关仲裁） | ≥90% |
| JSON 合法率 / 证据忠实率 / 防幻觉通过率 | ≥99% / ≥98% / 100%（硬门槛） |
| 时延 | 2B 前置 ≤5s；27B 兜底 ≤10s |
| 级联升级率 | ≤40% |

### 1.3 技术选型（ADR 摘要）

| 决策 | 选择 | 理由 |
|---|---|---|
| 底座 | MiniCPM5-2B（Apache-2.0，LlamaForCausalLM） | 可商用、标准架构、2B SOTA |
| 训练方式 | 首轮全参 SFT；2026-09-23 补做 LoRA 对照 | 单卡训练，对比行为效果、时间、显存和产物体积（见第九节） |
| 部署 | **级联**（2B 前置 + 27B 兜底） | 零回归风险；80% 明确场景 1s 级响应 |
| 环境 | 训练/服务/**量化**多环境隔离 | 框架 transformers 版本互相冲突 |
| 权重来源 | ModelScope | 本机 huggingface.co 不可达 |
| 数据 | 仿真自生成（4 类故障物理机理自带金标签） | 零标注、与线上 report 严格同构 |
| RL（DPO/GRPO） | 列为可选阶段 | ≥500 偏好对前置未满足 |

---

## 二、硬件与环境

### 2.1 硬件

| 项 | 值 |
|---|---|
| GPU | **4× RTX 3090 24GB**（共享机：GPU2/3 常被 27B 主服务占用；MOS 期间 GPU1 常空） |
| 驱动 / CUDA | 580.126.09 / CUDA 13.0（满足 vLLM ≥0.21） |
| 磁盘 | 根分区约 2.1T 可用（训练/权重/缓存充足） |
| 网络 | huggingface.co ❌ 不可达；GitHub / ModelScope / PyPI（清华镜像）✅ |

### 2.2 Conda 环境（四环境隔离）

| 环境 | Python | 关键依赖 | 用途 |
|---|---|---|---|
| `mff_sft` | 3.11 | LLaMA-Factory 0.9.6.dev0（**源码版**）、transformers 5.8.0、torch 2.14.0+cu130 | 全参 SFT 训练 |
| `mff_sft_serve` | 3.11 | vLLM **0.28.0**、transformers 5.16.1、torch 2.13.0+cu130、compressed-tensors 0.17 | 推理服务 |
| `mff_agent` | 3.10 | 主项目依赖（openai 等） | 评估/回归/主项目 |
| `mff_sft_quant` | 3.11 | **llmcompressor 0.13.0**、compressed-tensors 0.18.0、torch 2.13.0+cu130、transformers 5.14.1 | INT4 量化（L3） |

> **为何训练要源码版 LLaMA-Factory**：`template: minicpm5` 仅在源码版注册，
> pip 版会报 "Template does not exist"。源码版要求 Python ≥3.11。

---

## 三、总体流程（MS0 → MS8）

```
MS0 环境/权重 ──► MS1 零成本基线(决策门) ──► MS2 数据构造+审计 ──► MS3 评估基建
                                                                      │
                                                                      ▼
MS8 数据飞轮 ◄── MS7 DPO/GRPO(可选) ◄── MS6 级联部署 ◄── MS5 离线评估+INT4 ◄── MS4 全参 SFT
```

| 里程碑 | 状态 | 完成日期 |
|---|---|---|
| MS0 前期准备 | ✅ | 2026-09-09 |
| MS1 零成本基线（决策门） | ✅ 通过 | 2026-09-09 |
| MS2 数据构造与审计 | ✅ 审计全绿 | 2026-09-09 |
| MS3 评估基建 | ✅ | 2026-09-10 |
| MS4 全参 SFT 训练 | ✅ | 2026-09-10 |
| MS5 离线评估 + INT4 量化 | ✅（量化判定弃用） | 2026-09-10 / 09-13 |
| MS6 级联部署与回归 | ✅ 线上验收 | 2026-09-12 |
| MS7 DPO/GRPO（可选） | 🔒 阻塞（偏好对 0/500） | — |
| MS8 数据飞轮 | 🔄 脚手架就绪 | 2026-09-13 |
| 补充：LoRA 训练与全参对照 | ✅ 各三次复测完成 | 2026-09-23～24 |
| 补充：合成偏好 DPO / 规则奖励 GRPO | ✅ 训练、主集三次及压力集评测完成；未观察到效果增益 | 2026-09-28 |

---

## 四、分阶段执行记录

### MS0 前期准备（2026-09-09）

**动作**：创建双环境 + ModelScope 下载底座 + vLLM 探活 + `enable_thinking` 冒烟 + 模板比对。

- **显存**：vLLM 服务 `--gpu-memory-utilization 0.5`（约 11.8GB 上限），底座权重约 5GB。
- **耗时**：环境与下载约 0.5 天（脚本自动）。
- **产物**：`~/models/MiniCPM5-2B`（4.7GB），端口 3761。

### MS1 零成本基线试验（决策门，2026-09-09）

**目的**：不训练，测原模型在窄域鉴别上的上限。

- **口径**：镜像线上实时流路径（去 L1/L2 注入），原样 `_build_prompt`（含判定规则）。
- **指标（3 次复跑）**：**关仲裁 64.0%±0.9%**、**含仲裁 81.4%**、JSON 100%、时延 1.2s。
- **失败模式**：气蚀 0% / 泄漏 2.8%，**全误判为堵塞**——SFT 靶点明确。
- **决策**：≥60% → **通过**，进入 MS2。
- **显存/耗时**：仅推理，GPU0；评估约 10 分钟。

> **关键排障**：v1/v2 批量路径把 L1 全量注入 prompt → prompt 爆炸；v3 改为镜像实时流路径
> （不注入 L1/L2），**这才是线上真实分布**；vLLM 上下文扩到 32768。

### MS2 数据构造与审计（2026-09-09）

**数据规模**：train **2337** / val **259** / eval **200**（eval seed 与训练不相交）。

| 类别 | train | 说明（置信度） |
|---|---|---|
| clear | 1082 | 4 类故障清晰签名（conf 0.88） |
| boundary | 546 | 混淆带：泄漏湿度 60~70%RH / 气蚀 std 2~4.5kPa / 结垢 PQ 7~14% / 堵塞流量 6.0~7.2（conf 0.78） |
| transition | 258 | 快工况循环，输出"无故障（工况切换正常波动）"（conf 0.90） |
| composite | 185 | 复合故障主次因（conf 0.85） |
| misleading | 181 | 工单指向错误根因，金标签按实时 stats |
| antihall | 85 | 特征物理违规 →"数据质量异常（建议人工核查）"（conf 0.50） |

- **输入**：退化版实时流提示（删"判定规则"节与注意项，schema 扩展无故障/数据异常）。
- **输出**：CoT 引用真实 stats 数值 + 金标签 JSON。
- **审计**：七项（JSON / GOLD / SOP / FAITH / QUOTA / LEAK / **GVIO**）全绿 + 抽检通过。
- **显存**：CPU 生成，无 GPU。
- **耗时**：生成约 1 天（含三轮迭代）。
- **产物**：`data/sft/{train,val,eval}.jsonl`。

### MS3 评估基建（2026-09-10）

- **`tests/eval_sft_model.py`**：逐条请求任意 OpenAI 兼容端点，一键输出 L2 九指标全表 +
  分层/混淆矩阵 + 硬门槛判定；支持 `--arbitration on|off`。
- **基线复算校验**：原模型 × eval 集（关仲裁）故障子集 **20.6%**、混淆对 **30.0%**、
  JSON 100%、证据忠实 98.4%、防幻觉 100%、P95 1.42s。
- **显存/耗时**：全 CPU 跑评估栈（GPU0 留给 vLLM）；200 条约 5 分钟。

### MS4 全参 SFT 训练（2026-09-10）

**配置**（`tools/mff_full_sft.yaml`）：

| 项 | 值 |
|---|---|
| 底座 | MiniCPM5-2B（LlamaForCausalLM，2.52B） |
| 方式 | `finetuning_type: full`（全参） |
| 模板 | `template: minicpm5` + `enable_thinking: false` |
| 数据 | train 2337 / val 259 |
| 超参 | lr 1e-5 / cosine / warmup 0.1 / 3 epochs / 有效 batch 16 / `pure_bf16` + `paged_adamw_8bit` / cutoff 4096 |

**显存**：

- **单卡 RTX 3090（GPU1），峰值 21.2GB / 24GB**。
- 分解：params(bf16) ≈5GB + grads ≈5GB + 8bit 优化器态 ≈2.5GB + 激活/其他 ≈8GB。
- ⚠️ 若不用 `pure_bf16`，LLaMA-Factory 默认把可训练参数**升 fp32**（params 10G + grads 10G）→
  首步即 OOM（实测 23.5GB 爆）。

**耗时与曲线（v3，最终）**：

| 项 | 值 |
|---|---|
| 训练时长 | **3604s（1:00:04）** / 441 步 / 3 epochs |
| 吞吐 | 1.945 samples/s，0.122 steps/s |
| train_loss | **0.1225** |
| total_flos | 9.82218×10¹⁶ FLOPs（`train_results.json` 中的 Trainer 估算） |
| val loss 轨迹（事后补算） | base 2.0774 → ckpt200 0.0023 → ckpt400/441 **0.0004**（单调无回升） |

> 三次迭代同配置（v1/v2/v3）：59 / 60 / **60 min**，train_loss 0.1239 / 0.1266 / **0.1225**。

**模板 sanity check**：3 条样本 × 两种 `enable_thinking`，LLaMA-Factory 渲染与模型自带
`chat_template.jinja` **逐 token 一致**；`enable_thinking=false` 与线上 vLLM 路径完全相同。

**产物**：`~/models/mff-sft-minicpm5-2b`（4.7GB）+ 训练输出 `saves/mff-sft-full/`。

### MS5 离线评估（2026-09-10）与 INT4 量化（2026-09-13）

#### 5.1 三轮迭代的因果（核心工程结论）

| | 数据生成器 bug | 置信度标注 | 关仲裁准确率 | 拒诊率 | 校准 gap | 判定 |
|---|---|---|---|---|---|---|
| v1 | ❌ 存在（28 条错标签） | 按类别常数 | 99.4% | 50%\* | −0.045 | 有 bug |
| v2 | ✅ 已修 | 随证据余量梯度 | **97.2%** ⬇ | 80% | **−0.088** ⬇ | **净负面** |
| **v3** | ✅ 已修 | **回退常数** | **99.4%** ⬆ | **90%** ⬆ | −0.043 | **最优** |

> **三轮数据迭代结论**：v1→v3 的拒诊改善与修复标签 bug 一致；本轮改置信度标注未带来收益。
> 这不表示原模型→SFT 的全部收益来自修 bug。\*v1 拒诊 50% 系在被污染 eval 集测得，真实值 71%。

#### 5.2 v3 最终指标（关仲裁，200 条）

| 指标 | 基线（原模型） | **v3** | 门槛 |
|---|---|---|---|
| 根因准确率（故障子集 n=170） | 20.6% | **99.4%** | ≥90% ✅ |
| 混淆对准确率（n=60） | 30.0% | **100%** | ≥90% ✅ |
| JSON 合法率 | 100% | **100%** | ≥99% ✅ |
| 证据忠实率 | 98.9% | **100%** | ≥98% ✅ |
| 防幻觉通过率 | 100% | **100%** | 100% ✅ |
| 无故障误报率（n=20） | 100% | **0%** | ≤5% ✅ |
| antihall 拒诊率（n=10） | 0% | **90%** | — |
| 时延 P95 | 1.34s | **1.75s** | ≤5s ✅ |
| 含仲裁准确率 | 80.6% | 82.9% | — |
| 置信度校准 gap | — | −0.043（n_wrong=2） | >0.15 ❌ |

- **评估耗时**：bf16 三次复跑各 **282~288s**（200 条）。
- **校准项**：经证据判定为**判据不适用**（判对集合必然含按设计低置信的类 antihall 0.50/boundary 0.78；
  `n_wrong=2` 无统计意义）。用户决策 **B**：不确定性表达推迟到 MS7，MS6 不依赖模型置信度。

#### 5.3 INT4 量化（L3 决策门，2026-09-13）

**方法**：`autoawq` 与 transformers 5.8 不兼容 → 改用 vLLM 官方 **llmcompressor**
（compressed-tensors W4A16），独立环境 `mff_sft_quant`。校准语料=训练集退化版 prompt 128 条。

| 指标 | bf16(v3) | INT4 RTN | INT4 GPTQ | 门槛 |
|---|---|---|---|---|
| 根因准确率 | **99.4%** | 0.0% | 58.2% | ≥90% |
| 混淆对准确率 | **100%** | 0.0% | 55.0% | ≥90% |
| 无故障误报率 | **0%** | 0% | 95.0% | ≤5% |
| JSON / 证据忠实 / 防幻觉 | 100/100/100 | 100/100/100 | 99.0/99.5/100 | 硬门槛 |
| 时延 P95 | 1.75s | 0.74s | 0.95s | ≤5s |

- **显存**：量化在 GPU1 进行，模型 bf16 加载约 5GB + 激活，峰值 < 12GB；INT4 服务权重约 2GB。
- **耗时**：RTN 约 1 分钟；GPTQ 带校准约 2~3 分钟（产品 2.0GB）。
- **结论**：混淆对掉 **45pp ≫ 3pp** → 按 6.5.5 决策树 **放弃量化、bf16 部署**。
  实测说明本次量化配置明显退化；模板记忆、训练数据重复及量化实现的影响尚未通过消融分别验证，
  不能只凭 val loss ~0.0004 认定唯一根因。
- GPTQ 产物 `~/models/mff-sft-minicpm5-2b-int4-gptq` 留档为**负面证据**。

### MS6 级联部署与线上回归（2026-09-12）

**架构**（蓝图 4.2 + 决策 B）：

```
诊断请求 ──► 2B SFT 前置（GPU0:3761，关仲裁，信任原始输出）
              ├─ JSON可解析 且 防幻觉通过 且 端点正常 且 与统计预判一致 ──► 直出（front_2b）
              └─ 任一可验证信号不过 ──► 升级 27B（含仲裁+重试，非思考）──► 仍失败 ──► manual_required
```

- **升级判据不含模型自报置信度**（决策 B）；`CascadeGate` 只吃 `parse_ok/check_passed/
  endpoint_error/expected_candidates`。
- 改动点：`llm_client.py`（front 端点）、`confidence.py`（CascadeGate）、
  `root_cause.py`（cascade）、`service.py`（装配+快照缓存）、`config`（cascade_enabled）、
  `action/feedback.py`（prompt 快照 + DPO 偏好对）。

**线上回归结果**：

| 项 | 结果 |
|---|---|
| HTTP endpoint | **18/18 通过** |
| workflow 4 类故障 ×3 复跑 | 根因 **12/12** |
| 2B 前置路径时延 | **1.1~1.4s**（pump_cavitation / pipe_leak / scale_buildup） |
| 27B 兜底路径时延 | **4.2~4.6s**（filter_clog，因 2B 与统计预判不一致升级） |
| 回归集 | 110 条，基线 108/110，**零退化** |

### MS8 数据飞轮（2026-09-13，脚手架）

- **`tools/flywheel.py`**：`status` / `build`（真实:仿真 ≥1:3 + 注册 dataset_info）/
  `train`（生成增量 yaml + llamafactory-cli）/ `eval`（L2 全表+回归）/ `promote`（**仅打印**人工灰度指令）。
- **幂等演练**：`--simulate-feedback N` 注入合成真实样本；`tests/test_flywheel.py` 4 项单测全绿。
- **冒烟**：GPU1 `--max-steps 5` 训练成功（增量数据 5 真实+30 仿真），**"触发→构建→训练"链路打通**。

---

## 五、产物与数据清单

| 类别 | 路径 | 大小 |
|---|---|---|
| 底座权重 | `~/models/MiniCPM5-2B` | 4.7GB |
| **SFT 产物（生产）** | `~/models/mff-sft-minicpm5-2b` | 4.7GB |
| 训练输出 | `saves/mff-sft-full/` | ~38GB（含 checkpoint） |
| INT4（留档，弃用） | `~/models/mff-sft-minicpm5-2b-int4-gptq` | 2.0GB |
| 数据 | `data/sft/{train,val,eval}.jsonl` | 8.2/0.9/0.7MB |
| 回归集 | `data/sft/regression.jsonl` + `.baseline.json` | 110 条 |
| 脚本 | `tools/`：`sft_prep.sh` `gen_sft_data.py` `audit_sft_data.py` `ms1_baseline.py` `mff_full_sft.yaml` `mff_export.yaml` `check_lf_template.py` `eval_loss.py` `verify_export.py` `serve_sft.sh` `serve_sft_supervised.sh` `build_regression_set.py` `mff_quantize.py` `flywheel.py` | — |
| 评估 | `tests/eval_sft_model.py` `run_regression.py` `test_cascade.py` `test_flywheel.py` | — |

---

## 六、问题与对策（核心）

> 以下均为本次实际踩坑，按"现象 → 根因 → 对策"记录。

### 6.1 环境与训练

| # | 问题现象 | 根因 | 对策 |
|---|---|---|---|
| 1 | 训练报 `KeyError: 'from'` | sharegpt 默认 tags 是 `from`/`value` | `dataset_info.json` 显式 `columns.messages` + `tags` |
| 2 | "Cannot find valid samples" | 默认 `user_tag/assistant_tag` 为 `human`/`gpt` | 显式给 `role_tag/content_tag/user_tag/assistant_tag/system_tag` |
| 3 | 全参微调首步 OOM（23.5GB） | LLaMA-Factory 默认把可训练参数升 fp32（params 10G+grads 10G） | `pure_bf16: true` + `paged_adamw_8bit`（峰值降到 ~21.2GB） |
| 4 | 训练时 val loss 不触发 | transformers 5.x 需 `eval_strategy`，`eval_steps` 不够 | 训练后 `tools/eval_loss.py` 补算 val loss 轨迹 |
| 5 | `minicpm5_nothink` 报模板不存在 | LLaMA-Factory 未注册该变体 | 关思考走 DataArgument `enable_thinking: false`（与自带模板天然一致） |
| 6 | 导入路径 `llamafactory.extras.template` 报错 | 旧版文档路径过时 | 正确路径 `llamafactory.data.template` |

### 6.2 数据生成

| # | 问题现象 | 根因 | 对策 |
|---|---|---|---|
| 7 | transition 类样本 = 0 | 工况切换点与 600s 对齐网格**恰好错位** | 快工况循环调度（2700s/周期）+ 步长 60 + 尾 120s 判定 |
| 8 | boundary 仅 160/600 | severity 网格 0.4 起步太粗 + 泄漏带误覆盖铁证区 | 网格加密 0.12~1.0 + 泄漏带改 60~70 |
| 9 | **antihall 毒样本（GVIO）**：输入无越界却标"数据质量异常" | 对**已渲染 prompt** 做正则手术，`'( "出水温度": )'` 要求键前有空格 → features 首键永不匹配 | 改走 `build_sample(corrupt=...)` **结构化注入** + 显式 `reason` + 出口自检 `antihall_grounded` + 审计**新增第 7 项 GVIO**（旧数据精确命中 28 条，新数据 0 条） |

### 6.3 评估与决策

| # | 问题现象 | 根因 | 对策 |
|---|---|---|---|
| 10 | MS1 批量路径 prompt 爆炸 | 全量注入 L1/L2 | v3 镜像实时流路径（不注入 L1/L2）= 线上真实分布 |
| 11 | 置信度校准 gap 始终不达标 | 判据与"99% 准确率记忆型 SFT"**结构性冲突**（判对含低置信类、n_wrong=2） | 判定"判据不适用"，决策 B：推迟到 MS7，MS6 不依赖模型置信度 |
| 12 | v2 改置信度标注后准确率反降 | 让模型"更谦虚"同时压低了判对样本置信度 | 回退常数标注（v3），收益全来自修 bug |
| 13 | INT4 RTN 全输出"无故障"、GPTQ 掉 45pp | 记忆型过拟合权重对 4bit 扰动极敏感 | 按 6.5.5 决策树**放弃量化、bf16 部署** |
| 14 | `autoawq` 装不上 | 0.2.9 要求 transformers≤4.x，本机 5.8 | 改用 vLLM 官方 `llmcompressor`，建独立环境 `mff_sft_quant` |
| 15 | `llmcompressor` 导入/调用报错 | API 变更为 `from llmcompressor import oneshot`；`dataset` 需 HF Dataset 而非 list | 修正导入；`datasets.Dataset.from_list([{"text":...}])` |

### 6.4 级联与线上

| # | 问题现象 | 根因 | 对策 |
|---|---|---|---|
| 16 | 含仲裁反而改错 SFT 正确答案（99.4%→82.9%） | 仲裁层是为弱基线设计的兜底 | 级联前置 **`arbitrate=False`** |
| 17 | `filter_clog` 在完整 workflow prompt 下被 2B 误判、且防幻觉仍通过 → 不升级 | 防幻觉只校验 schema/物理，不校验**与统计预判的一致性** | `CascadeGate` 增**统计预判一致性判据**（不依赖置信度） |
| 18 | 27B 兜底一次耗时数十秒，违反时延 | 完整 L1 告警注入 + 27B 思考链 | `_build_prompt` **L1 截最近 10 条**；级联兜底强制**非思考** + `max_tokens=512`；前置 `max_tokens=1000` |
| 19 | 线上回归脚本把结果全判失败 | `BASE` 硬编码远端 `124.65.133.158:9000`；故障名用了 API 不支持的 `cavitation`/`scale` | 本机临时重定向 `127.0.0.1:8000`；改用 **`pump_cavitation`/`scale_buildup`**（并修正测试方案文档） |
| 20 | 反馈快照 `diagnosis:{}` 为空，DPO 无源 | `ContinuousOptimizerAgent.feedback` 构造 snapshot 却**未传给** `submit_feedback`；`to_dict()` 无 prompt | `DiagnosisResult.prompt` + `training_snapshot()`；`submit_feedback` 收快照并按 order_id 回填缓存；缺 prompt 告警 + `build_preference_pairs()` |
| 21 | 诊断异常被静默吞掉 | `agents.py:242`/`service.py:428` 裸 `except Exception` | 改为 `logger.exception` 显式记录（`diagnose` 已对端点异常优雅降级） |

### 6.5 服务运维（澄清误判）

> ⚠️ **更正**：过程中曾把 `EngineDeadError` 误判为"vLLM 自发崩溃/稳定性风险"。复核日志后确认：
> 该错误出现在 `send sigterm to process EngineCore` **之后**，是**手动 `kill`（SIGTERM）
> 触发的正常关闭收尾日志**（vLLM 异步 handler 对主动关闭的 noisy 报错），**并非自发崩溃**；
> 服务消失均为人工停服。`ptxas [Errno 14] Bad address` 是编译缓存加载失败后**自动重编译**的
> **非致命告警**，不影响可用性。故 **vLLM 稳定性无实质问题**。

| # | 现象 | 说明 | 处置 |
|---|---|---|---|
| 22 | 日志出现 `EngineDeadError` / `ptxas Bad address` | 前者=手动 SIGTERM 关闭时的收尾日志；后者=编译缓存加载失败后自动重编译的非致命告警 | **无需修复**；仅澄清日志含义，避免误判 |
| 23 | `--enforce-eager` 使 2B 时延增至 3.8~6.1s | 禁用 torch.compile/cudagraphs（约 4x） | **默认关闭**（编译模式 ~0.99s），仅在确有需要时作降级选项 |
| 24 | 首次启动需数分钟才监听端口 | FlashInfer JIT 编译（一次性，后续走缓存） | 属正常冷启动；等待就绪探活即可 |

**可选运维工具**（非故障修复，便于常驻）：
- `tools/serve_sft.sh` 增 `EAGER`（降级选项）/ `CLEAN_CACHE`（缓存异常时清理）开关，默认编译模式；
- `tools/serve_sft_supervised.sh`：进程退出后自动重启 + 退避，便于生产常驻（可选）。

---

## 七、最终指标汇总（北极星）

| 维度 | 指标 | 结果 | 达标 |
|---|---|---|---|
| 准确率 | 关仲裁根因（200 条中的故障子集 n=170） | 20.6% → **99.4%** | ✅ |
| | 混淆对 | 30.0% → **100%** | ✅ |
| | 含仲裁 | **82.9%** | — |
| 硬门槛 | JSON / 证据忠实 / 防幻觉 | **100% / 100% / 100%** | ✅ |
| 安全 | 无故障误报率 | 100% → **0%** | ✅ |
| | 数据质量拒诊率 | **90%** | — |
| 时延 | 2B 前置 / 27B 兜底 | **1.1~1.4s / 4.2~4.6s** | ✅ |
| 级联 | 升级率 / endpoint / 4 故障×3 | ≤40% / **18/18** / **12/12** | ✅ |
| 回归 | 回归集零退化 | 108/110 基线，**无退化** | ✅ |
| 训练 | 峰值显存 / 时长 / 最终 loss | **21.2GB / 1:00:04 / 0.1225** | — |
| 量化 | INT4 vs bf16 混淆对差 | 掉 45pp → **弃量化** | 决策完成 |

---

## 八、结论与后续

### 结论

1. **知识内化成功**：2B SFT 关仲裁根因 20.6%→99.4%，混淆对 30%→100%，硬门槛全绿。
2. **v1→v3 的数据迭代收益主要来自修复标签 bug**（GVIO）；所试置信度改造未带来收益。
3. **级联部署可行**：升级不依赖模型自报置信度；明确场景 1s 级、兜底 ≤10s。
4. **已测试的量化配置未达标**：本次 RTN/GPTQ 退化明显，采用 **bf16 部署**；不外推到所有量化方案。
5. **工程收益显著**：修了 3 类隐藏缺陷（数据生成 GVIO bug、反馈快照缺 prompt、静默吞异常），
   并统一了线上时延契约（2B ≤5s / 27B 兜底 ≤10s）。

### 后续（阻塞项）

| 项 | 前置 | 状态 |
|---|---|---|
| MS7 DPO/GRPO | ≥500 偏好对（当前 0，历史反馈缺 prompt） | 🔒 阻塞 |
| MS8 全量闭环 | 真实反馈积累；build/train 已冒烟，eval/promote 已接线 | 🔄 半就绪 |
| 生产常驻 | 主服务 8000 正式部署；SFT 前置可用 `serve_sft_supervised.sh`（可选） | 待办 |

---

## 九、LoRA 补充训练与全参对照（2026-09-23～24）

详细数据集说明、环境、完整参数、复现命令及行为评估见
[MiniCPM5-2B LoRA 与全参 SFT 对照报告](SFT_LORA_COMPARISON.md)。
本次从原始 MiniCPM5-2B 训练 LoRA，未接续全参权重。

### 9.1 数据和对照口径

沿用现存 v3 的 `data/sft/{train,val,eval}.jsonl`，规模 **2337 / 259 / 200**。
数据全部来自物理仿真：四类故障、工况切换、复合故障、误导工单和物理违规拒诊。
输入删除显式判定规则，输出为引用真实统计值的推理链和诊断 JSON；仅 assistant 回答计算 loss。
完整序列平均约 1017～1019 token、最大 1080，cutoff=4096 下没有样本截断。

七项审计通过；全量 **2796 条**原生模板与全参 `minicpm5` 模板的 token 和 loss 掩码一致。
本次新增审计发现 train/val 共享 119 个唯一输入，涉及 **128/259 条验证样本**。
train/val 与 eval 的 seed、完整输入都无重叠；eval 内部仍有 18 条重复输入。
因此近零 val loss 不能证明泛化，行为对照以独立 seed 的 eval 为主，并保留仿真分布局限。
历史全参训练没有保存数据 SHA256，本次复测则对两种权重使用同一份带指纹的评估文件。

### 9.2 参数、时间与显存

| 项目 | 全参 SFT v3 | 本次 LoRA |
|---|---|---|
| 训练方式/底座 | 全参 / MiniCPM5-2B | LoRA / 同一底座，BF16，非 QLoRA |
| 可训练参数 | 2,516,756,480 | **50,233,344**（含 adapter 总参数的 1.9569%） |
| LoRA 参数 | — | rank 32 / alpha 64 / dropout 0.05；294 个线性层 |
| 精度 | pure_bf16 | 冻结底座 BF16；adapter FP32；BF16 autocast |
| 学习率 | 1e-5 | 1e-4 |
| 共同参数 | 单卡 RTX 3090，3 epochs，batch 2 × 累积 8，有效 batch 16，cosine，warmup 0.1，paged_adamw_8bit，seed 42 | 同左 |
| 训练步数 | 441 | 441 |
| Trainer 训练时长 | 3604.12 s（60 分 04 秒） | **3849.92 s（64 分 10 秒）** |
| 进程端到端时长 | 未留同口径测量 | 3947.12 s（65 分 47 秒） |
| 全程平均 train_loss | 0.1225035 | 0.0880876 |
| val loss | 约 0.0004，历史逐 token NLL | 0.0005608，Trainer 汇总，口径不完全相同 |
| 显存峰值 | 历史记载约 21.2 GB，缺原始采样/单位定义 | 活跃张量 10.30 GiB；缓存保留 21.00 GiB；进程 **21.31 GiB**；整卡 **22.31 GiB** |
| 训练输出（含 checkpoint） | 约 40.30 GB | **约 1.15 GB** |
| 最终权重 | 完整权重约 4.69 GiB | adapter **191.70 MiB**；合并后完整权重仍约 4.69 GiB |

LoRA 训练内执行了 3 次验证，合计 104.03 s，全参历史训练没有执行验证。
本轮 LoRA 的 Trainer 时长增加约 6.8%，不能宣称训练提速；主要明确收益是可训练参数和产物体积减少。
进程显存曾在首次验证前约为 13.0 GiB，但随缓存保留升高，完整运行峰值为 21.31 GiB。
两种方式的历史/当前显存口径不同，不能给出严格的显存节省百分比。

训练与显存原始证据位于 `logs/lora_minicpm5_20260923/train/`，3489 次显存采样全部成功。
LoRA adapter 位于 `saves/mff-lora-minicpm5-2b-20260923/`，
合并权重位于 `saves/mff-lora-minicpm5-2b-20260923-merged/`。

### 9.3 实际训练指令

```bash
/home/huachenghao/.conda/envs/mff_agent/bin/python tools/measure_run.py \
  --gpu 1 --out-dir logs/lora_minicpm5_20260923/train -- \
  /home/huachenghao/.conda/envs/mff_sft/bin/python tools/train_lora.py \
  --config tools/mff_lora_minicpm5.yaml \
  --output-dir saves/mff-lora-minicpm5-2b-20260923
```

新脚本拒绝覆盖已有实验目录，复跑时需选新目录。
数据检查、全参历史命令、CPU 合并及同配置评估命令见对照报告第 6 节。

### 9.4 同配置行为复测结论

2026-09-24 在 GPU0 上依次运行全参与 LoRA 的 BF16 权重，各 **3 次 × 200 条**。
两者使用同一份评估数据、同一个 vLLM 环境、temperature=0.1、top_p=0.95、
max_tokens=4000、关闭思考及仲裁，无级联兜底；各自三次的逐条根因预测完全一致。

| 指标 | 全参 v3（本次复测） | LoRA |
|---|---:|---:|
| 故障根因准确率（n=170） | **169/170 = 99.41%** | **169/170 = 99.41%** |
| 混淆对（n=60） | 100% | 100% |
| 总体准确率（n=200） | **198/200 = 99.0%** | **197/200 = 98.5%** |
| 数据异常拒诊（n=10） | **9/10 = 90%** | **8/10 = 80%** |
| JSON / 证据忠实 / 防幻觉 | 100% / 100% / 100% | 100% / 100% / 100% |
| 正常工况误报（n=20） | 0% | 0% |
| 三次 P95 的平均 | 1.779 s | 1.782 s |
| 200 条平均评估耗时 | 287.80 s | 288.07 s |
| 置信度 gap | −0.04343 | −0.05008 |

每次两模型仅在 `idx=123`（文件第 124 行）不同：输入流量 80 L/s 超过项目校验上限 50 L/s，
全参正确拒诊，LoRA 错判为无故障。两者还共同错判 `idx=90` 的另一条量程异常和
`idx=124` 的复合故障主根因。防幻觉 100% 的分母排除了 antihall，不能替代拒诊指标。

将评估输入去重为 182 条后，全参 180/182（98.90%）、LoRA 179/182（98.35%），
故障子集均为 151/152（99.34%），对照结论不变。

**结论**：本配置下，LoRA 的故障鉴别与全参持平，训练产物显著缩小，数据质量拒诊少对一条，
训练与推理没有实测提速。建议当前优先保留全参方案，LoRA 作为存储成本较低的备选；
新一轮改进应先处理训练/验证重复输入，再补强量程异常拒诊与真实设备验证。
这些结果只支持当前数据、配置和硬件，不外推为所有 LoRA/全参方案的优劣结论。

![训练曲线与 LoRA 显存](assets/minicpm5_lora_training.png)

## 十、加入 RL 后训练对照（2026-09-28）

已按用户授权构造偏好数据，并在 LoRA-SFT 合并权重上分别开展 DPO 与 GRPO。
方法定义、数据隔离、配置、实测结果与复现入口见 [SFT / RL 对照报告](SFT_RL_COMPARISON.md)。
上文“0/500 偏好对”的历史门槛针对真实产线反馈；本轮是明确标注来源的合成数据实验，
不声称已有人工反馈 RLHF 或 AI 裁判 RLAIF。

原始配对汇总：`logs/lora_minicpm5_20260923/comparison_summary.json`。

## 十一、八类故障仿真与难例集（2026-09-29）

按“扩展仿真和数据集”范围完成四种新机制：换热器散热衰退、水质恶化、
电气柜散热风机失效、补水阀卡开。`data/sft_hard_v1/` 包含训练 912、验证 200、
评估 268 条样本；使用多根因集合标签，加入复合、弱信号、缺失通道和工况偏移。
全量数据核验、51 个炉次抽样回放和四项回归测试通过，旧 SFT/RL 文件指纹一致。
数据直接保存在 `/models/hch`；此前新增 RL 权重也已迁移 10.18 GiB 并保留路径链接。
本阶段尚未进行八类模型训练，旧四类评测成绩不代表新增故障能力。
详情、标签局限及复现入口见[八类故障数据集报告](HARD_FAULT_DATASET.md)。
