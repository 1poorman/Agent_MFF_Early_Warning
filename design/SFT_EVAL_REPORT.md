# SFT 离线评估报告（MS5）

> **最终产物：v3** → `~/models/mff-sft-minicpm5-2b`（v1/v2 留档为 `-v1` / `-v2`）
> 评估口径：`tests/eval_sft_model.py`，退化版 prompt × `data/sft/eval.jsonl`（200 条，独立 seed）
> 对照基线：`data/sft/baseline_degraded_results_v2.jsonl`（原模型 × 同一 eval 集）
> 日期：2026-09-10

---

## 一、结论速览

| 指标 | 基线（原模型） | v1 | v2 | **v3（最终）** | 门槛 |
|---|---|---|---|---|---|
| 根因准确率（关仲裁·故障子集 n=170） | 20.6% | 99.4% | 97.2% | **99.4%** | ≥90% ✅ |
| 混淆对准确率（n=60） | 30.0% | 100% | 100% | **100%** | ≥90% ✅ |
| JSON 合法率 | 100% | 100% | 100% | **100%** | ≥99%（硬）✅ |
| 证据忠实率 | 98.9% | 100% | 100% | **100%** | ≥98%（硬）✅ |
| 防幻觉通过率 | 100% | 100% | 100% | **100%** | 100%（硬）✅ |
| 无故障误报率（n=20） | **100%** | 0% | 0% | **0%** | ≤5% ✅ |
| antihall 拒诊率（n=10） | 0% | 50%\* | 80% | **90%** | — |
| 时延 P95 | 1.34s | 1.79s | 1.70s | **1.75s** | ≤5s ✅ |
| 含仲裁准确率 | 80.6% | 58.8% | 82.9% | **82.9%** | — |
| 置信度校准 gap | — | −0.045 | −0.088 | **−0.043** | >0.15 ❌ |

\* v1 的 50% 是在**被污染**的 eval 集上测的（10 条里 3 条错标签），真实值为 5/7 = 71%。

### 一句话结论

**MS5 的主指标与三项硬门槛全部达标且大幅超额**：关仲裁根因准确率
**20.6% → 99.4%（+78.8pp）**，混淆对 30% → 100%，无故障误报率 100% → 0%，
拒诊率 0% → 90%。**唯一不达标的是置信度校准，且该判据经证据判定为"用错场景"**（第六节）。

---

## 二、三次迭代的因果（本轮最重要的工程结论）

| | 数据生成器 bug | 置信度标注 | 关仲裁准确率 | 拒诊率 | 校准 gap | 判定 |
|---|---|---|---|---|---|---|
| v1 | ❌ 存在（28 条错标签） | 按类别常数 | 99.4% | 50%\* | −0.045 | 有 bug |
| v2 | ✅ 已修 | 改为"随证据余量梯度" | **97.2%** ⬇ | 80% | **−0.088** ⬇ | **净负面** |
| **v3** | ✅ 已修 | **回退为按类别常数** | **99.4%** ⬆ | **90%** ⬆ | −0.043 | **最优** |

**结论：收益全部来自"修 bug"，"改置信度标注"是纯粹的倒退。**
v2 把 boundary 置信度从常数 0.78 改为梯度 0.71，反而拉低了判对均值，
并让复合故障准确率从 95% 掉到 80%。v3 撤销该改动后，准确率与拒诊率**同时**回到最优。

> **教训（值得记住）**：当准确率已经很高时，任何"让模型在模糊样本上更谦虚"的标注改动，
> 都会同时压低**判对**样本的置信度。校准类指标不能靠改标注去凑，要先问判据本身是否成立。

---

## 三、训练档案（MS4，三次同配置，仅数据不同）

| 项 | 值 |
|---|---|
| 底座 | MiniCPM5-2B（`LlamaForCausalLM`，2.52B） |
| 方式 | LLaMA-Factory `finetuning_type: full`（全参） |
| 模板 | `template: minicpm5` + `enable_thinking: false` |
| 数据 | train 2337 / val 259 |
| 超参 | lr 1e-5 / cosine / warmup 0.1 / 3 epochs / 有效 batch 16 / `pure_bf16` + `paged_adamw_8bit` |
| 硬件 | 单卡 RTX 3090（GPU1），峰值 21.2 GB |
| v1 / v2 / v3 | 59 / 60 / 60 min，441 步，train_loss 0.1239 / 0.1266 / 0.1225 |

**模板 sanity check**（`tools/check_lf_template.py`，MS4 验收 2）：
3 条样本 × 两种 `enable_thinking`，LLaMA-Factory 渲染与模型自带 `chat_template.jinja`
**逐 token 一致**；`enable_thinking=false` 的 prompt 与线上 `LLMClient(enable_thinking=False)`
→ vLLM `chat_template_kwargs` **完全相同**。结论：**无需 `minicpm5_nothink`**（该模板名未注册）。

**验证集 loss 轨迹**（`tools/eval_loss.py`，训练时 `eval_steps` 未触发评估故事后补算）：

| 模型 | v1 | v2 | v3 |
|---|---|---|---|
| base（未微调） | 2.0878 | 2.0956 | 2.0774 |
| checkpoint-200 | 0.0025 | 0.0044 | 0.0023 |
| checkpoint-400 | 0.0009 | 0.0011 | 0.0004 |
| checkpoint-441（final） | 0.0009 | 0.0011 | **0.0004** |

均单调下降、无回升。⚠️ val 与 train 同源生成器，近零 loss 主要是**模板级记忆**，
不是泛化证据——判据是行为评估（第一/三节）。

---

## 四、分层与混淆矩阵（v3，关仲裁）

| 类别 | n | 基线 | v1 | v2 | **v3** |
|---|---|---|---|---|---|
| clear | 80 | 21.2% | 100% | 100% | **100%** |
| boundary | 60 | 30.0% | 100% | 100% | **100%** |
| transition | 20 | 0.0% | 100% | 100% | **100%** |
| composite | 20 | 0.0% | 95.0% | 80.0% | **95.0%** |
| misleading | 10 | 0.0% | 100% | 100% | **100%** |
| antihall | 10 | 0.0% | 50.0% | 80.0% | **90.0%** |

混淆矩阵（v3，故障子集）：管道泄漏 55/56、水泵气蚀 43/44、过滤器堵塞 39/39、
线圈结垢 31/31。v3 全部 200 条中**仅 2 条判错**：
`idx=90` antihall→"无故障"（conf 0.90）、`idx=124` 水泵气蚀→管道泄漏（conf 0.85）。

**MS1 发现的"气蚀/泄漏全误判为堵塞"偏置被彻底消除。**

---

## 五、基线复现（新 eval 集）

原模型 × 同一 eval 集（`baseline_degraded_results_v2.jsonl`）：关仲裁故障子集 **20.6%**、
混淆对 **30.0%**、JSON 100%、证据忠实 98.9%、防幻觉 100%、P95 1.34s、
**无故障误报率 100%**（原模型从不输出"无故障"）、拒诊率 0%。

与 v1 基线（20.6% / 30.0%）一致 → **对照同分布，比较有效**。

---

## 六、唯一不达标项：置信度校准（判据与场景不匹配）

### 实测（v3）

| | 判对均值 conf | 判错均值 conf | gap | 判错 n |
|---|---|---|---|---|
| v3 | 0.832 | 0.875 | **−0.043** | **2** |

**判对样本按类别的平均 conf**：antihall 0.500（9 条）、boundary 0.780（60 条）、
composite 0.850、clear 0.880（80 条）、misleading 0.880、transition 0.900。

### 为什么该判据在本场景不成立

`判对 − 判错 > 0.15` 与"99% 准确率的记忆型 SFT"**结构性冲突**：

1. **判对集合必然含有按设计就低置信的类**——antihall 0.50、boundary 0.78，
   它们把 `mean_correct` 压到 0.83 以下；
2. **判错样本反而高置信**——模型在 99.4% 准确率下，残余错误来自记忆映射的边界外推
   （v3 两条判错 conf 分别为 0.90 / 0.85），`mean_wrong` 自然更高；
3. **`n_wrong = 2`**——该统计量是在 **2 个样本**上算出来的，任何结论都不具统计意义；
4. **v2 的实验已验证反向不可行**：把 boundary 置信度改成梯度后，gap 反而恶化到 −0.088，
   且准确率掉了 2.2pp（第二节）。

**结论：这不是数据缺陷，是该门槛被套用在了错误的场景**——它本是为"会含糊"的弱模型设计的。

### 建议（三选一）→ **已采纳 B（2026-09-11）**

- **A 改判据**：将校准项改为"**判错样本 conf < 0.8**"或 **ECE**，
  或限定在故障子集内评估。当前 v3 的两条判错 conf 为 0.90 / 0.85，仍未满足 <0.8，
  但这是一个**可验证、可优化**的目标（可通过在少数难例上引入低置信标注来达成）；
- **✅ B（采纳）记入待办**：承认 SFT 阶段的置信度不可用作安全闸门，把"不确定性表达"放到
  MS7（DPO/GRPO）用偏好数据教；级联部署（MS6）**不依赖**模型置信度做升级判断，
  改用仲裁层或规则；
- **C 维持现状**：接受该校准项不达标并记录豁免理由。

> **决策记录**：用户于 2026-09-11 选择 **B**。落地：MS6 级联升级门控
> `reasoning/confidence.py:CascadeGate` **只吃可验证信号**（JSON 解析 / 防幻觉校验 /
> 端点可用性），`decide()` 签名不含置信度入参（`tests/test_cascade.py::test_L4_8` 断言）；
> 不确定性表达推迟至 MS7。校准项在 SFT 阶段记为**已知豁免（deferred to MS7）**，非数据缺陷。

---

## 七、硬门槛结论

`JSON 100% ≥99%` ✅ ｜ `证据忠实 100% ≥98%` ✅ ｜ `防幻觉 100%` ✅

**三项硬门槛全部通过**，据 `docs/LLM_SFT后训练实施指南.md` 5.3 节，具备进入
MS6 级联部署的资格（校准项按第六节处理）。

---

## 八、给 MS6 的前置结论（2026-09-11 均已落地）

1. **必须关闭代码仲裁**：v1 关仲裁 99.4% vs 含仲裁 58.8%；v3 关仲裁 99.4% vs 含仲裁 82.9%。
   仲裁层是为弱基线设计的兜底，会改错 SFT 的正确答案。
   → ✅ 已实现：`RootCauseReasoner` 参数化 `arbitrate`，级联前置 2B `arbitrate=False`。
2. **不要让级联升级逻辑依赖模型自报置信度**（理由见第六节）。
   → ✅ 已实现：`CascadeGate` 只吃 JSON/防幻觉/端点三信号（决策 B）。
3. **补回归集**：当前评估第 9 项"未接入"。
   → ✅ 已实现：`tools/build_regression_set.py` + `tests/run_regression.py`（基线 108/110，零退化）。
4. 量化复评（INT4 vs bf16 混淆对差 ≤3pp）尚未执行。
   → 🔄 deferred：`tools/mff_quantize.py` 就绪，待装 autoawq；bf16 先行部署。

---

## 附录 A：数据生成器 bug（已修复）

### 根因
v1 `gen_antihall` 对**已渲染好的 prompt 字符串**做正则手术：

```python
re.sub(rf'( "{feat_key}": )[-\d.]+', ...)   # 要求键前有空格
```

`"出水温度"` 是 features JSON 的**首个键**（前面是 `{` 而非空格），**永远匹配不到**；
但同一 corruption 的 stats 行 `进出水温差_℃` 被成功改写，于是 `new_prompt != prompt`
成立、样本未被跳过 → 产出"输入无任何越界、却声称出水温度=5.0 越界"的**毒样本**
（且 5.0 本就在 [0,100] 内，判据自相矛盾）。

### 影响面
train **20/85**、val **5/15**、eval **3/10** 的 antihall 样本标签错误（占该类 24%）。

### 修复（4 处）
1. **结构化注入**：改走 `build_sample(corrupt=...)` 直接改 `report` dict，
   不再对渲染后的字符串做正则手术；出水温度场景用温差 stat 联动，消除 stats/features 矛盾；
2. **违规说明与注入一致**：每条 corruption 显式带 `reason`，废弃"一律套物理界限模板"；
3. **出口自检** `antihall_grounded()`：声称的违规值必须真出现在输入特征中且确实越界；
4. **审计新增第 7 项 GVIO**（`tools/audit_sft_data.py`）独立复检同一不变量。
   用新审计回归 v1 数据，**精确报出 28 项失败（train 20 / val 5 / eval 3）**，
   与人工核查完全吻合；v2/v3 数据 GVIO 违规 **0 条**。

### 效果
拒诊率 **50% → 90%**（v1 真实值 71% → v3 90%）。

---

## 附录 B：复现命令

```bash
# 数据生成 + 七项审计
conda run -n mff_agent python tools/gen_sft_data.py --out-dir data/sft
conda run -n mff_agent python tools/audit_sft_data.py --dir data/sft

# 模板 sanity check（开训前必做）
conda run -n mff_sft python tools/check_lf_template.py

# 训练 / 导出 / 校验
conda run -n mff_sft llamafactory-cli train  tools/mff_full_sft.yaml
conda run -n mff_sft llamafactory-cli export tools/mff_export.yaml
conda run -n mff_sft python tools/verify_export.py
conda run -n mff_sft python tools/eval_loss.py --device cuda:0

# 起服务（SFT 产物，GPU0:3761）+ MS5 三次复跑
nohup bash tools/serve_sft.sh > logs/vllm_sft_3761.log 2>&1 &
for s in 1 2 3; do
  conda run -n mff_agent python tests/eval_sft_model.py \
    --url http://localhost:3761/v1 --model mff-sft-minicpm5-2b \
    --arbitration off --seed $s \
    --out logs/sft_eval_v3_seed$s.jsonl --report logs/sft_eval_v3_report.md
done
```
