# Agent_MFF_Early_Warning

中频炉水冷系统多参数融合预警智能体（第十一届"创客中国"工业智能体大赛）。

实现从**"事后报警"到"事前预警"**、从**"单点监测"到"系统健康诊断"**的跨越：
L1 规则预警（微秒级）→ L2 趋势预测（提前 10~30min）→ L3 大模型根因诊断（三层防幻觉），
并由**四大智能体**串联形成完整闭环。

## 一、系统架构

```
┌──────────────────── 四大智能体（server/agents.py）────────────────────┐
│                                                                        │
│  ① 数据管理智能体        ② 预警分析智能体         ③ 故障处置智能体      │
│  ────────────────  ──►  ──────────────────  ──►  ──────────────────  │
│  ·传感器采集/接收         ·L1 规则预警(14µs)         ·工单生成(10字段)   │
│   (预留Modbus/OPC UA/    ·L2 趋势预测+异常检测       ·应急预案联动       │
│    MQTT/RTSP)           (GRU/VAE+IF/PatchTST)       ·红橙黄分级推送     │
│  ·质量管控(对齐/插补/     ·L3 LLM根因诊断                  │            │
│   剔除, 误差<2%)          (图谱+知识库+工况上下文)            │            │
│  ·规整数据输出                   ▲                    ▼            │
│    (1Hz/1位小数)                │ 知识库更新      ④ 持续优化智能体     │
│                                 └──────────────── 反馈归档/训练样本/   │
│                                                    知识库迭代         │
└────────────────────────────────────────────────────────────────────────┘
        ▲                    ▲                        ▲
        │ 1Hz 物理机理仿真    │ 时序模型                │ 大模型
   simulator/           detection/ + models/      reasoning/ + .env
```

**技术栈**：numpy/pandas（物理仿真）· GRU+attention/PatchTST（时序预测）· VAE+孤立森林（异常检测）· 知识图谱+Qwen LLM（根因推理）· FastAPI+MCP+ECharts（服务与界面）

## 二、快速开始

### 2.1 环境搭建

```bash
# 1. 创建 conda 环境
conda create -n mff_agent python=3.10 -y
conda activate mff_agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 PyTorch（CUDA 13.0，按需选择）
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130

# 4. 配置大模型（可选）：项目根目录创建 .env
#    url: <LLM服务地址>/v1
#    key: <API_KEY>
#    big_model_name: Qwen3.6-27B-INT4
#    未配置时自动降级为"图谱+数值鉴别"兜底模式，功能完整可用

# 5. 训练时序模型（生成 models/ 下的权重，需 GPU 或耐心）
conda run -n mff_agent python -u tests/test_ms3_detection.py
#    首次运行自动训练并缓存：快轨 GRU×3 + VAE+IF 异常检测 + 精轨 PatchTST
```

### 2.2 启动 Demo

```bash
# 启动服务（RESTful API + Web 界面 + WebSocket 实时流）
uvicorn server.api:app --host 0.0.0.0 --port 8000
# 结束服务
pkill -f "uvicorn server.api"

# 浏览器访问
#   http://localhost:8000/       四大智能体实时监控界面（单屏）
#   http://localhost:8000/docs   Swagger API 文档
```

**界面操作**（单屏四智能体布局，无需滚动）：
1. 顶部选择故障场景（正常/过滤器堵塞/水泵气蚀/管道泄漏/线圈结垢）与回放倍速
2. 点击"开始监测"——数据 1Hz 按倍速流入：
   - **① 数据管理智能体**（左）：实时物理量曲线 + 6 指标卡片 + 质量管控状态
   - **② 预警分析智能体**（中）：L1 规则日志 / L2 异常分仪表与趋势预测 / L3 根因诊断卡片
   - **③ 故障处置智能体**（右上）：自动生成工单（级别/根因/SOP/备件/推送明细/应急预案）
   - **④ 持续优化智能体**（右下）：反馈归档统计、训练样本计数、知识库更新日志

```bash
# MCP Server（可选，供 Agent 平台挂载）
python -m server.mcp_server                                    # stdio 传输
python -m server.mcp_server --transport streamable-http --port 8100   # HTTP 传输
```

### 2.3 一键演示（赛事评审用）

```bash
# 编排工作流：四大智能体一键串联（数据→预警→工单→反馈）
curl -s -X POST http://localhost:8000/api/v1/workflow/run \
  -H "Content-Type: application/json" \
  -d '{"duration": 600, "fault": "pipe_leak", "fault_start": 180}' | python3 -m json.tool
```

返回全链路结果：预警级别 → L3 根因（置信度/证据/防幻觉校验）→ 工单+推送 → 反馈归档，
含各智能体环节耗时 trace（端到端约 5s，其中 LLM 推理 3~5s）。

### 2.4 Docker 一键部署（可选，基于 conda 环境 mff_agent）

```bash
# 1. 构建并启动（PostgreSQL + API + MCP 三个服务）
docker compose up -d --build

# 2. 访问
#   http://localhost:8000/       四智能体监控界面
#   http://localhost:8000/docs   Swagger API 文档
#   MCP Server: http://localhost:8100

# 3. 常用命令
docker compose logs -f api      # 查看 API 运行日志
docker compose ps               # 服务状态
docker compose down             # 停止（保留数据卷）
```

- 镜像基于 `continuumio/miniconda3`，内部创建与本地一致的 `mff_agent` conda 环境（Python 3.10）
- 大模型配置通过只读挂载项目根 `.env` 自动注入；`config/settings.yaml` 同样只读挂载，改后重启生效
- 模型权重 `models/` 只读挂载、`data/` 与 `logs/` 读写挂载，不随镜像分发
- 默认 CPU 版 PyTorch；如需 CUDA：`docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 -t mff-agent:latest .`

## 三、四大智能体与 API

| 智能体 | 职责 | RESTful | MCP 工具 |
|---|---|---|---|
| ① 数据管理 | 采集/接收/预处理，输出 L1/L2 可直接使用格式 | `/api/v1/agents/data-manager/{collect,ingest,schema}` | `data_manager_*` |
| ② 预警分析 | L1~L3 多级预警与根因诊断（注入图谱/知识库/工况表） | `/api/v1/agents/warning-analyzer/analyze` | `warning_analyzer_analyze` |
| ③ 故障处置 | 工单生成 + 应急预案 + 分级通知 | `/api/v1/agents/fault-handler/handle` | `fault_handler_handle` |
| ④ 持续优化 | 反馈归档、知识库更新、微调触发 | `/api/v1/agents/optimizer/{feedback,update-knowledge,status}` | `optimizer_*` |
| ★ 编排工作流 | 四智能体一键串联演示 | `/api/v1/workflow/run` | `workflow_run` |

完整接口规范与 curl 示例见 `design/API_INTERFACE.md`（含大赛提交 docx 版）。

## 四、关键技术实现

### 4.1 物理机理数据仿真（simulator/）

不做随机数生成，各物理量严格满足物理约束（评委可用公式独立验证）：

| 物理规律 | 实现方式 | 验证结果 |
|---|---|---|
| 炉体热平衡 `C·dT/dt=P·η-h·(T-T_amb)` | 一阶微分方程数值积分，功率比例调节 | 熔炼炉温 35→1650℃ 合理爬升 |
| 冷却水热平衡 `T_out=T_in+Q/(c·ṁ)` | 热功率与流量联立，流量过低温差封顶 | ΔT 重建中位误差 0.06℃ |
| 管网特性 `P=P_s+R·Q²`，`v=Q/A` | 水泵扬程-管路阻抗模型，一阶惯性 | 重建静压 120.1±1.3kPa（设定120） |
| 电气公式 `P=√3·U·I·cosφ` | 电流由功率推导 | 误差 <1% |
| 故障物理联动 | 4 类故障注入（堵塞/气蚀/泄漏/结垢），多参数耦合 | 泄漏→压力降+湿度升+液位降同现 |

### 4.2 数据质量管控（perception/quality.py）

三段式流水线：**时间戳对齐**（去重/排序/1Hz 补洞）→ **异常点剔除**（物理约束裁剪 +
Hampel 滤波 + 速率限制）→ **同工况加权插补**（误差 <2%）。
关键细节：速率限制阈值按物理可变性整定（压力 50kPa/s 保留气蚀真实震荡，
尖刺由 Hampel 24kPa 阈值拦截）。

### 4.3 L1 规则引擎（detection/rule_engine.py）

- 8 条瞬时规则 + 2 条组合逻辑（流量低+温度高→积热；压力低+湿度高→疑似泄漏）
- 衍生特征规则：单位电耗温升率漂移（换热效率衰减前兆）、P-Q 特性偏移度（管网阻抗前兆）
- 进水温度阈值**季节动态修正**（24h 滚动基线上浮，消除夏季误报）
- 电气柜**凝露预测**：Magnus-Tetens 露点 + 裕度线性趋势外推（提前 ~10min 预警）

### 4.4 L2 双轨预测与异常检测（detection/）

| 轨道 | 模型 | 场景 | 实测 |
|---|---|---|---|
| 快轨 | GRU+attention **残差学习**（末值持续基线+模型学偏差） | 单参数 ≤10min，边缘常驻 | MAPE 1.05% |
| 精轨 | 通道独立 PatchTST | 多参数联合 30min | MAPE <5% |
| 异常轨 | VAE（仅正常数据训练）+ 孤立森林（残差特征空间） | 始终在线 | 检出 99.9%/误报 0.61% |
| 路由 | 参数数×步长×异常置信度规则分流 | 快轨↔精轨自动升级/降级 | 4/4 正确 |

**残差学习**是快轨精度的关键：直接预测原始序列 MAPE 10% 不达标，
改为预测"相对末值持续基线的偏差"后降至 1%。

### 4.5 L3 大模型根因推理（reasoning/）

**四层输入上下文**：L1/L2 上报 + 统计鉴别特征（尾段 120s 去趋势 std/湿度均值等）+
近期维修工单 + 工况运行表 → CoT 提示 → LLM 多跳推理。

**三层防幻觉**：
1. 物理硬约束（热力学第二定律/数值界限）
2. 知识图谱交叉验证（34 节点 39 边，推理路径必须存在）
3. 置信度分级门控（≥90% 直出 / 70~90% Top3 / <70% 人工）

**三重仲裁**（演示稳定性核心）：
1. 图谱 Top1 物理先验 vs LLM 结果一致性
2. 统计硬校验：湿度未升高禁止判泄漏（防把气蚀误判为泄漏）
3. 统计强先验：确定性规则（如湿度>70%RH=泄漏铁证）与 LLM 矛盾时回退

**性能优化**：`chat_template_kwargs.enable_thinking=false` 关闭思考模式，
推理时延 30~60s → **3~5s**，4 类故障根因正确率保持 100%（3 次复跑稳定）。

### 4.6 闭环处置（action/）

- 结构化工单（10 字段，对齐大赛 6.1 格式）+ 故障→备件自动映射
- 红色预警自动挂载应急预案（EP-001 停电/EP-002 漏水入铁水，含禁止事项）
- 红(声光+短信+电话)/橙(APP+短信)/黄(APP) 三级推送矩阵
- 反馈归档 JSONL 持久化，真实故障样本 ≥5 触发微调标记，根因修正自动更新知识库

### 4.7 电气柜凝露风险预警

- 露点温度：Magnus-Tetens 公式 `T_d = c·γ/(b-γ)`（含温度+湿度）
- 凝露判据：柜体表面温度 − 露点温度 < 3℃（绝缘下降/短路风险）
- 缓变趋势：裕度线性外推，预测 10min 内跌破阈值即预警（提前 ~9min）
- 实时流集成：`stream_step` 每帧计算露点/裕度，L2 面板显示"露点/裕度/预测"

### 4.8 时序数据存储（storage/）

本地 PostgreSQL 16 + 声明式月度 RANGE 分区（等效时序库 hypertable，无需 TimescaleDB 扩展）：

```bash
# 建库（一次性）
PGPASSWORD=postgres psql -h localhost -U postgres -c "CREATE DATABASE mff_tsdb;"
python -c "from storage import TimeSeriesDB; print(TimeSeriesDB().stats())"  # 自动建表
```

- 写入：WebSocket 实时流异步落库（每 10 条合并，失败不影响 Demo）
- 查询：`GET /api/v1/tsdb/stats`（分区/记录数）、`POST /api/v1/tsdb/query`（区间/工况/故障过滤）
- 分区：按月度 RANGE，自动预建下月分区，索引覆盖 `ts/工况/故障标签`

## 五、目录结构

```
config/        集中参数设置（settings.yaml）+ 统一日志配置
simulator/     物理机理数据仿真（含 4 类故障注入）
perception/    数据接入与质量管控
detection/     L1 规则引擎 / L2 快轨+精轨+异常检测 / 模型路由
reasoning/     知识图谱 / LLM 根因推理 / 三层防幻觉 / 置信度门控
action/        工单 / 分级推送 / 应急预案 / 反馈归档
workflow/      五节点端到端编排
context/       维修工单 / 工况运行表（LLM 上下文）
tools/         热平衡 / 露点 / 时序衍生特征计算器
server/        四大智能体 + RESTful API + MCP Server + Web 界面
storage/       时序数据存储（本地 PostgreSQL 月度分区表，自动建分区）
design/        蓝图 / 里程碑 / 测试报告 / 接口文档
tests/         42 项验收测试（全部通过）
logs/          运行日志（app.log / error.log，自动滚动）
```

## 七、集中配置与日志

### 7.1 集中参数设置（config/）

所有运行参数统一集中在 `config/settings.yaml`，启动时由 `config` 包加载为类型化配置对象：

```python
from config import get_settings
cfg = get_settings()
cfg.app.port            # 服务端口
cfg.database.dsn        # 时序库连接串
cfg.rules               # L1 规则阈值
cfg.simulator           # 仿真器参数
```

**覆盖优先级（低 → 高）**：
1. `config/settings.yaml` 默认值
2. 项目根 `.env`（`url` / `key` / `big_model_name` 等 llm 段自动合并）
3. 环境变量 `MFF_<段>_<键>`（如 `MFF_APP_PORT=8080`、`MFF_DATABASE_DSN=...`）

各模块提供 `from_settings()` 便捷构造（`SimConfig`、`RuleThresholds`、`DBConfig`、`QualityController`），
服务层 `AgentService` 已全部改为从集中配置读取。

### 7.2 运行日志（config/logging_config.py）

应用启动时自动初始化，控制台 + 文件双输出：

```bash
logs/app.log       # 全量运行日志（10MB 滚动 ×10）
logs/error.log     # ERROR 及以上错误日志
```

关键日志埋点：服务初始化、实时流接入/断开、L1/L2/L3 预警触发、根因诊断结果、工单生成、编排工作流执行。
自定义日志级别/目录见 `config/settings.yaml` 的 `logging` 段，或环境变量 `MFF_LOGGING_LEVEL=DEBUG`。

## 六、测试与验证

```bash
python tests/test_simulator.py        # 物理一致性（8 项）
python tests/test_ms1_perception.py   # 感知层（4 项）
python tests/test_ms2_rule_engine.py  # L1 规则（7 项）
python tests/test_ms3_detection.py    # L2 预测+异常+路由（8 项）
python tests/test_ms4_reasoning.py    # 根因推理+防幻觉（4 项）
python tests/test_ms5_action.py       # 闭环处置（5 项）
python tests/test_ms6_workflow.py     # 工作流集成（4 项）
```

**核心指标（实测，对照赛事要求）**：

| 指标 | 要求 | 实测 |
|---|---|---|
| L1 响应时延 | <10ms | **14.3µs** |
| L2 预警提前量 | ≥10min | **302min**（缓变故障趋势越限预测） |
| 预测误差 | <5% | **MAPE 1.05%** |
| 根因定位准确率 | ≥85% | **100%**（4/4×3 次复跑） |
| 误报率 | <5% | **0~0.61%** |
| 端到端时延 | <3s | **75ms**（兜底）/ **~5s**（含 LLM） |
| 特征过滤效率 | ≥90% | **98%** |

## 八、文档

| 文档 | 说明 |
|---|---|
| `design/BLUEPRINT.md` | 总体蓝图（架构/数据契约/阈值字典） |
| `design/MILESTONES.md` | 6 个里程碑拆解与验收标准 |
| `design/TEST_REPORT.md` | 测试报告汇总（40 项 100% 通过） |
| `design/API_INTERFACE.md` | 四大智能体接口文档（OpenAPI+MCP+curl 示例） |
| `docs/*.docx` | 大赛提交版（接口文档/测试报告） |
