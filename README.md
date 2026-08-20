# Agent_MFF_Early_Warning

中频炉水冷系统多参数融合预警智能体（第十一届"创客中国"工业智能体大赛）。

实现从**"事后报警"到"事前预警"**、从**"单点监测"到"系统健康诊断"**的三级预警体系：
L1 规则预警（毫秒级）→ L2 趋势预测（提前 10~30min）→ L3 大模型根因诊断（三层防幻觉），
并通过工单生成、分级推送、应急预案、反馈归档形成完整闭环。

## 架构

```
┌─────────────────────── Agent 工作流编排 (workflow/) ───────────────────────┐
│  数据接入 → 边缘特征提取 → 根因推理(防幻觉) → 工单/应急/推送 → 反馈归档      │
└──────────────┬──────────────────────────────────┬─────────────────────────┘
               │                                  │
┌──────────────▼───────────┐          ┌───────────▼────────────┐
│ 边缘端 · 小模型集群        │ 异常片段  │ 云端 · 大模型大脑         │
│ perception/  数据质量管控  │ ───────► │ reasoning/ 知识图谱      │
│ detection/   L1 规则引擎   │ >60%置信 │            LLM 多跳推理   │
│              L2 双轨预测   │ 特征上传 │            三层防幻觉     │
│              VAE+IF 异常   │ ◄─────── │ action/    工单/推送/应急 │
└──────────────┬───────────┘  基线下发 └─────────────────────────┘
               │
┌──────────────▼───────────┐
│ simulator/  物理机理仿真   │  热平衡/水力/电气全约束的 1Hz 数据生成
└──────────────────────────┘
```

**服务接入**：`server/` 提供 RESTful API + MCP Server + Web Demo 三种接入方式。

## 快速开始

### 1. 环境

```bash
conda create -n mff_agent python=3.10 -y
conda activate mff_agent
pip install -r requirements.txt
# PyTorch（CUDA 13.0）
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130
```

### 2. 配置大模型（可选，缺失时自动降级图谱兜底）

创建 `.env`（YAML 风格）：

```
url: <LLM服务地址>/v1
key: <API_KEY>
big_model_name: Qwen3.6-27B-INT4
```

### 3. 启动 Demo

```bash
# RESTful API + Web 界面
uvicorn server.api:app --host 0.0.0.0 --port 8000

# 浏览器访问
#   http://localhost:8000/       Web Demo（实时监控+预警+工单面板）
#   http://localhost:8000/docs   Swagger API 文档
```

```bash
# MCP Server（供 Agent 平台挂载，stdio）
python -m server.mcp_server

# MCP Server（HTTP 远程调用）
python -m server.mcp_server --transport streamable-http --port 8100
```

### 4. 运行测试

```bash
python tests/test_simulator.py        # 数据模拟物理一致性
python tests/test_ms1_perception.py   # MS1 感知层
python tests/test_ms2_rule_engine.py  # MS2 L1 规则
python tests/test_ms3_detection.py    # MS3 L2 预测+异常+路由
python tests/test_ms4_reasoning.py    # MS4 根因推理+防幻觉
python tests/test_ms5_action.py       # MS5 闭环处置
python tests/test_ms6_workflow.py     # MS6 工作流集成
```

## 模块说明

| 模块 | 功能 | 关键指标（实测） |
|---|---|---|
| `simulator/` | 物理机理数据仿真 | 热平衡重建误差 0.06℃、1Hz、1 位小数 |
| `perception/` | 数据接入+质量管控 | 插补误差 0.158%、完整度 100% |
| `detection/` | L1 规则 + L2 双轨预测 + VAE/IF 异常 + 路由 | L1 14µs、MAPE 1%、检出 99.9%、提前 302min |
| `reasoning/` | 知识图谱 + LLM 根因推理 + 三层防幻觉 | 准确率 100%、幻觉 100% 拦截 |
| `action/` | 工单 + 分级推送 + 应急预案 + 反馈 | 10 字段工单、红/橙/黄分级 |
| `workflow/` | 五节点端到端编排 | 过滤 98%、端到端 75ms |
| `context/` | 维修工单 + 工况运行表（LLM 上下文） | 复现 3号阀泄漏案例 |
| `tools/` | 热平衡/露点/衍生特征计算 | 露点误差 <0.5℃ |
| `server/` | RESTful API + MCP + Web Demo | 双协议接入 |

## 数据模拟（simulator/）

基于物理机理模型生成 1Hz 传感器时序数据，严格满足热平衡/水力/电气约束。

```bash
# 24h 正常数据
python -m simulator --duration 86400 --out data/simulated/normal_24h.csv

# 注入故障（名称@起始秒:爬升秒:程度0~1）
python -m simulator --duration 21600 \
    --faults "filter_clog@10800:1800:0.9" "pipe_leak@14400:900:0.7" \
    --out data/simulated/fault_demo_6h.csv
```

可选故障：`filter_clog`（过滤器堵塞）、`pump_cavitation`（水泵气蚀）、
`pipe_leak`（管道泄漏）、`scale_buildup`（水垢缓变）。

## API 概览（四大智能体）

| 智能体 | 接口 | 说明 |
|---|---|---|
| ① 数据管理 | `POST /api/v1/agents/data-manager/collect` | 传感器数据采集（预留 Modbus/OPC UA/MQTT/RTSP） |
| | `POST /api/v1/agents/data-manager/ingest` | 原始数据接收与预处理 |
| | `GET /api/v1/agents/data-manager/schema` | L1/L2 数据格式契约 |
| ② 预警分析 | `POST /api/v1/agents/warning-analyzer/analyze` | L1~L3 多级预警与根因诊断 |
| ③ 故障处置 | `POST /api/v1/agents/fault-handler/handle` | 工单生成+应急联动+分级通知 |
| ④ 持续优化 | `POST /api/v1/agents/optimizer/feedback` | 处置反馈归档 |
| | `POST /api/v1/agents/optimizer/update-knowledge` | 知识库更新 |
| | `GET /api/v1/agents/optimizer/status` | 优化状态查询 |
| ★ 编排工作流 | `POST /api/v1/workflow/run` | 四大智能体一键串联演示 |

**MCP 工具（9 个）**：`data_manager_*`、`warning_analyzer_analyze`、`fault_handler_handle`、`optimizer_*`、`workflow_run`

完整接口规范与 curl 使用示例见 `design/API_INTERFACE.md`。

## 文档

| 文档 | 说明 |
|---|---|
| `design/BLUEPRINT.md` | 总体蓝图（模块划分+数据契约+阈值字典） |
| `design/MILESTONES.md` | 里程碑拆解（6 个可独立验收） |
| `design/TEST_REPORT.md` | 测试报告汇总（40 项测试 100% 通过） |
| `design/API_INTERFACE.md` | 接入接口文档（OpenAPI + MCP） |
| `docs/中频炉预警智能体接入接口文档.docx` | 接口文档（大赛提交版） |
| `docs/中频炉预警智能体测试报告.docx` | 测试报告（大赛提交版） |

## 性能指标（实测）

| 指标 | 赛事要求 | 实测 |
|---|---|---|
| L1 响应时延 | <10ms | **14.3µs** |
| L2 预警提前量 | ≥10min | **302min** |
| 预测误差 | <5% | **MAPE 1.05%** |
| 根因定位准确率 | ≥85% | **100%** |
| 误报率 | <5% | **0~0.61%** |
| 端到端时延 | <3s | **75ms** |
| 特征过滤效率 | ≥90% | **98%** |

## 技术栈

- **仿真**：numpy/pandas 物理机理建模
- **时序预测**：GRU+attention（残差学习）、PatchTST（多参数精轨）
- **异常检测**：VAE 自编码器 + 孤立森林
- **根因推理**：知识图谱 + 大模型（Qwen）CoT + 三层防幻觉
- **服务**：FastAPI（RESTful）+ MCP SDK（JSON-RPC 2.0）+ ECharts（Web）
