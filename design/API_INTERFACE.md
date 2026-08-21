# 中频炉水冷系统预警智能体 · 接入接口文档

> 依据大赛《工业智能体大赛智能体接入接口文档》规范，提供 **OpenAPI（RESTful）** 与 **MCP（Model Context Protocol）** 双协议接入方式。
> 版本：v1.1.0　　更新日期：2026-08-20　　基础地址：`http://<host>:8000`

---

## 一、四大智能体总览

本智能体系统由四大自主执行智能体组成，串联形成完整工作流闭环：

```
┌──────────────┐     规整数据      ┌──────────────┐    预警+根因(json)   ┌──────────────┐
│ ① 数据管理    │ ───────────────► │ ② 预警分析    │ ──────────────────► │ ③ 故障处置    │
│ 智能体        │                  │ 智能体        │                     │ 智能体        │
│ 采集/接收/    │                  │ L1规则/L2趋势 │                     │ 工单生成/     │
│ 预处理        │                  │ /L3根因诊断   │                     │ 应急/通知     │
└──────────────┘                  └──────────────┘                     └──────┬───────┘
                                                                             │ 处置反馈
                                                                     ┌──────▼───────┐
       知识图谱/知识库/工况表 ◄───────────────────────────────────────── │ ④ 持续优化    │
                              知识库更新                                │ 智能体        │
                                                                     └──────────────┘
```

| # | 智能体 | 职责 | RESTful 前缀 | MCP 工具组 |
|---|---|---|---|---|
| ① | 数据管理智能体 | 传感器数据接收、预留传感器传输接口（Modbus/OPC UA/MQTT/RTSP）、数据预处理，返回 L1/L2 可直接使用的数据格式 | `/api/v1/agents/data-manager` | `data_manager_*` |
| ② | 预警分析智能体 | L1~L3 多级预警，接收数据管理智能体数据，注入知识图谱/知识库/工况表上下文，返回预警信息与根因（json） | `/api/v1/agents/warning-analyzer` | `warning_analyzer_analyze` |
| ③ | 故障处置智能体 | 接收预警分析结果，调用 generate_workorder 实现工单生成与预警通知 | `/api/v1/agents/fault-handler` | `fault_handler_handle` |
| ④ | 持续优化智能体 | 基于处置反馈持续更新知识库、归档训练样本、触发模型迭代 | `/api/v1/agents/optimizer` | `optimizer_*` |
| ★ | 编排工作流 | 四大智能体一键串联演示 | `/api/v1/workflow/run` | `workflow_run` |

**统一响应格式**：`{"code": 0, "data": {...}}`；错误：`{"code": 4001, "detail": "..."}`

---

## 二、RESTful 接口（OpenAPI 3.0）

### 2.1 ① 数据管理智能体

#### `POST /api/v1/agents/data-manager/collect` — 传感器数据采集

预留传感器传输接口（Modbus/OPC UA/MQTT/RTSP 真实接入时替换协议适配层，返回格式不变）。当前由物理机理仿真器供数，可注入故障用于演示。

**请求体**：

```json
{
  "duration": 600,          // 采集时长（秒），1Hz
  "fault": "pipe_leak",     // 可选注入故障: filter_clog/pump_cavitation/pipe_leak/scale_buildup，null 为正常
  "fault_start": 180,       // 故障起始秒
  "severity": 0.9           // 故障程度 0~1
}
```

**curl 示例**：

```bash
# 采集 10 分钟正常数据
curl -X POST http://localhost:8000/api/v1/agents/data-manager/collect \
  -H "Content-Type: application/json" \
  -d '{"duration": 600}'

# 采集 10 分钟数据，第 3 分钟起注入管道泄漏故障
curl -X POST http://localhost:8000/api/v1/agents/data-manager/collect \
  -H "Content-Type: application/json" \
  -d '{"duration": 600, "fault": "pipe_leak", "fault_start": 180, "severity": 0.9}'
```

**响应**（截选）：

```json
{
  "code": 0,
  "data": {
    "agent": "data_manager",
    "records": [
      {"timestamp": "2026-08-20 00:00:00", "inlet_temp": 28.0, "outlet_temp": 42.0,
       "pressure": 156.0, "flow_rate": 4.1, "cabinet_humidity": 57.0, "...": "..."}
    ],
    "quality": {"total_out": 600, "completeness": 1.0, "missing_filled": 0, "outliers_removed": 0},
    "schema": {"precision": "1 位小数（L1/L2 直接可用）", "...": "..."}
  }
}
```

#### `POST /api/v1/agents/data-manager/ingest` — 原始数据接收与预处理

**curl 示例**：

```bash
curl -X POST http://localhost:8000/api/v1/agents/data-manager/ingest \
  -H "Content-Type: application/json" \
  -d '{"records": [{"timestamp": "2026-08-20 00:00:00", "inlet_temp": 28.0, "outlet_temp": 42.0, "pressure": 156.0, "flow_rate": 4.1, "flow_velocity": 1.2, "tank_level": 192.0, "conductivity": 550.0, "cabinet_temp": 40.0, "cabinet_humidity": 57.0, "furnace_temp": 1645.0, "electric_power": 3000.0, "electric_current": 1882.0, "operating_condition": "melting"}]}'
```

#### `GET /api/v1/agents/data-manager/schema` — 数据格式契约

```bash
curl http://localhost:8000/api/v1/agents/data-manager/schema
```

### 2.2 ② 预警分析智能体

#### `POST /api/v1/agents/warning-analyzer/analyze` — 多级预警分析

接收数据管理智能体返回的 `records`，执行 L1 规则 → L2 异常检测/趋势预测 → L3 大模型根因诊断（自动注入知识图谱、维修工单知识库、工况运行表上下文），返回预警信息与根因。

**curl 示例**（链式调用：先采集再分析）：

```bash
# 第一步：采集数据（含管道泄漏故障）
curl -s -X POST http://localhost:8000/api/v1/agents/data-manager/collect \
  -H "Content-Type: application/json" \
  -d '{"duration": 600, "fault": "pipe_leak", "fault_start": 180}' > dm.json

# 第二步：将规整数据送预警分析
curl -s -X POST http://localhost:8000/api/v1/agents/warning-analyzer/analyze \
  -H "Content-Type: application/json" \
  -d "{\"records\": $(python3 -c 'import json;print(json.dumps(json.load(open("dm.json"))["data"]["records"]))')}" > wa.json

# 查看结果
python3 -m json.tool wa.json | head -50
```

**响应**（对齐大赛 6.1 成果展示格式）：

```json
{
  "code": 0,
  "data": {
    "agent": "warning_analyzer",
    "timestamp": "2026-08-20 00:09:59",
    "condition": "melting",
    "level": "orange",
    "l1": {"triggered": true, "alerts": [
      {"rule_id": "FLOW_LOW", "message": "流量 4.1L/s 低于额定 80%"}]},
    "l2": {"anomaly_score": 0.82, "anomaly_triggered": true, "trend_exceed": null},
    "l3": {
      "root_cause": "管道泄漏",
      "confidence": 0.95,
      "level": "orange",
      "evidence": ["压力下降+湿度上升", "近期工单: 3号阀更换DN50密封圈"],
      "sop": ["停炉降温至800℃", "携带测漏仪检查3号阀后管道", "备件: DN50密封圈"],
      "hallucination_check": {"physics": true, "kg": true, "confidence": 0.95}
    },
    "context_used": {"knowledge_graph": "...", "maintenance_log": "...", "operating_schedule": "..."}
  }
}
```

### 2.3 ③ 故障处置智能体

#### `POST /api/v1/agents/fault-handler/handle` — 工单生成与预警通知

接收预警分析智能体完整返回结果（含 `l3`），调用 `generate_workorder` 生成标准化工单、联动应急预案、完成分级推送。

**curl 示例**：

```bash
curl -s -X POST http://localhost:8000/api/v1/agents/fault-handler/handle \
  -H "Content-Type: application/json" \
  -d "{\"analysis\": $(cat wa.json | python3 -c 'import json,sys;print(json.dumps(json.load(sys.stdin)["data"]))')}" > fh.json
python3 -m json.tool fh.json | head -40
```

**响应**（截选）：

```json
{
  "code": 0,
  "data": {
    "agent": "fault_handler",
    "handled": true,
    "order_id": "WO-20260820-0001",
    "level": "orange",
    "root_cause": "管道泄漏",
    "confidence": 0.95,
    "sop": ["..."],
    "spare_parts": ["DN50密封圈", "检漏仪"],
    "emergency_plan": null,
    "notifications": [{"channel": "mobile_app", "receiver": "运维班长", "status": "sent"}],
    "notification_summary": {"total": 4, "channels": ["mobile_app", "sms"],
                             "receivers": ["维修工", "运维班长"]}
  }
}
```

> 红色预警（如严重泄漏）时 `emergency_plan` 自动挂载应急预案（如 EP-002 炉体漏水入铁水应急），通知升级为声光+短信+电话三级责任人。

### 2.4 ④ 持续优化智能体

#### `POST /api/v1/agents/optimizer/feedback` — 处置反馈归档

**curl 示例**：

```bash
curl -X POST http://localhost:8000/api/v1/agents/optimizer/feedback \
  -H "Content-Type: application/json" \
  -d '{"order_id": "WO-20260820-0001", "actual_root_cause": "管道泄漏", "is_true_fault": true, "handling_time_min": 25.0, "effect": "漏点补焊完成", "work_order": null}'
```

**响应**：

```json
{
  "code": 0,
  "data": {
    "agent": "continuous_optimizer",
    "archived": true,
    "order_id": "WO-20260820-0001",
    "knowledge_updated": true,
    "knowledge_update": "新增维修工单记录（修正根因: ...）",
    "stats": {"total_feedback": 1, "true_faults": 1, "false_alarms": 0, "retrain_pending": true}
  }
}
```

#### `POST /api/v1/agents/optimizer/update-knowledge` — 知识库更新

```bash
curl -X POST http://localhost:8000/api/v1/agents/optimizer/update-knowledge \
  -H "Content-Type: application/json" \
  -d '{"component": "3号阀", "action": "更换DN50密封圈", "note": "关键记录"}'
```

#### `GET /api/v1/agents/optimizer/status` — 优化状态查询

```bash
curl http://localhost:8000/api/v1/agents/optimizer/status
```

### 2.5 ★ 编排工作流（可直接演示）

#### `POST /api/v1/workflow/run` — 四大智能体一键串联

数据管理 → 预警分析 → 故障处置 → 持续优化（含模拟处置反馈），返回全链路结果与各环节耗时。

**curl 示例**：

```bash
# 一键演示：注入管道泄漏，观察全链路（数据→预警→工单→反馈）
curl -s -X POST http://localhost:8000/api/v1/workflow/run \
  -H "Content-Type: application/json" \
  -d '{"duration": 600, "fault": "pipe_leak", "fault_start": 180, "simulate_feedback": true}' | \
  python3 -m json.tool | head -60
```

**响应**（截选）：

```json
{
  "code": 0,
  "data": {
    "workflow": "数据管理 -> 预警分析 -> 故障处置 -> 持续优化",
    "fault_injected": "pipe_leak",
    "warning": {"level": "orange", "l3": {"root_cause": "管道泄漏", "confidence": 0.88, "...": "..."}},
    "work_order": {"order_id": "WO-20260820-0001", "level": "orange", "root_cause": "管道泄漏"},
    "optimization": {"archived": true, "stats": {"total_feedback": 1}},
    "trace": [
      {"agent": "data_manager", "latency_ms": 95.8, "records": 600, "completeness": 1.0},
      {"agent": "warning_analyzer", "latency_ms": 47747.2, "level": "orange"},
      {"agent": "fault_handler", "latency_ms": 0.2, "handled": true},
      {"agent": "optimizer", "latency_ms": 0.3, "archived": true}
    ],
    "total_latency_ms": 47843.5
  }
}
```

> 注：`warning_analyzer` 耗时主要来自 LLM 根因推理（Qwen 思考模型约 30~60s）；LLM 不可用时自动降级图谱+数值鉴别（毫秒级）。

### 2.6 辅助接口

| 接口 | 说明 | curl |
|---|---|---|
| `GET /api/v1/agents` | 四大智能体清单 | `curl http://localhost:8000/api/v1/agents` |
| `GET /api/v1/health` | 健康检查 | `curl http://localhost:8000/api/v1/health` |
| `GET /api/v1/docs` | Swagger 文档 | 浏览器访问 `http://localhost:8000/docs` |
| `GET /` | Web 实时监控 Demo | 浏览器访问 |

---

## 三、MCP 标准协议规范

本智能体作为 **MCP Server**，通过 JSON-RPC 2.0 暴露四大智能体工具。启动：

```bash
# stdio（供 MCP Host / Agent 平台挂载）
python -m server.mcp_server
# HTTP 远程调用
python -m server.mcp_server --transport streamable-http --port 8105
```

### 3.1 握手（initialize）

```json
// 客户端 -> 服务端
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
  "protocolVersion":"2024-11-05","capabilities":{"tools":{}},
  "clientInfo":{"name":"mff-platform","version":"1.0.0"}}}

// 服务端 -> 客户端
{"jsonrpc":"2.0","id":1,"result":{
  "protocolVersion":"2024-11-05","capabilities":{"tools":{}},
  "serverInfo":{"name":"mff-early-warning","version":"1.1.0"}}}
```

### 3.2 工具列表（tools/list，9 个）

| 工具 | 所属智能体 | 说明 |
|---|---|---|
| `data_manager_collect` | ① 数据管理 | 传感器数据采集（预留协议接口） |
| `data_manager_ingest` | ① 数据管理 | 原始数据接收与预处理 |
| `data_manager_schema` | ① 数据管理 | 数据格式契约查询 |
| `warning_analyzer_analyze` | ② 预警分析 | L1~L3 多级预警与根因诊断 |
| `fault_handler_handle` | ③ 故障处置 | 工单生成+应急+分级通知 |
| `optimizer_feedback` | ④ 持续优化 | 处置反馈归档 |
| `optimizer_update_knowledge` | ④ 持续优化 | 知识库更新 |
| `optimizer_status` | ④ 持续优化 | 优化状态查询 |
| `workflow_run` | ★ 编排 | 四大智能体一键串联演示 |

### 3.3 工具调用（tools/call）

```json
// Request：预警分析
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
  "name":"warning_analyzer_analyze",
  "arguments":{"records":[{"timestamp":"2026-08-20 00:00:00","outlet_temp":55.2,
    "pressure":140.0,"flow_rate":4.0,"cabinet_humidity":74.0,"inlet_temp":28.0}]}}

// Response
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":
  "{\"level\":\"orange\",\"l3\":{\"root_cause\":\"管道泄漏\",\"confidence\":0.95,...}}"}]}}
```

```json
// Request：编排工作流一键演示
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{
  "name":"workflow_run",
  "arguments":{"duration":600,"fault":"pipe_leak","fault_start":180}}}
```

### 3.4 客户端配置示例（Claude Desktop / Agent 平台）

```json
{
  "mcpServers": {
    "mff-early-warning": {
      "command": "conda",
      "args": ["run", "-n", "mff_agent", "python", "-m", "server.mcp_server"]
    }
  }
}
```

---

## 四、Web 实时监控 Demo

浏览器访问 `http://<host>:8000/`：

- 实时物理量曲线（进/出水温度、压力、流量、湿度，随数据滚动）
- **L1 规则预警窗口**（黄色，毫秒级触发日志）
- **L2 趋势/异常窗口**（异常分仪表 + 趋势越限预测）
- **L3 根因诊断窗口**（根因/置信度/证据链/防幻觉/SOP）
- 故障场景选择（4 类）+ 倍速控制（WebSocket `/ws/stream` 实时流）

---

## 五、部署与对接说明

| 项目 | 说明 |
|---|---|
| 运行环境 | conda `mff_agent`（Python 3.10），依赖见 `requirements.txt` |
| 启动 | `uvicorn server.api:app --host 0.0.0.0 --port 8000` |
| 大模型 | `.env` 配置（Qwen3.6-27B，OpenAI 兼容）；缺失自动降级 |
| 时序模型 | `models/`（快轨 GRU / 异常 VAE+IF / 精轨 PatchTST） |
| 降级策略 | LLM 超时(60s)/失败时自动切换图谱+数值鉴别兜底 |
| 性能实测 | L1 14µs；端到端 75ms（兜底）/ ~50s（含 LLM）；根因准确率 100% |

**认证**（预留）：`security: bearerAuth`（HTTP Bearer Token）

**错误码**：`0` 成功；`4001` 参数校验失败/数据质量异常；`5001` 模型推理失败（自动降级）

---

> 本文档对应代码：`server/api.py`（RESTful）、`server/mcp_server.py`（MCP）、`server/agents.py`（四大智能体实现）。可导出 docx 提交大赛。
