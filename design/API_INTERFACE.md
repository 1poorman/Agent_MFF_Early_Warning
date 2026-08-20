# 中频炉水冷系统预警智能体 · 接入接口文档

> 依据大赛《工业智能体大赛智能体接入接口文档》规范，提供 **OpenAPI（RESTful）** 与 **MCP（Model Context Protocol）** 双协议接入方式。
> 版本：v1.0.0　　更新日期：2026-08-20

---

## 一、概述

本智能体对外提供中频炉水冷系统的**多级预警、根因诊断、工单闭环**能力，支持两种接入模式：

| 模式 | 协议 | 适用场景 |
|---|---|---|
| RESTful 服务 | OpenAPI 3.0 | 第三方系统/平台直接调用预警、诊断、工单接口 |
| 智能体工具 | MCP (JSON-RPC 2.0) | 作为 MCP Server 暴露给大模型/Agent 平台调用 |

**核心能力映射**

| 能力 | RESTful 路径 | MCP 工具 |
|---|---|---|
| 数据接入 | `POST /api/v1/data/ingest` | `ingest_data` |
| L1 规则预警 | `POST /api/v1/warn/l1` | `rule_warn` |
| L2 趋势预测 | `POST /api/v1/warn/l2` | `trend_forecast` |
| L3 根因诊断 | `POST /api/v1/diagnose` | `root_cause_diagnose` |
| 工单生成 | `POST /api/v1/workorder` | `generate_workorder` |
| 反馈归档 | `POST /api/v1/feedback` | `submit_feedback` |
| 健康检查 | `GET /api/v1/health` | — |

---

## 二、OpenAPI 标准协议规范

### 2.1 文档信息

```yaml
openapi: 3.0.0
info:
  title: 中频炉水冷系统预警智能体 API
  version: 1.0.0
  description: 多级预警 + 根因诊断 + 工单闭环
servers:
  - url: http://localhost:8000/api/v1
    description: 本地部署
  - url: http://edge-gateway:8000/api/v1
    description: 边缘网关
```

### 2.2 核心接口

#### 2.2.1 数据接入 `POST /data/ingest`

**请求体**（对齐 simulator 数据契约，1Hz、1 位小数）：

```json
{
  "records": [
    {
      "timestamp": "2026-08-20 14:30:00",
      "inlet_temp": 28.0, "outlet_temp": 42.0,
      "pressure": 156.0, "flow_rate": 4.1, "flow_velocity": 1.2,
      "tank_level": 192.0, "conductivity": 550.0,
      "cabinet_temp": 40.0, "cabinet_humidity": 57.0,
      "furnace_temp": 1645.0, "electric_power": 3000.0, "electric_current": 1882.0,
      "operating_condition": "melting"
    }
  ]
}
```

**响应**：

```json
{
  "code": 0,
  "data": {
    "accepted": 1,
    "quality": {"completeness": 1.0, "imputed": 0, "outliers_removed": 0}
  }
}
```

#### 2.2.2 L3 根因诊断 `POST /diagnose`

**请求体**（异常特征 + L1/L2 上报 + 上下文自动注入）：

```json
{
  "features": {"出水温度": 55.2, "流量": 4.0, "压力": 140.0, "湿度": 74.0},
  "condition": "melting",
  "l1_alerts": [{"rule_id": "FLOW_LOW", "message": "流量低于额定80%"}],
  "l2_forecast": {"压力@+10min": "135kPa(下降)"},
  "sensor_names": ["出水温度", "流量", "压力", "湿度"]
}
```

**响应**（对齐大赛 6.1 成果展示格式）：

```json
{
  "code": 0,
  "data": {
    "root_cause": "管道泄漏",
    "confidence": 0.98,
    "level": "red",
    "evidence": ["湿度上升+压力下降", "近期工单: 3号阀更换DN50密封圈"],
    "sop": ["停炉降温至800℃", "携带测漏仪检查3号阀后管道", "备件: DN50密封圈"],
    "hallucination_check": {"physics": true, "kg": true, "confidence": 0.98},
    "manual_required": false
  }
}
```

#### 2.2.3 工单生成 `POST /workorder`

**响应**（结构化工单）：

```json
{
  "code": 0,
  "data": {
    "order_id": "WO-20260820-0007",
    "level": "red",
    "trigger_time": "2026-08-20 14:30:00",
    "features": "出水温度55.2℃且流量<额定80%，压力缓慢震荡下跌",
    "root_cause": "管道泄漏",
    "confidence": 0.98,
    "evidence": ["..."],
    "hallucination_check": {"physics": true, "kg": true, "confidence": 0.98},
    "sop": ["..."],
    "spare_parts": ["DN50密封圈", "检漏仪"],
    "emergency_plan": {"plan_id": "EP-002", "name": "炉体漏水入铁水应急"},
    "push_records": [{"channel": "sms", "receiver": "厂长", "status": "sent"}]
  }
}
```

#### 2.2.4 反馈归档 `POST /feedback`

```json
{
  "order_id": "WO-20260820-0007",
  "actual_root_cause": "管道泄漏",
  "is_true_fault": true,
  "handling_time_min": 25.0,
  "effect": "漏点补焊完成"
}
```

### 2.3 认证与错误码

- 认证：`security: bearerAuth`（HTTP Bearer Token）
- 错误码：`0` 成功；`4001` 参数校验失败；`4003` 数据质量不达标（重采）；`5001` 模型推理失败（自动降级）

---

## 三、MCP 标准协议规范

本智能体可作为 **MCP Server**，通过 JSON-RPC 2.0 暴露工具给大模型/Agent 平台。

### 3.1 握手

```json
// 客户端 -> 服务端
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
  "protocolVersion":"2024-11-05","capabilities":{"tools":{}},
  "clientInfo":{"name":"mff-platform","version":"1.0.0"}}}

// 服务端 -> 客户端
{"jsonrpc":"2.0","id":1,"result":{
  "protocolVersion":"2024-11-05","capabilities":{"tools":{}},
  "serverInfo":{"name":"mff-early-warning","version":"1.0.0"}}}
```

### 3.2 工具列表 `tools/list`

```json
{"jsonrpc":"2.0","id":2,"result":{"tools":[
  {"name":"root_cause_diagnose","description":"中频炉水冷系统根因诊断（多跳因果推理+三层防幻觉）",
   "inputSchema":{"type":"object","properties":{
     "features":{"type":"object","description":"异常特征 {传感器名: 数值}"},
     "condition":{"type":"string","description":"当前工况"},
     "l1_alerts":{"type":"array","description":"L1规则预警"},
     "l2_forecast":{"type":"object","description":"L2趋势预测"}},
    "required":["features"]}},
  {"name":"generate_workorder","description":"生成标准化运维工单并分级推送",
   "inputSchema":{"type":"object","properties":{
     "root_cause":{"type":"string"},"confidence":{"type":"number"},
     "level":{"type":"string","enum":["red","orange","yellow"]}},
    "required":["root_cause"]}},
  {"name":"submit_feedback","description":"归档处置反馈，触发模型迭代",
   "inputSchema":{"type":"object","properties":{
     "order_id":{"type":"string"},"actual_root_cause":{"type":"string"},
     "is_true_fault":{"type":"boolean"},"handling_time_min":{"type":"number"},
     "effect":{"type":"string"}},"required":["order_id","is_true_fault"]}}
]}}
```

### 3.3 工具调用 `tools/call`

```json
// Request
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
  "name":"root_cause_diagnose",
  "arguments":{"features":{"出水温度":55.2,"流量":4.0,"压力":140.0,"湿度":74.0},
               "condition":"melting"}}}

// Response
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":
  "{\"root_cause\":\"管道泄漏\",\"confidence\":0.98,\"level\":\"red\",...}"}]}}
```

---

## 四、部署与对接说明

| 项目 | 说明 |
|---|---|
| 运行环境 | conda `mff_agent`（Python 3.10），依赖见 `requirements.txt` |
| 大模型 | `.env` 配置（Qwen3.6-27B，OpenAI 兼容接口） |
| 时序模型 | `models/`（快轨 GRU / 异常 VAE+IF / 精轨 PatchTST 权重） |
| 边缘-云协同 | 边缘端跑 L1/L2（毫秒级），云端跑 L3 大模型推理 |
| 降级策略 | LLM 不可用时自动切换图谱+数值鉴别兜底，保证 7×24 可用 |

**性能指标（实测）**

| 指标 | 数值 |
|---|---|
| L1 单条判定时延 | 15µs |
| 端到端时延（图谱兜底） | 72ms |
| 特征过滤效率 | 98% |
| 根因定位准确率 | 100%（4/4，3次复跑稳定） |
| 预警提前量（缓变故障） | 302min |

---

> 注：本文档可导出为 docx 提交大赛（格式对齐《工业智能体大赛智能体接入接口文档》模板）。
