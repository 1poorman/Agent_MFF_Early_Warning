"""MCP Server：四大智能体能力暴露为 MCP 工具（JSON-RPC 2.0）。

传输方式：stdio（默认，供 Agent 平台挂载）/ sse / streamable-http。

四大智能体工具组：
  1. 数据管理智能体   data_manager_collect / data_manager_ingest / data_manager_schema
  2. 预警分析智能体   warning_analyzer_analyze
  3. 故障处置智能体   fault_handler_handle
  4. 持续优化智能体   optimizer_feedback / optimizer_update_knowledge / optimizer_status
  编排工作流          workflow_run（一键演示）

启动（stdio，供 MCP Host 挂载）：
    python -m server.mcp_server
启动（HTTP 远程调用）：
    python -m server.mcp_server --transport streamable-http --port 8100
"""

import argparse
import json
from typing import Optional

from mcp.server.mcpserver import MCPServer

from .service import AgentService

mcp = MCPServer(name="mff-early-warning", version="1.1.0")

_ag = None


def ag():
    global _ag
    if _ag is None:
        from .agents import (ContinuousOptimizerAgent, DataManagementAgent,
                             FaultHandlingAgent, WarningAnalysisAgent)
        s = AgentService.get()
        _ag = {
            "data_manager": DataManagementAgent(s),
            "warning_analyzer": WarningAnalysisAgent(s),
            "fault_handler": FaultHandlingAgent(s),
            "optimizer": ContinuousOptimizerAgent(s),
        }
    return _ag


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# ===========================================================================
# 1. 数据管理智能体
# ===========================================================================

@mcp.tool(name="data_manager_collect",
          description="【数据管理智能体】传感器数据采集（预留 Modbus/OPC UA/MQTT/RTSP 真实接口，当前由物理仿真器供数），返回预处理后的规整数据")
def data_manager_collect(duration: int = 300, fault: str = None,
                         fault_start: int = 120, severity: float = 0.9) -> str:
    return _j(ag()["data_manager"].collect(duration, fault, fault_start, severity))


@mcp.tool(name="data_manager_ingest",
          description="【数据管理智能体】接收原始传感器记录并预处理（时间戳对齐/缺失插补/异常剔除），返回 L1/L2 可直接使用的数据格式")
def data_manager_ingest(records: list) -> str:
    return _j(ag()["data_manager"].ingest(records))


@mcp.tool(name="data_manager_schema",
          description="【数据管理智能体】查询 L1/L2 可直接使用的数据格式契约（字段/单位/精度）")
def data_manager_schema() -> str:
    return _j(ag()["data_manager"].schema())


# ===========================================================================
# 2. 预警分析智能体
# ===========================================================================

@mcp.tool(name="warning_analyzer_analyze",
          description="【预警分析智能体】多级预警分析：L1 规则 -> L2 异常检测/趋势预测 -> L3 大模型根因诊断（自动注入知识图谱/维修工单/工况表上下文），返回预警信息与根因 json")
def warning_analyzer_analyze(records: list) -> str:
    return _j(ag()["warning_analyzer"].analyze(records))


# ===========================================================================
# 3. 故障处置智能体
# ===========================================================================

@mcp.tool(name="fault_handler_handle",
          description="【故障处置智能体】接收预警分析结果，生成标准化运维工单、联动应急预案并完成分级预警通知")
def fault_handler_handle(analysis: dict) -> str:
    return _j(ag()["fault_handler"].handle(analysis))


# ===========================================================================
# 4. 持续优化智能体
# ===========================================================================

@mcp.tool(name="optimizer_feedback",
          description="【持续优化智能体】归档处置反馈，积累真实故障样本触发模型迭代，并按需更新知识库")
def optimizer_feedback(order_id: str, actual_root_cause: str, is_true_fault: bool,
                       handling_time_min: float, effect: str,
                       work_order: dict = None) -> str:
    return _j(ag()["optimizer"].feedback(order_id, actual_root_cause, is_true_fault,
                                         handling_time_min, effect, work_order))


@mcp.tool(name="optimizer_update_knowledge",
          description="【持续优化智能体】手动更新知识库（新增维修工单记录，将作为 L3 诊断上下文）")
def optimizer_update_knowledge(component: str, action: str,
                               note: str = "", date: str = None) -> str:
    return _j(ag()["optimizer"].update_knowledge(
        {"component": component, "action": action, "note": note, "date": date}))


@mcp.tool(name="optimizer_status",
          description="【持续优化智能体】查询优化状态：反馈统计/微调触发/知识库规模")
def optimizer_status() -> str:
    return _j(ag()["optimizer"].status())


# ===========================================================================
# 编排工作流
# ===========================================================================

@mcp.tool(name="workflow_run",
          description="【编排工作流】四大智能体一键串联演示：数据采集->预警分析->故障处置->持续优化，返回全链路结果")
def workflow_run(duration: int = 600, fault: str = "pipe_leak",
                 fault_start: int = 180, severity: float = 0.9) -> str:
    import time as _t
    a = ag()
    trace = []
    t0 = _t.perf_counter()
    dm = a["data_manager"].collect(duration, fault, fault_start, severity)
    trace.append({"agent": "data_manager", "latency_ms": round((_t.perf_counter() - t0) * 1000, 1)})

    t0 = _t.perf_counter()
    wa = a["warning_analyzer"].analyze(dm["records"])
    trace.append({"agent": "warning_analyzer", "latency_ms": round((_t.perf_counter() - t0) * 1000, 1)})

    t0 = _t.perf_counter()
    fh = a["fault_handler"].handle(wa)
    trace.append({"agent": "fault_handler", "latency_ms": round((_t.perf_counter() - t0) * 1000, 1)})

    co = None
    if fh.get("handled"):
        t0 = _t.perf_counter()
        co = a["optimizer"].feedback(fh["order_id"], fh["root_cause"], True, 25.0,
                                     "工作流演示模拟反馈", fh)
        trace.append({"agent": "optimizer", "latency_ms": round((_t.perf_counter() - t0) * 1000, 1)})

    return _j({
        "workflow": "数据管理 -> 预警分析 -> 故障处置 -> 持续优化",
        "warning_level": wa["level"],
        "root_cause": wa["l3"]["root_cause"] if wa["l3"] else None,
        "work_order": {"order_id": fh.get("order_id"), "level": fh.get("level"),
                       "root_cause": fh.get("root_cause")} if fh.get("handled") else None,
        "optimization": co,
        "trace": trace,
    })


def main():
    parser = argparse.ArgumentParser(description="中频炉预警智能体 MCP Server")
    parser.add_argument("--transport", default="stdio",
                        choices=["stdio", "sse", "streamable-http"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
