"""MCP Server：将预警智能体能力暴露为 MCP 工具（JSON-RPC 2.0）。

传输方式：stdio（默认，供 Agent 平台挂载）/ sse / streamable-http。

启动（stdio，供 MCP Host 挂载）：
    python -m server.mcp_server
启动（HTTP，供远程调用）：
    python -m server.mcp_server --transport streamable-http --port 8100
"""

import argparse
import json
from typing import Optional

from mcp.server.mcpserver import MCPServer

from .service import AgentService

mcp = MCPServer(name="mff-early-warning", version="1.0.0")

_svc: Optional[AgentService] = None


def svc() -> AgentService:
    global _svc
    if _svc is None:
        _svc = AgentService.get()
    return _svc


@mcp.tool(name="root_cause_diagnose",
          description="中频炉水冷系统根因诊断：多跳因果推理 + 三层防幻觉校验，输出结构化诊断结果（根因/置信度/证据/SOP）")
def root_cause_diagnose(features: dict, condition: str = "unknown",
                        l1_alerts: list = None, l2_forecast: dict = None) -> str:
    """L3 根因诊断。features: {传感器中文名: 数值}，如 {"出水温度":55.2,"流量":4.0}。"""
    diag = svc().diagnose(features, condition, None, l1_alerts, l2_forecast)
    return json.dumps(diag.to_dict(), ensure_ascii=False)


@mcp.tool(name="rule_warn",
          description="L1 规则预警：对传感器时序记录做阈值/组合/衍生特征判定，返回黄色预警列表")
def rule_warn(records: list) -> str:
    """L1 规则预警。records: 传感器记录列表（对齐 simulator 数据契约）。"""
    import pandas as pd
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    clean, report = svc().qc.process(df)
    alerts = svc().rule_engine.evaluate(clean)
    return json.dumps({"count": len(alerts), "alerts": alerts.to_dict("records"),
                       "quality": {"completeness": report.completeness}},
                      ensure_ascii=False, default=str)


@mcp.tool(name="generate_workorder",
          description="生成标准化运维工单并按级别分级推送（红/橙/黄），可关联应急预案")
def generate_workorder(features: dict, condition: str = "unknown") -> str:
    """由异常特征生成工单并推送。返回工单 + 应急预案 + 推送记录。"""
    s = svc()
    diag = s.diagnose(features, condition)
    wo = s.wo_gen.generate(diag, features_text="；".join(diag.evidence[:2]))
    plan = s.emergency.attach(wo)
    recs = s.notifier.push(wo)
    data = wo.to_dict()
    data["emergency_plan"] = {"plan_id": plan.plan_id, "name": plan.name} if plan else None
    data["push_count"] = len(recs)
    return json.dumps(data, ensure_ascii=False)


@mcp.tool(name="submit_feedback",
          description="归档运维处置反馈，积累真实故障样本以触发模型迭代")
def submit_feedback(order_id: str, actual_root_cause: str, is_true_fault: bool,
                    handling_time_min: float, effect: str) -> str:
    """反馈闭环。返回归档统计。"""
    stats = svc().submit_feedback(order_id, actual_root_cause, is_true_fault,
                                  handling_time_min, effect)
    return json.dumps(stats, ensure_ascii=False)


@mcp.tool(name="health",
          description="查询智能体运行状态（LLM连接/模型加载/过滤比/累计预警）")
def health() -> str:
    s = svc()
    return json.dumps({
        "status": "ok", "use_llm": s.use_llm,
        "fast_models": list(s.pipeline.fast_models.keys()),
        "anomaly_detector": s.pipeline.detector is not None,
        "stats": s.pipeline.get_stats(),
    }, ensure_ascii=False)


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
