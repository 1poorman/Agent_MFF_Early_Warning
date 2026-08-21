"""RESTful API 服务（OpenAPI 3.0）。

启动：
    uvicorn server.api:app --host 0.0.0.0 --port 8000
    # 或：python -m server.api（host/port 从 config/settings.yaml 读取）
文档：
    http://localhost:8000/docs  (Swagger UI)
界面：
    http://localhost:8000/      (Web Demo)
"""

import asyncio
import io
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_logger, get_settings, setup_logging

from .service import AgentService
from .agents import (
    DataManagementAgent, WarningAnalysisAgent,
    FaultHandlingAgent, ContinuousOptimizerAgent,
)

setup_logging()
cfg = get_settings()
logger = get_logger("server.api")

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "server" / "static"

app = FastAPI(
    title=cfg.app.title,
    version=cfg.app.version,
    description="多级预警 + 根因诊断 + 工单闭环",
)

_service: Optional[AgentService] = None


def svc() -> AgentService:
    global _service
    if _service is None:
        _service = AgentService.get()
    return _service


def agents():
    """四大智能体实例（懒加载，基于共享服务层）。"""
    s = svc()
    return {
        "data_manager": DataManagementAgent(s),
        "warning_analyzer": WarningAnalysisAgent(s),
        "fault_handler": FaultHandlingAgent(s),
        "optimizer": ContinuousOptimizerAgent(s),
    }


# ---------------- 请求/响应模型 ----------------

class IngestRequest(BaseModel):
    records: List[Dict]


class DiagnoseRequest(BaseModel):
    features: Dict[str, float]
    condition: str = "unknown"
    sensor_names: Optional[List[str]] = None
    l1_alerts: Optional[List[Dict]] = None
    l2_forecast: Optional[Dict] = None


class FeedbackRequest(BaseModel):
    order_id: str
    actual_root_cause: str
    is_true_fault: bool
    handling_time_min: float
    effect: str


class SimulateRequest(BaseModel):
    duration: int = 300          # 回放时长（秒）
    fault: Optional[str] = None  # filter_clog/pump_cavitation/pipe_leak/scale_buildup
    fault_start: int = 120


# ---------------- 接口 ----------------

@app.get("/api/v1/health")
def health():
    s = svc()
    return {"code": 0, "data": {
        "status": "ok",
        "use_llm": s.use_llm,
        "fast_models": list(s.pipeline.fast_models.keys()),
        "anomaly_detector": s.pipeline.detector is not None,
        "stats": s.pipeline.get_stats(),
    }}


@app.post("/api/v1/data/ingest")
def ingest(req: IngestRequest):
    df = pd.DataFrame(req.records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    clean, report = svc().qc.process(df)
    return {"code": 0, "data": {
        "accepted": len(clean),
        "quality": {
            "completeness": report.completeness,
            "imputed": report.missing_filled,
            "outliers_removed": report.outliers_removed,
        },
    }}


@app.post("/api/v1/warn/l1")
def warn_l1(req: IngestRequest):
    df = pd.DataFrame(req.records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    clean, _ = svc().qc.process(df)
    alerts = svc().rule_engine.evaluate(clean)
    return {"code": 0, "data": {"count": len(alerts), "alerts": alerts.to_dict("records")}}


@app.post("/api/v1/diagnose")
def diagnose(req: DiagnoseRequest):
    diag = svc().diagnose(req.features, req.condition, req.sensor_names,
                          req.l1_alerts, req.l2_forecast)
    return {"code": 0, "data": diag.to_dict()}


@app.post("/api/v1/workorder")
def workorder(req: DiagnoseRequest):
    s = svc()
    diag = s.diagnose(req.features, req.condition, req.sensor_names,
                      req.l1_alerts, req.l2_forecast)
    wo = s.wo_gen.generate(diag, features_text="；".join(diag.evidence[:2]))
    plan = s.emergency.attach(wo)
    recs = s.notifier.push(wo)
    data = wo.to_dict()
    data["emergency_plan"] = {"plan_id": plan.plan_id, "name": plan.name} if plan else None
    data["push_records"] = [vars(r) for r in recs]
    return {"code": 0, "data": data}


@app.post("/api/v1/feedback")
def feedback(req: FeedbackRequest):
    stats = svc().submit_feedback(req.order_id, req.actual_root_cause,
                                  req.is_true_fault, req.handling_time_min, req.effect)
    return {"code": 0, "data": stats}


@app.post("/api/v1/simulate")
def simulate(req: SimulateRequest):
    """回放 simulator 数据驱动端到端流程（Demo 用）。"""
    from simulator import DataSimulator, FaultSpec, SimConfig
    faults = []
    if req.fault:
        faults = [FaultSpec(name=req.fault, start=req.fault_start, ramp=600, severity=0.9)]
    sim = DataSimulator(config=SimConfig(seed=42), faults=faults)
    df = sim.run(req.duration)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    result = svc().process_window(df)
    return {"code": 0, "data": {
        "alerted": result.alerted,
        "anomaly_score": result.anomaly_score,
        "work_order": result.work_order,
        "push_count": result.push_count,
        "latency_ms": result.total_latency_ms,
        "nodes": [{"name": n.name, "status": n.status, "latency_ms": n.latency_ms}
                  for n in result.node_logs],
    }}


@app.get("/api/v1/latest")
def latest():
    """最近一次窗口结果 + 时序数据（界面展示）。"""
    s = svc()
    out = {"result": None, "series": None}
    if s.latest_result is not None:
        r = s.latest_result
        out["result"] = {
            "alerted": r.alerted, "anomaly_score": r.anomaly_score,
            "work_order": r.work_order, "push_count": r.push_count,
            "latency_ms": r.total_latency_ms,
        }
    if s.latest_window is not None:
        df = s.latest_window
        out["series"] = {
            "timestamp": df["timestamp"].astype(str).tolist(),
            "outlet_temp": df["outlet_temp"].tolist(),
            "pressure": df["pressure"].tolist(),
            "flow_rate": df["flow_rate"].tolist(),
            "cabinet_humidity": df["cabinet_humidity"].tolist(),
        }
    return {"code": 0, "data": out}


# ---------------- 时序数据文件上传（自动预警诊断） ----------------

@app.get("/api/v1/forecast-model")
async def get_forecast_model():
    """获取当前 L2 预测模型与可用模型列表。"""
    s = svc()
    engine = getattr(s.pipeline, "forecast_engine", None)
    return {"code": 0, "data": {
        "current": engine.name if engine else "gru",
        "available": ["gru", "patchtst", "timesfm"],
        "horizon_s": engine.horizon_s if engine else 600,
    }}


@app.post("/api/v1/upload")
async def upload_analyze(file: UploadFile = File(...), model: str = Form("gru")):
    """上传时序数据文件（CSV/JSON，格式同 simulator 输出契约），
    自动执行 数据管理 -> 预警分析 -> 故障处置 全链路并返回结果。

    CSV 列契约见 perception/ingest.py ALL_COLUMNS；
    缺列自动补 NaN（质量管控插补），timestamp 为必填列。
    model: L2 预测模型（gru/patchtst/timesfm），默认取 config 配置。
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传文件为空")
    try:
        name = (file.filename or "").lower()
        if name.endswith((".json", ".jsonl")):
            df = pd.read_json(io.BytesIO(raw))
        else:
            df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件解析失败: {e}")
    if "timestamp" not in df.columns:
        raise HTTPException(status_code=400, detail="数据缺少 timestamp 列")

    # 补全数据契约列（缺失列填 NaN，交由质量管控插补/兜底），保证全字段可处理
    from perception.ingest import ALL_COLUMNS
    for c in ALL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[ALL_COLUMNS].dropna(subset=["timestamp"])
    if df.empty:
        raise HTTPException(status_code=400, detail="无有效数据行")
    logger.info("上传文件解析完成 | %s 行=%d", file.filename, len(df))

    s = svc()
    records = df.to_dict("records")
    # L2 预测模型临时切换（model 参数），分析后恢复
    engine = getattr(s.pipeline, "forecast_engine", None)
    prev = None
    if engine is not None and model in ("gru", "patchtst", "timesfm"):
        prev = engine.name
        if model != prev:
            engine.switch(model, fast_models=s.pipeline.fast_models,
                          models_dir=str(s.cfg.paths.models))
            logger.info("上传分析切换预测模型: %s -> %s", prev, model)
    try:
        # ① 数据管理智能体：质量管控
        dm = DataManagementAgent(s).ingest(records)
        # ② 预警分析智能体：L1/L2/L3
        wa = WarningAnalysisAgent(s).analyze(dm["records"])
        # ③ 故障处置智能体：工单 + 通知
        fh = FaultHandlingAgent(s).handle(wa)
    finally:
        if engine is not None and prev is not None and model != prev:
            engine.switch(prev, fast_models=s.pipeline.fast_models,
                          models_dir=str(s.cfg.paths.models))
    logger.info("上传数据分析完成 | %s 预警级别=%s 根因=%s",
                file.filename, wa["level"],
                (wa["l3"] or {}).get("root_cause", "无"))

    # 绘图用序列（等间隔抽样，最多 2000 点）
    step = max(1, len(df) // 2000)
    sub = df.iloc[::step]
    series = {
        "timestamp": sub["timestamp"].astype(str).tolist(),
        "inlet_temp": sub["inlet_temp"].round(1).tolist(),
        "outlet_temp": sub["outlet_temp"].round(1).tolist(),
        "pressure": sub["pressure"].round(1).tolist(),
        "flow_rate": sub["flow_rate"].round(2).tolist(),
        "cabinet_humidity": sub["cabinet_humidity"].round(1).tolist(),
    }
    return {"code": 0, "data": {
        "filename": file.filename,
        "points": len(dm["records"]),
        "quality": dm["quality"],
        "warning": wa,
        "work_order": fh,
        "series": series,
    }}


# ---------------- 实时流（WebSocket） ----------------

class StreamRequest(BaseModel):
    fault: Optional[str] = None
    fault_start: int = 120
    speed: float = 20.0          # 回放倍速（条/秒）
    duration: int = 1800


@app.websocket("/ws/stream")
async def stream_ws(ws: WebSocket):
    """实时流：simulator 逐秒生成数据 -> 逐条处理 -> 推送指标与 L1/L2/L3 预警。"""
    await ws.accept()
    s = svc()
    s.reset_stream()
    logger.info("WebSocket 实时流接入 | client=%s", ws.client.host if ws.client else "unknown")
    try:
        req_raw = await ws.receive_json()
        fault = req_raw.get("fault")
        fault_start = int(req_raw.get("fault_start", 120))
        speed = float(req_raw.get("speed", 20.0))
        duration = int(req_raw.get("duration", 1800))
        forecast_model = str(req_raw.get("forecast_model") or "").lower()
        logger.info("实时流参数 | fault=%s fault_start=%d speed=%.1f duration=%d forecast_model=%s",
                    fault, fault_start, speed, duration, forecast_model)

        # L2 预测模型运行时切换（GRU/PatchTST/TimesFM）
        engine = getattr(s.pipeline, "forecast_engine", None)
        if engine is not None and forecast_model in ("gru", "patchtst", "timesfm"):
            engine.switch(forecast_model, fast_models=s.pipeline.fast_models,
                          models_dir=str(s.cfg.paths.models))
            logger.info("实时流预测模型: %s", engine.name)
        model_name = engine.name if engine else "gru"

        from simulator import DataSimulator, FaultSpec, SimConfig
        # 预加载量 = 当前预测后端窗口（GRU 16800s=4.6h / PatchTST 7200s / TimesFM 1024s）
        warmup_s = engine.backend_window() if engine is not None else 16800
        faults = [FaultSpec(name=fault, start=warmup_s + fault_start, ramp=300, severity=0.9)] \
            if fault else []
        sim = DataSimulator(config=SimConfig(seed=42), faults=faults)

        loop = asyncio.get_event_loop()
        # ---- 预加载阶段：填充缓冲不推送，界面提示 ----
        await ws.send_json({"preload": True, "points": warmup_s, "forecast_model": model_name})
        for _ in range(warmup_s):
            row = sim._sense(sim.step())
            row["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            s.preload_row(row)
        await ws.send_json({"preload_done": True, "points": s.preloaded,
                            "forecast_model": model_name})

        # ---- 实时推送阶段 ----
        interval = 1.0 / max(speed, 1.0)
        for i in range(duration):
            row = sim._sense(sim.step())   # 带测量噪声的传感器读数
            row["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            # stream_step 含潜在 LLM 同步推理，放入线程池避免阻塞事件循环（WS keepalive）
            out = await loop.run_in_executor(None, s.stream_step, row)
            # 时序落库（异步，每 10 条合并落一次，失败不影响 Demo）
            if getattr(s, "tsdb_ok", False) and i % 10 == 0:
                s.persist(row)
            await ws.send_json(out)
            await asyncio.sleep(interval)
        await ws.send_json({"done": True})
        logger.info("实时流完成 | 推送 %d 条", duration)
    except WebSocketDisconnect:
        logger.info("实时流客户端断开 | client=%s", ws.client.host if ws.client else "unknown")
    except Exception as e:
        logger.exception("实时流异常: %s", e)
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


@app.get("/api/v1/stream/logs")
def stream_logs():
    return {"code": 0, "data": svc().get_stream_logs()}


@app.post("/api/v1/stream/reset")
def stream_reset():
    """重置实时流缓冲与日志（新一次监测）。"""
    svc().reset_stream()
    return {"code": 0, "data": {"reset": True}}


class TsQueryRequest(BaseModel):
    start: str                        # "2026-08-21 00:00:00"
    end: str
    condition: Optional[str] = None
    fault: Optional[str] = None
    columns: Optional[List[str]] = None


@app.get("/api/v1/tsdb/stats")
def tsdb_stats():
    """时序数据库存储统计（分区/记录数）。"""
    s = svc()
    if not getattr(s, "tsdb_ok", False):
        return {"code": 0, "data": {"enabled": False, "reason": "数据库未连接"}}
    try:
        return {"code": 0, "data": {"enabled": True, **s.tsdb.stats()}}
    except Exception as e:
        return {"code": 0, "data": {"enabled": False, "reason": str(e)}}


@app.post("/api/v1/tsdb/query")
def tsdb_query(req: TsQueryRequest):
    """从时序数据库读取历史数据（区间/工况/故障过滤）。"""
    s = svc()
    if not getattr(s, "tsdb_ok", False):
        raise HTTPException(status_code=5001, detail="时序数据库未连接")
    from datetime import datetime
    df = s.tsdb.query(
        start=datetime.fromisoformat(req.start),
        end=datetime.fromisoformat(req.end),
        condition=req.condition, fault=req.fault, columns=req.columns)
    return {"code": 0, "data": {
        "count": len(df),
        "records": df.to_dict("records"),
    }}


# ---------------- 四大智能体接口 ----------------

class CollectRequest(BaseModel):
    duration: int = 300            # 采集时长（秒）
    fault: Optional[str] = None    # filter_clog/pump_cavitation/pipe_leak/scale_buildup
    fault_start: int = 120
    severity: float = 0.9


class AnalyzeRequest(BaseModel):
    records: List[Dict]            # 数据管理智能体返回的规整数据


class HandleRequest(BaseModel):
    analysis: Dict                  # 预警分析智能体返回结果


class FeedbackRequest2(BaseModel):
    order_id: str
    actual_root_cause: str
    is_true_fault: bool
    handling_time_min: float
    effect: str
    work_order: Optional[Dict] = None


class KnowledgeRequest(BaseModel):
    component: str
    action: str
    order_id: Optional[str] = None
    date: Optional[str] = None
    note: str = ""


class WorkflowRunRequest(BaseModel):
    """编排工作流一键演示请求。"""
    duration: int = 600            # 数据时长（秒）
    fault: Optional[str] = "pipe_leak"
    fault_start: int = 180
    severity: float = 0.9
    simulate_feedback: bool = True  # 是否模拟处置反馈（触发持续优化）


@app.get("/api/v1/agents")
def list_agents():
    """四大智能体清单。"""
    return {"code": 0, "data": [
        {"name": "data_manager", "title": "数据管理智能体",
         "role": "传感器数据接收/采集接口/预处理", "endpoints": [
             "POST /api/v1/agents/data-manager/collect",
             "POST /api/v1/agents/data-manager/ingest",
             "GET  /api/v1/agents/data-manager/schema"]},
        {"name": "warning_analyzer", "title": "预警分析智能体",
         "role": "L1~L3 多级预警与根因诊断", "endpoints": [
             "POST /api/v1/agents/warning-analyzer/analyze"]},
        {"name": "fault_handler", "title": "故障处置智能体",
         "role": "工单生成与预警通知", "endpoints": [
             "POST /api/v1/agents/fault-handler/handle"]},
        {"name": "optimizer", "title": "持续优化智能体",
         "role": "反馈归档与知识库持续更新", "endpoints": [
             "POST /api/v1/agents/optimizer/feedback",
             "POST /api/v1/agents/optimizer/update-knowledge",
             "GET  /api/v1/agents/optimizer/status"]},
    ]}


# ---- 1. 数据管理智能体 ----

@app.post("/api/v1/agents/data-manager/collect")
def dm_collect(req: CollectRequest):
    """传感器数据采集（预留 Modbus/OPC UA/MQTT/RTSP 接口，当前由仿真器供数）。"""
    return {"code": 0, "data": agents()["data_manager"].collect(
        req.duration, req.fault, req.fault_start, req.severity)}


@app.post("/api/v1/agents/data-manager/ingest")
def dm_ingest(req: IngestRequest):
    """接收原始传感器数据 -> 预处理（对齐/插补/剔除）-> L1/L2 可直接使用格式。"""
    try:
        return {"code": 0, "data": agents()["data_manager"].ingest(req.records)}
    except Exception as e:
        raise HTTPException(status_code=4001, detail=str(e))


@app.get("/api/v1/agents/data-manager/schema")
def dm_schema():
    """L1/L2 可直接使用的数据格式契约。"""
    return {"code": 0, "data": agents()["data_manager"].schema()}


# ---- 2. 预警分析智能体 ----

@app.post("/api/v1/agents/warning-analyzer/analyze")
def wa_analyze(req: AnalyzeRequest):
    """多级预警分析：L1 规则 -> L2 异常/趋势 -> L3 根因诊断（注入知识图谱/知识库/工况表上下文）。"""
    try:
        return {"code": 0, "data": agents()["warning_analyzer"].analyze(req.records)}
    except Exception as e:
        raise HTTPException(status_code=4001, detail=str(e))


# ---- 3. 故障处置智能体 ----

@app.post("/api/v1/agents/fault-handler/handle")
def fh_handle(req: HandleRequest):
    """工单生成 + 应急预案联动 + 分级预警通知。"""
    return {"code": 0, "data": agents()["fault_handler"].handle(req.analysis)}


# ---- 4. 持续优化智能体 ----

@app.post("/api/v1/agents/optimizer/feedback")
def co_feedback(req: FeedbackRequest2):
    """处置反馈归档 -> 训练样本积累 -> 知识库按需更新。"""
    return {"code": 0, "data": agents()["optimizer"].feedback(
        req.order_id, req.actual_root_cause, req.is_true_fault,
        req.handling_time_min, req.effect, req.work_order)}


@app.post("/api/v1/agents/optimizer/update-knowledge")
def co_update_knowledge(req: KnowledgeRequest):
    """手动更新知识库（新增维修工单记录）。"""
    return {"code": 0, "data": agents()["optimizer"].update_knowledge(
        {"order_id": req.order_id, "date": req.date,
         "component": req.component, "action": req.action, "note": req.note})}


@app.get("/api/v1/agents/optimizer/status")
def co_status():
    """持续优化状态：反馈统计/微调触发/知识库规模。"""
    return {"code": 0, "data": agents()["optimizer"].status()}


# ---- 编排工作流（一键演示） ----

@app.post("/api/v1/workflow/run")
def workflow_run(req: WorkflowRunRequest):
    """编排好的工作流：四大智能体串联一键演示。

    数据管理(采集+预处理) -> 预警分析(L1/L2/L3) -> 故障处置(工单+通知)
    -> 持续优化(模拟反馈归档)，返回全链路结果与各环节耗时。
    """
    ag = agents()
    trace = []

    # 节点1: 数据管理智能体
    t0 = time.perf_counter()
    dm = ag["data_manager"].collect(req.duration, req.fault, req.fault_start, req.severity)
    trace.append({"agent": "data_manager", "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                  "records": dm["quality"]["total_out"],
                  "completeness": dm["quality"]["completeness"]})

    # 节点2: 预警分析智能体
    t0 = time.perf_counter()
    wa = ag["warning_analyzer"].analyze(dm["records"])
    trace.append({"agent": "warning_analyzer", "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                  "level": wa["level"], "l1_triggered": wa["l1"]["triggered"],
                  "anomaly_score": wa["l2"]["anomaly_score"]})

    # 节点3: 故障处置智能体
    t0 = time.perf_counter()
    fh = ag["fault_handler"].handle(wa)
    trace.append({"agent": "fault_handler", "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                  "handled": fh["handled"],
                  "order_id": fh.get("order_id"), "level": fh.get("level")})

    # 节点4: 持续优化智能体（模拟处置反馈）
    co = None
    if req.simulate_feedback and fh.get("handled"):
        t0 = time.perf_counter()
        co = ag["optimizer"].feedback(
            order_id=fh["order_id"],
            actual_root_cause=fh["root_cause"],
            is_true_fault=True,
            handling_time_min=25.0,
            effect="故障处置完成（工作流演示模拟反馈）",
            work_order=fh)
        trace.append({"agent": "optimizer", "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                      "archived": co["archived"]})

    total_latency_ms = round(sum(t["latency_ms"] for t in trace), 1)
    logger.info("编排工作流执行完成 | fault=%s duration=%d 预警级别=%s 工单=%s 总时延=%.1fms",
                req.fault, req.duration, wa["level"],
                fh.get("order_id") if fh.get("handled") else "无", total_latency_ms)
    return {"code": 0, "data": {
        "workflow": "数据管理 -> 预警分析 -> 故障处置 -> 持续优化",
        "fault_injected": req.fault,
        "warning": wa,
        "work_order": fh,
        "optimization": co,
        "trace": trace,
        "total_latency_ms": total_latency_ms,
    }}


# ---------------- Web 界面 ----------------

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn

    c = get_settings()
    logger.info("启动 %s v%s | host=%s port=%d env=%s",
                c.app.title, c.app.version, c.app.host, c.app.port, c.app.env)
    uvicorn.run("server.api:app", host=c.app.host, port=c.app.port, reload=c.app.debug)
