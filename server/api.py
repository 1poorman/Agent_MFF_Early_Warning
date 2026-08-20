"""RESTful API 服务（OpenAPI 3.0）。

启动：
    uvicorn server.api:app --host 0.0.0.0 --port 8000
文档：
    http://localhost:8000/docs  (Swagger UI)
界面：
    http://localhost:8000/      (Web Demo)
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .service import AgentService

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "server" / "static"

app = FastAPI(
    title="中频炉水冷系统预警智能体 API",
    version="1.0.0",
    description="多级预警 + 根因诊断 + 工单闭环",
)

_service: Optional[AgentService] = None


def svc() -> AgentService:
    global _service
    if _service is None:
        _service = AgentService.get()
    return _service


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
    try:
        req_raw = await ws.receive_json()
        fault = req_raw.get("fault")
        fault_start = int(req_raw.get("fault_start", 120))
        speed = float(req_raw.get("speed", 20.0))
        duration = int(req_raw.get("duration", 1800))

        from simulator import DataSimulator, FaultSpec, SimConfig
        faults = [FaultSpec(name=fault, start=fault_start, ramp=600, severity=0.9)] if fault else []
        sim = DataSimulator(config=SimConfig(seed=42), faults=faults)

        interval = 1.0 / max(speed, 1.0)
        loop = asyncio.get_event_loop()
        for _ in range(duration):
            row = sim._sense(sim.step())   # 带测量噪声的传感器读数
            row["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            # stream_step 含潜在 LLM 同步推理，放入线程池避免阻塞事件循环（WS keepalive）
            out = await loop.run_in_executor(None, s.stream_step, row)
            await ws.send_json(out)
            await asyncio.sleep(interval)
        await ws.send_json({"done": True})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


@app.get("/api/v1/stream/logs")
def stream_logs():
    return {"code": 0, "data": svc().get_stream_logs()}


# ---------------- Web 界面 ----------------

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
