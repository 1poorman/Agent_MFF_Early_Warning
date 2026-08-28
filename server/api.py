"""RESTful API 服务（OpenAPI 3.0）。

启动：
    uvicorn server.api:app --host 0.0.0.0 --port 8000
    # 或：python -m server.api（host/port 从 config/settings.yaml 读取）
文档：
    http://localhost:8000/docs  (Swagger UI)
界面：
    http://localhost:8000/      (Web Demo)
MCP（挂载于同一端口）：
    http://localhost:8000/mcp   (streamable-http 端点)
"""

import asyncio
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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


def _fill_columns(df: pd.DataFrame) -> pd.DataFrame:
    """按数据契约补全缺失列（缺列填 NaN 交质量管控插补），与文档"缺列自动补"一致。"""
    from perception.ingest import ALL_COLUMNS
    for c in ALL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[ALL_COLUMNS]

app = FastAPI(
    title=cfg.app.title,
    version=cfg.app.version,
    description="""
中频炉水冷系统多参数融合预警智能体 —— 面向工业场景的多级预警 + 根因诊断 + 工单闭环。

**能力架构（四大智能体）**：
1. **数据管理智能体**：传感器数据接收 / 采集接口 / 预处理（缺失插补、异常剔除）
2. **预警分析智能体**：L1 规则预警 → L2 异常检测与时序预测（GRU/PatchTST/TimesFM 可切换）→ L3 大模型根因诊断（注入知识图谱、维修工单、工况表上下文）
3. **故障处置智能体**：自动生成工单、联动应急预案、分级预警通知（声光/短信/电话/APP）
4. **持续优化智能体**：处置反馈归档、训练样本积累、知识库持续更新

**快速体验**：
- 界面：`/`（单屏四智能体 Web Demo，含实时流与文件上传）
- 一键演示：`POST /api/v1/workflow/run`
- 上传数据：`POST /api/v1/upload`（支持 data/simulated/*.csv 同格式）
- 在线预览：https://revolt-yahoo-doing.ngrok-free.dev/docs
""",
    openapi_tags=[
        {"name": "健康检查", "description": "服务状态与运行统计"},
        {"name": "数据管理智能体", "description": "传感器采集 / 数据接入预处理 / 数据格式契约 / 文件上传 / 时序库"},
        {"name": "预警分析智能体", "description": "L1 规则预警 / L2 异常检测与预测 / L3 根因诊断 / 最近窗口"},
        {"name": "故障处置智能体", "description": "工单生成 + 应急预案 + 分级通知"},
        {"name": "持续优化智能体", "description": "处置反馈归档 / 知识库更新 / 优化状态"},
        {"name": "编排工作流", "description": "四大智能体一键串联演示 / 智能体清单"},
        {"name": "实时流", "description": "WebSocket 实时监测与预警日志"},
    ],
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
    """数据接入请求：原始传感器记录数组。"""
    records: List[Dict] = Field(..., description="原始传感器记录数组，字段见 GET /api/v1/agents/data-manager/schema 数据契约；timestamp 为必填列，其余缺列自动补 NaN")


class DiagnoseRequest(BaseModel):
    """L3 根因诊断请求。"""
    features: Dict[str, float] = Field(..., description="异常特征字典，键为传感器名、值为读数，如 {\"outlet_temp\": 56.2, \"pressure\": 175.0, \"flow_rate\": 6.1}")
    condition: str = Field("unknown", description="当前工况：startup / melting / holding / tapping / idle")
    sensor_names: Optional[List[str]] = Field(None, description="参与诊断的传感器名列表，如 [\"outlet_temp\", \"pressure\"]")
    l1_alerts: Optional[List[Dict]] = Field(None, description="L1 规则预警记录（由 /warn/l1 返回），供诊断参考")
    l2_forecast: Optional[Dict] = Field(None, description="L2 异常分/趋势预测结果，供诊断参考")


class FeedbackRequest(BaseModel):
    """处置反馈归档请求（持续优化）。"""
    order_id: str = Field(..., description="工单号，如 WO-20260821-0007")
    actual_root_cause: str = Field(..., description="实际根因，如 管道泄漏 / 过滤器堵塞")
    is_true_fault: bool = Field(..., description="是否为真实故障（true/false）")
    handling_time_min: float = Field(..., description="处置耗时（分钟）")
    effect: str = Field(..., description="处置效果描述")


class SimulateRequest(BaseModel):
    """仿真回放请求（Demo 演示用）。"""
    duration: int = Field(300, description="回放时长（秒），1–86400")
    fault: Optional[str] = Field(None, description="注入故障：filter_clog（过滤器堵塞）/ pump_cavitation（水泵气蚀）/ pipe_leak（管道泄漏）/ scale_buildup（线圈结垢）")
    fault_start: int = Field(120, description="故障起始时间（秒），相对回放起点")


# ---------------- 接口 ----------------

@app.get("/api/v1/health", tags=["健康检查"],
         summary="健康检查：服务状态与运行统计",
         description="返回服务是否就绪、LLM 是否可用、已加载的预测模型列表与累计预警/工单/反馈统计，用于探活与部署核验。")
def health():
    s = svc()
    return {"code": 0, "data": {
        "status": "ok",
        "use_llm": s.use_llm,
        "fast_models": list(s.pipeline.fast_models.keys()),
        "anomaly_detector": s.pipeline.detector is not None,
        "stats": s.pipeline.get_stats(),
    }}


@app.post("/api/v1/data/ingest", tags=["数据管理智能体"],
         summary="数据接入与质量管控",
         description="接收原始传感器记录，执行质量管控（时间戳对齐 / 缺失插补 / 异常剔除），返回可用条数与质量报告。")
def ingest(req: IngestRequest):
    df = _fill_columns(pd.DataFrame(req.records))
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


@app.post("/api/v1/warn/l1", tags=["预警分析智能体"],
         summary="L1 规则预警（15 条规则）",
         description="对输入记录执行 L1 规则引擎评估，返回命中的规则预警（出水温度超限 / 压差超限 / 流量异常 / 电导率超标 / 凝露风险 / 组合规则等）。")
def warn_l1(req: IngestRequest):
    df = _fill_columns(pd.DataFrame(req.records))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    clean, _ = svc().qc.process(df)
    alerts = svc().rule_engine.evaluate(clean)
    return {"code": 0, "data": {"count": len(alerts), "alerts": alerts.to_dict("records")}}


@app.post("/api/v1/diagnose", tags=["预警分析智能体"],
         summary="L3 根因诊断（大模型 + 知识图谱）",
         description="输入异常特征与工况，执行知识图谱召回 → LLM 多跳推理（CoT）→ 三层防幻觉校验 → 置信度门控，返回根因、置信度、证据链、处置 SOP。LLM 不可用时自动降级图谱/数值兜底。")
def diagnose(req: DiagnoseRequest):
    diag = svc().diagnose(req.features, req.condition, req.sensor_names,
                          req.l1_alerts, req.l2_forecast)
    return {"code": 0, "data": diag.to_dict()}


@app.post("/api/v1/workorder", tags=["故障处置智能体"],
         summary="工单生成（含应急预案与分级通知）",
         description="在 L3 诊断基础上自动生成标准化运维工单，红色预警联动应急预案，并按级别完成分级预警通知（声光/短信/电话/APP）。")
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


@app.post("/api/v1/feedback", tags=["持续优化智能体"],
         summary="处置反馈归档（持续优化）",
         description="归档处置反馈（工单号/实际根因/处置耗时/效果），积累真实故障样本触发模型迭代，并按需更新知识库。")
def feedback(req: FeedbackRequest):
    stats = svc().submit_feedback(req.order_id, req.actual_root_cause,
                                  req.is_true_fault, req.handling_time_min, req.effect)
    return {"code": 0, "data": stats}


@app.post("/api/v1/simulate", tags=["数据管理智能体"],
         summary="仿真回放端到端（Demo 演示）",
         description="回放物理机理仿真器数据驱动端到端流程（采集 → 预警 → 处置），返回预警级别、异常分、工单与各环节耗时，用于无传感器时的功能演示。")
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


@app.get("/api/v1/latest", tags=["预警分析智能体"],
         summary="最近窗口结果与绘图数据",
         description="返回最近一次窗口分析结果（预警级别/异常分/工单）与出水温度/压力/流量/湿度绘图序列，供界面展示。")
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

@app.get("/api/v1/forecast-model", tags=["预警分析智能体"],
         summary="查询 L2 预测模型",
         description="返回当前 L2 时序预测模型（gru / patchtst / timesfm）、可用模型列表与预测步长，供前端下拉框同步。")
async def get_forecast_model():
    """获取当前 L2 预测模型与可用模型列表。"""
    s = svc()
    engine = getattr(s.pipeline, "forecast_engine", None)
    return {"code": 0, "data": {
        "current": engine.name if engine else "gru",
        "available": ["gru", "patchtst", "timesfm"],
        "horizon_s": engine.horizon_s if engine else 600,
    }}


@app.post("/api/v1/upload", tags=["数据管理智能体"],
         summary="上传时序数据文件（选定监测数据源）",
         description="""
上传时序数据文件（CSV / JSON，格式同 data/simulated/*.csv 数据契约）。
上传成功即**选定该文件作为监测数据源**（等同选定故障场景），仅返回解析信息与曲线预览，
**不自动诊断**；点击"开始监测"后从该数据源逐条回放，运行 数据管理 → 预警分析 → 故障处置 全链路。

**输入**：
- `file`：CSV/JSON 文件，需含 timestamp 列，其余列按数据契约（缺列自动补 NaN 交质量管控插补）
- `model`：L2 预测模型，可选 gru / patchtst / timesfm

**输出**：文件名、条数、质量报告、绘图抽样序列。
""")
async def upload_analyze(file: UploadFile = File(..., description="时序数据文件（CSV/JSON，需含 timestamp 列）"),
                         model: str = Form("gru", description="L2 预测模型：gru / patchtst / timesfm")):
    """上传时序数据文件 -> 缓存为监测数据源（不自动诊断，点开始监测才运行四大智能体）。

    CSV 列契约见 perception/ingest.py ALL_COLUMNS；
    缺列自动补 NaN（质量管控插补），timestamp 为必填列。
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

    s = svc()
    # 数据管理智能体：质量管控（此时仅预处理，不触发预警分析）
    dm = DataManagementAgent(s).ingest(df.to_dict("records"))
    # 缓存为监测数据源：点击"开始监测"后回放此数据运行四大智能体
    s.set_uploaded_data(dm["records"], {"filename": file.filename,
                                        "points": len(dm["records"]),
                                        "quality": dm["quality"]})
    logger.info("上传数据源就绪（不自动诊断）| %s 行=%d",
                file.filename, len(dm["records"]))

    # 绘图用序列（等间隔抽样，最多 2000 点）——全部 12 个数值传感器
    step = max(1, len(df) // 2000)
    sub = df.iloc[::step]
    series = {"timestamp": sub["timestamp"].astype(str).tolist()}
    for c in ["inlet_temp", "outlet_temp", "pressure", "flow_rate", "flow_velocity",
              "tank_level", "conductivity", "cabinet_temp", "cabinet_humidity",
              "furnace_temp", "electric_power", "electric_current"]:
        series[c] = sub[c].round(1).fillna(0).tolist()
    return {"code": 0, "data": {
        "filename": file.filename,
        "points": len(dm["records"]),
        "quality": dm["quality"],
        "series": series,
        "hint": "数据源已就绪，点击「开始监测」运行四大智能体",
    }}


# ---------------- 实时流（WebSocket） ----------------

class StreamRequest(BaseModel):
    """WebSocket 实时流参数（连接后发送的首条 JSON）。"""
    fault: Optional[str] = Field(None, description="注入故障：filter_clog / pump_cavitation / pipe_leak / scale_buildup")
    fault_start: int = Field(120, description="故障起始时间（秒），相对实时流起点")
    speed: float = Field(20.0, description="回放倍速（条/秒），1–100")
    duration: int = Field(1800, description="监测时长（秒）")
    forecast_model: str = Field("gru", description="L2 预测模型：gru / patchtst / timesfm")
    data_source: str = Field("simulator", description="数据源：simulator（物理仿真生成）/ upload（回放已上传时序数据）")


async def _drain_and_done(ws: WebSocket, s, extra: Optional[Dict] = None,
                          timeout_s: float = 90.0, interval_s: float = 0.5):
    """流尾兜底：等待后台 L3/工单/反馈完成并在连接内推送，避免结果落在断连之后。

    LLM 推理较慢时，_diagnose_async 的结果可能在最后一条实时数据之后才产出；
    若直接发送 done 断开连接，诊断结果将永远无法到达前端（L3/③/④ 界面不同步）。
    此处在 done 前持续监听 pending_*，一旦有新结果立即按与实时帧相同的结构
    下发（前端无需任何改动即可渲染），超时或无残留则正常收尾。
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if s.pending_l3 is not None:
            # 诊断对象自带 timestamp（构造时写入），与 /stream/logs 载荷一致，
            # 保证前端跨通道去重生效
            await ws.send_json({
                "timestamp": s.pending_l3.get("timestamp", ""),
                "metrics": {},
                "l1": [],
                "l2": {"anomaly_score": 0.0, "exceed_eta": None, "forecast": None},
                "l3": s.pending_l3,
                "work_order": s.pending_wo,
                "optimization": s.pending_co,
            })
            s.pending_l3 = s.pending_wo = s.pending_co = None
        if s.pending_l3 is None and not getattr(s, "_diag_inflight", False):
            break
        await asyncio.sleep(interval_s)
    payload = {"done": True}
    if extra:
        payload.update(extra)
    await ws.send_json(payload)


@app.websocket("/ws/stream", name="实时流")
async def stream_ws(ws: WebSocket):
    """实时流：simulator 逐秒生成数据 -> 逐条处理 -> 推送指标与 L1/L2/L3 预警。

    连接后先发送参数 JSON（fault/speed/duration/fault_start/forecast_model），
    服务端按所选模型窗口预加载（回 preload/preload_done），随后逐条推送分析结果，结束回 done。
    """
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
        data_source = str(req_raw.get("data_source") or "simulator").lower()
        logger.info("实时流参数 | fault=%s fault_start=%d speed=%.1f duration=%d "
                    "forecast_model=%s data_source=%s",
                    fault, fault_start, speed, duration, forecast_model, data_source)

        # L2 预测模型运行时切换（GRU/PatchTST/TimesFM）
        engine = getattr(s.pipeline, "forecast_engine", None)
        if engine is not None and forecast_model in ("gru", "patchtst", "timesfm"):
            engine.switch(forecast_model, fast_models=s.pipeline.fast_models,
                          models_dir=str(s.cfg.paths.models))
            logger.info("实时流预测模型: %s", engine.name)
        model_name = engine.name if engine else "gru"

        loop = asyncio.get_event_loop()

        # ---- 数据源：upload（回放已上传时序数据，运行四大智能体） ----
        if data_source == "upload":
            rows = s.uploaded_data or []
            if not rows:
                # 服务端上传缓存为空（如服务重启清空内存态）：发错误+结束帧，
                # 让前端立即恢复按钮状态而不是挂着无输出
                await ws.send_json({"error": "未找到已上传数据（服务可能已重启），请重新上传"})
                await ws.send_json({"done": True, "points": 0, "data_source": "upload",
                                    "hint": "uploaded_data_missing"})
                logger.warning("回放请求缺少已上传数据（服务重启导致缓存丢失）")
                return
            logger.info("回放上传数据源 | %s 条=%d", s.uploaded_meta.get("filename", ""), len(rows))
            # 上传路径无服务端预加载段：回 preload+preload_done 让前端提示切换为
            # "回放中 + 预测启用条件"（缺失会导致界面永远停留在"预加载历史数据"）
            forecast_ready_at = engine.backend_window() if engine is not None else 0
            await ws.send_json({"preload": True, "points": 0, "forecast_model": model_name,
                                "data_source": "upload",
                                "forecast_ready_at": forecast_ready_at,
                                "filename": s.uploaded_meta.get("filename", "")})
            await ws.send_json({"preload_done": True, "points": s.preloaded,
                                "forecast_model": model_name, "data_source": "upload"})
            # 从头逐条回放：曲线从第 1 条开始实时显示（不跳过预加载段）；
            # stream_step 内部按窗口就绪才产出 L2 预测/L3 诊断，曲线始终连续
            interval = 1.0 / max(speed, 1.0)
            pushed = 0
            for row in rows:
                out = await loop.run_in_executor(None, s.stream_step, dict(row))
                await ws.send_json(out)
                pushed += 1
                await asyncio.sleep(interval)
            await _drain_and_done(ws, s, {"points": pushed, "data_source": "upload"})
            logger.info("上传数据回放完成 | 分析 %d 条", pushed)
            return

        # ---- 数据源：simulator（物理仿真生成） ----
        from simulator import DataSimulator, FaultSpec, SimConfig
        # 预加载量 = 当前预测后端窗口（GRU 16800s=4.6h / PatchTST 7200s / TimesFM 1024s）
        warmup_s = engine.backend_window() if engine is not None else 16800
        # 容错：无效故障名（如前端误传 "custom"，常见于上传未就绪即点开始监测）
        # 不再抛异常断流，降级为正常工况运行并提示
        faults = []
        if fault:
            from simulator.faults import FAULT_REGISTRY
            if fault in FAULT_REGISTRY:
                faults = [FaultSpec(name=fault, start=warmup_s + fault_start,
                                    ramp=300, severity=0.9)]
            else:
                logger.warning("忽略未知故障类型 %r（按正常工况运行）", fault)
                await ws.send_json({
                    "notice": f"已忽略无效故障场景 {fault}，按正常工况运行"})
        sim = DataSimulator(config=SimConfig(seed=42), faults=faults)

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
        await _drain_and_done(ws, s)
        logger.info("实时流完成 | 推送 %d 条", duration)
    except WebSocketDisconnect:
        logger.info("实时流客户端断开 | client=%s", ws.client.host if ws.client else "unknown")
    except Exception as e:
        logger.exception("实时流异常: %s", e)
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


@app.get("/api/v1/stream/logs", tags=["实时流"],
         summary="实时流预警日志",
         description="返回最近一次实时流的 L1/L2/L3 预警日志记录（含时间、规则、消息），供界面展示。")
def stream_logs():
    return {"code": 0, "data": svc().get_stream_logs()}


@app.post("/api/v1/stream/reset", tags=["实时流"],
         summary="重置实时流缓冲",
         description="清空实时流缓冲与预警日志，开始新一轮监测（新一轮 WebSocket 连接时自动执行）。")
def stream_reset():
    """重置实时流缓冲与日志（新一次监测）。"""
    svc().reset_stream()
    return {"code": 0, "data": {"reset": True}}


class TsQueryRequest(BaseModel):
    """时序数据库历史查询请求。"""
    start: str = Field(..., description="起始时间，格式 2026-08-21 00:00:00")
    end: str = Field(..., description="结束时间，格式 2026-08-21 06:00:00")
    condition: Optional[str] = Field(None, description="按工况过滤：startup/melting/holding/tapping/idle")
    fault: Optional[str] = Field(None, description="按故障过滤：filter_clog/pump_cavitation/pipe_leak/scale_buildup")
    columns: Optional[List[str]] = Field(None, description="返回列，如 [\"outlet_temp\", \"pressure\"]；缺省返回全部")


@app.get("/api/v1/tsdb/stats", tags=["数据管理智能体"],
         summary="时序库存储统计",
         description="返回 PostgreSQL 时序库月度分区数、各分区记录数；数据库未连接时返回 enabled=false 与原因。")
def tsdb_stats():
    """时序数据库存储统计（分区/记录数）。"""
    s = svc()
    if not getattr(s, "tsdb_ok", False):
        return {"code": 0, "data": {"enabled": False, "reason": "数据库未连接"}}
    try:
        return {"code": 0, "data": {"enabled": True, **s.tsdb.stats()}}
    except Exception as e:
        return {"code": 0, "data": {"enabled": False, "reason": str(e)}}


@app.post("/api/v1/tsdb/query", tags=["数据管理智能体"],
         summary="时序库历史查询",
         description="按时间区间（可选工况/故障过滤）从 PostgreSQL 时序库读取历史数据，返回记录数组。")
def tsdb_query(req: TsQueryRequest):
    """从时序数据库读取历史数据（区间/工况/故障过滤）。"""
    s = svc()
    if not getattr(s, "tsdb_ok", False):
        raise HTTPException(status_code=503, detail="时序数据库未连接")
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
    """传感器数据采集请求（数据管理智能体）。"""
    duration: int = Field(300, description="采集时长（秒），1–86400")
    fault: Optional[str] = Field(None, description="注入故障：filter_clog / pump_cavitation / pipe_leak / scale_buildup")
    fault_start: int = Field(120, description="故障起始时间（秒）")
    severity: float = Field(0.9, description="故障严重度 0–1（1 为最严重）")


class AnalyzeRequest(BaseModel):
    """多级预警分析请求（预警分析智能体）。"""
    records: List[Dict] = Field(..., description="数据管理智能体返回的规整数据（见数据契约）")


class HandleRequest(BaseModel):
    """故障处置请求（故障处置智能体）。"""
    analysis: Dict = Field(..., description="预警分析智能体返回结果（含 level、l3 根因等）")


class FeedbackRequest2(BaseModel):
    """处置反馈归档请求（持续优化智能体）。"""
    order_id: str = Field(..., description="工单号，如 WO-20260821-0007")
    actual_root_cause: str = Field(..., description="实际根因")
    is_true_fault: bool = Field(..., description="是否为真实故障")
    handling_time_min: float = Field(..., description="处置耗时（分钟）")
    effect: str = Field(..., description="处置效果")
    work_order: Optional[Dict] = Field(None, description="工单详情（可选）")


class KnowledgeRequest(BaseModel):
    """知识库更新请求（持续优化智能体）。"""
    component: str = Field(..., description="部件，如 管道 / 过滤器 / 水泵 / 线圈")
    action: str = Field(..., description="处置动作，如 更换3号阀后密封圈")
    order_id: Optional[str] = Field(None, description="关联工单号（可选）")
    date: Optional[str] = Field(None, description="日期，格式 2026-08-21")
    note: str = Field("", description="备注")


class WorkflowRunRequest(BaseModel):
    """编排工作流一键演示请求。"""
    duration: int = Field(600, description="数据时长（秒）")
    fault: Optional[str] = Field("pipe_leak", description="注入故障：filter_clog / pump_cavitation / pipe_leak / scale_buildup")
    fault_start: int = Field(180, description="故障起始时间（秒）")
    severity: float = Field(0.9, description="故障严重度 0–1")
    simulate_feedback: bool = Field(True, description="是否模拟处置反馈（触发持续优化）")


@app.get("/api/v1/agents", tags=["编排工作流"],
         summary="四大智能体清单",
         description="返回数据管理 / 预警分析 / 故障处置 / 持续优化四个智能体的名称、职责与关联端点。")
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

@app.post("/api/v1/agents/data-manager/collect", tags=["数据管理智能体"],
         summary="传感器数据采集",
         description="传感器数据采集接口（预留 Modbus/OPC UA/MQTT/RTSP 真实协议接入，当前由物理机理仿真器供数，支持注入 4 类故障），返回规整后的采集数据。")
def dm_collect(req: CollectRequest):
    """传感器数据采集（预留 Modbus/OPC UA/MQTT/RTSP 接口，当前由仿真器供数）。"""
    return {"code": 0, "data": agents()["data_manager"].collect(
        req.duration, req.fault, req.fault_start, req.severity)}


@app.post("/api/v1/agents/data-manager/ingest", tags=["数据管理智能体"],
         summary="数据接入预处理",
         description="接收原始传感器记录并预处理（时间戳对齐 / 缺失插补 / 异常剔除），返回 L1/L2 可直接使用的数据格式与质量报告。")
def dm_ingest(req: IngestRequest):
    """接收原始传感器数据 -> 预处理（对齐/插补/剔除）-> L1/L2 可直接使用格式。"""
    try:
        return {"code": 0, "data": agents()["data_manager"].ingest(req.records)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"数据接入处理异常: {e}")


@app.get("/api/v1/agents/data-manager/schema", tags=["数据管理智能体"],
         summary="数据格式契约",
         description="查询 L1/L2 可直接使用的数据格式契约（字段 / 单位 / 精度），供上传文件与第三方数据接入参考。")
def dm_schema():
    """L1/L2 可直接使用的数据格式契约。"""
    return {"code": 0, "data": agents()["data_manager"].schema()}


# ---- 2. 预警分析智能体 ----

@app.post("/api/v1/agents/warning-analyzer/analyze", tags=["预警分析智能体"],
         summary="多级预警分析（L1/L2/L3）",
         description="多级预警分析：L1 规则 → L2 异常检测与趋势预测（GRU/PatchTST/TimesFM）→ L3 大模型根因诊断，自动注入知识图谱 / 维修工单 / 工况表上下文，返回预警级别、异常分、根因与上下文。")
def wa_analyze(req: AnalyzeRequest):
    """多级预警分析：L1 规则 -> L2 异常/趋势 -> L3 根因诊断（注入知识图谱/知识库/工况表上下文）。"""
    try:
        return {"code": 0, "data": agents()["warning_analyzer"].analyze(req.records)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"预警分析异常: {e}")


# ---- 3. 故障处置智能体 ----

@app.post("/api/v1/agents/fault-handler/handle", tags=["故障处置智能体"],
         summary="故障处置（工单 + 应急 + 通知）",
         description="接收预警分析结果，生成标准化运维工单（含 SOP / 备件）、红色预警联动应急预案，并完成分级预警通知（声光/短信/电话/APP）。")
def fh_handle(req: HandleRequest):
    """工单生成 + 应急预案联动 + 分级预警通知。"""
    return {"code": 0, "data": agents()["fault_handler"].handle(req.analysis)}


# ---- 4. 持续优化智能体 ----

@app.post("/api/v1/agents/optimizer/feedback", tags=["持续优化智能体"],
         summary="处置反馈归档",
         description="归档处置反馈（工单号 / 实际根因 / 处置耗时 / 效果），积累真实故障样本触发模型迭代，并按需更新知识库。")
def co_feedback(req: FeedbackRequest2):
    """处置反馈归档 -> 训练样本积累 -> 知识库按需更新。"""
    return {"code": 0, "data": agents()["optimizer"].feedback(
        req.order_id, req.actual_root_cause, req.is_true_fault,
        req.handling_time_min, req.effect, req.work_order)}


@app.post("/api/v1/agents/optimizer/update-knowledge", tags=["持续优化智能体"],
         summary="更新知识库",
         description="手动更新知识库（新增维修工单记录），新记录将作为后续 L3 诊断的上下文参与推理。")
def co_update_knowledge(req: KnowledgeRequest):
    """手动更新知识库（新增维修工单记录）。"""
    return {"code": 0, "data": agents()["optimizer"].update_knowledge(
        {"order_id": req.order_id, "date": req.date,
         "component": req.component, "action": req.action, "note": req.note})}


@app.get("/api/v1/agents/optimizer/status", tags=["持续优化智能体"],
         summary="持续优化状态",
         description="查询持续优化状态：反馈统计（总数/真实故障/误报）、模型微调触发情况与知识库规模。")
def co_status():
    """持续优化状态：反馈统计/微调触发/知识库规模。"""
    return {"code": 0, "data": agents()["optimizer"].status()}


# ---- 编排工作流（一键演示） ----

@app.post("/api/v1/workflow/run", tags=["编排工作流"],
         summary="编排工作流一键演示",
         description="四大智能体一键串联：数据采集 → 预警分析 → 故障处置 → 持续优化，返回全链路结果与各环节耗时（端到端约 5s，其中 LLM 推理 3~5s）。")
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
                  "anomaly_score": wa["l2"]["anomaly_score"],
                  "l2_trend_exceed": wa["l2"]["trend_exceed"],
                  "l2_forecast_end": (wa["l2"]["forecast"] or {}).get("end_value")})

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


# ---------------- MCP（streamable-http）挂载 ----------------
# 将四大智能体 MCP 能力挂载到 /mcp，与 API/Web 共用同一端口（8000），
# 公网只需映射一个端口（如 9000->8000）即可同时访问 API 与 MCP。
# 端点：http://<host>:8000/mcp
# 说明：streamable_http_app 自带 lifespan（启动 session_manager 的 task group），
# 作为子应用挂载后父应用不会自动触发其 lifespan，需手动并入，否则请求会报
# "Task group is not initialized"。
try:
    from .mcp_server import mcp as _mcp
    _mcp_app = _mcp.streamable_http_app(streamable_http_path="/")
    _mcp_lifespan = _mcp_app.router.lifespan_context
    _orig_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _combined_lifespan(app_self):
        async with _mcp_lifespan(app_self):
            async with _orig_lifespan(app_self):
                yield

    app.router.lifespan_context = _combined_lifespan
    app.mount("/mcp", _mcp_app, name="mcp")
    logger.info("MCP streamable-http 已挂载至 /mcp（与 API 共用端口 %s）", cfg.app.port)
except Exception as _e:  # 挂载失败不影响 API 本身
    logger.warning("MCP 挂载至 /mcp 失败（API 仍可用）: %s", _e)

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn

    c = get_settings()
    logger.info("启动 %s v%s | host=%s port=%d env=%s",
                c.app.title, c.app.version, c.app.host, c.app.port, c.app.env)
    uvicorn.run("server.api:app", host=c.app.host, port=c.app.port, reload=c.app.debug)
