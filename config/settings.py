"""集中配置对象（dataclass），供各模块类型化访问。

用法：
    from config import get_settings
    cfg = get_settings()
    cfg.app.port          # 服务端口
    cfg.database.dsn      # 时序库 DSN
    cfg.simulator.seed    # 仿真器随机种子
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .loader import load_raw_config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------- 分片配置 ----------------

@dataclass
class AppSettings:
    name: str = "mff-early-warning"
    title: str = "中频炉水冷系统多参数融合预警智能体"
    version: str = "1.0.0"
    env: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


@dataclass
class MCPSettings:
    host: str = "0.0.0.0"
    port: int = 8100
    transport: str = "streamable-http"


@dataclass
class LoggingSettings:
    level: str = "INFO"
    dir: str = "logs"
    file: str = "app.log"
    error_file: str = "error.log"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 10
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    @property
    def log_dir(self) -> Path:
        d = Path(self.dir)
        return d if d.is_absolute() else _PROJECT_ROOT / d


@dataclass
class LLMSettings:
    url: str = ""
    key: str = ""
    big_model_name: str = "Qwen3.6-27B-INT4"
    small_model_name: str = "qwen3-14b-local"
    api_key: str = "empty"
    base_url: str = ""
    enable_thinking: bool = False
    timeout: float = 60.0
    max_tokens: int = 1500
    temperature: float = 0.2

    def to_client_dict(self) -> Dict[str, Any]:
        """转换为 reasoning.llm_client.LLMClient 接受的 config dict。"""
        return {
            "url": self.url, "key": self.key,
            "big_model_name": self.big_model_name,
            "small_model_name": self.small_model_name,
            "api_key": self.api_key, "base_url": self.base_url,
        }


@dataclass
class DatabaseSettings:
    dsn: str = "postgresql://postgres:postgres@localhost:5432/mff_tsdb"
    pool_min: int = 1
    pool_max: int = 5


@dataclass
class SimulatorSettings:
    """仿真器参数（字段与 simulator.SimConfig 对齐）。"""
    dt: float = 1.0
    seed: Optional[int] = 42
    start_time: str = "2026-08-20 00:00:00"
    furnace_heat_capacity: float = 4200.0
    furnace_eta: float = 0.72
    furnace_loss_coef: float = 0.12
    ambient_temp: float = 30.0
    rated_power: float = 3000.0
    power_band: float = 50.0
    line_voltage: float = 1000.0
    power_factor: float = 0.92
    pump_head: float = 0.30
    static_pressure: float = 0.12
    coil_resistance: float = 0.0022
    filter_resistance: float = 0.0006
    flow_tau: float = 8.0
    pipe_area: float = 3.318e-3
    rated_flow: float = 8.0
    max_delta_t: float = 60.0
    inlet_temp_base: float = 28.0
    inlet_temp_drift: float = 2.0
    coil_loss_frac: float = 0.08
    furnace_water_coef: float = 0.10
    water_cp: float = 4.186
    tank_area: float = 4.0
    tank_level_init: float = 2.0
    tank_level_low: float = 1.9
    tank_level_high: float = 2.1
    refill_rate: float = 8.0
    conductivity_init: float = 550.0
    conductivity_drift: float = 0.5
    cabinet_temp_coef: float = 0.004
    humidity_base: float = 50.0
    humidity_daily_amp: float = 8.0
    phase_schedule: Optional[List[Tuple[str, int, float]]] = field(default=None)


@dataclass
class RuleSettings:
    """L1 规则阈值（字段与 detection.rule_engine.RuleThresholds 对齐）。"""
    outlet_temp_high: float = 55.0
    inlet_temp_high_base: float = 35.0
    delta_t_high: float = 25.0
    pressure_low_kpa: float = 150.0
    pressure_high_kpa: float = 300.0
    flow_low_ratio: float = 0.8
    flow_high_ratio: float = 1.2
    rated_flow_lps: float = 8.0
    conductivity_high: float = 800.0
    dew_margin_warn_c: float = 3.0
    heat_rate_drift: float = 0.15
    pq_offset_limit: float = 0.10


@dataclass
class DetectionSettings:
    fast_window: int = 3600
    fast_horizon: int = 600
    fast_period: int = 8400
    fast_downsample: int = 10
    fast_hidden: int = 32
    anomaly_window: int = 60
    anomaly_threshold: float = 0.6
    anomaly_vae_weight: float = 0.5
    anomaly_latent: int = 8
    stream_l2_interval: int = 60
    stream_l3_min_realtime: int = 600
    forecast_threshold: float = 55.0
    forecast_horizon: int = 600
    exceed_lookback: int = 600


@dataclass
class QualitySettings:
    hampel_window_s: int = 121
    hampel_n_sigma: float = 5.0
    max_rate_per_s: Dict[str, float] = field(default_factory=dict)
    physical_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class PathSettings:
    models_dir: str = "models"
    data_dir: str = "data"
    feedback_file: str = "data/feedback/service.jsonl"

    def resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else _PROJECT_ROOT / path

    @property
    def models(self) -> Path:
        return self.resolve(self.models_dir)

    @property
    def data(self) -> Path:
        return self.resolve(self.data_dir)

    @property
    def feedback(self) -> Path:
        return self.resolve(self.feedback_file)


# ---------------- 总配置 ----------------

@dataclass
class Settings:
    app: AppSettings = field(default_factory=AppSettings)
    mcp: MCPSettings = field(default_factory=MCPSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    simulator: SimulatorSettings = field(default_factory=SimulatorSettings)
    rules: RuleSettings = field(default_factory=RuleSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    quality: QualitySettings = field(default_factory=QualitySettings)
    paths: PathSettings = field(default_factory=PathSettings)

    # ---------------- 构造 ----------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Settings":
        return cls(
            app=_fill(AppSettings(), d.get("app", {})),
            mcp=_fill(MCPSettings(), d.get("mcp", {})),
            logging=_fill(LoggingSettings(), d.get("logging", {})),
            llm=_fill(LLMSettings(), d.get("llm", {})),
            database=_fill(DatabaseSettings(), d.get("database", {})),
            simulator=_fill(SimulatorSettings(), d.get("simulator", {})),
            rules=_fill(RuleSettings(), d.get("rules", {})),
            detection=_fill(DetectionSettings(), d.get("detection", {})),
            quality=_fill(QualitySettings(), d.get("quality", {})),
            paths=_fill(PathSettings(), d.get("paths", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ---------------- 便捷构造器 ----------------

    def to_sim_config(self):
        """构造 simulator.SimConfig（惰性导入避免循环依赖）。"""
        from simulator.config import SimConfig
        kw = asdict(self.simulator)
        if kw.get("phase_schedule") is None:
            kw.pop("phase_schedule", None)
        return SimConfig(**kw)

    def to_rule_thresholds(self):
        """构造 detection.RuleThresholds。"""
        from detection.rule_engine import RuleThresholds
        return RuleThresholds(**asdict(self.rules))

    def to_db_config(self):
        """构造 storage.DBConfig。"""
        from storage.tsdb import DBConfig
        return DBConfig(dsn=self.database.dsn,
                        pool_min=self.database.pool_min,
                        pool_max=self.database.pool_max)


def _fill(obj: Any, data: Dict[str, Any]) -> Any:
    """用 dict 覆盖 dataclass 中已存在的字段（未知键忽略，类型不强转）。"""
    if not data:
        return obj
    for k, v in data.items():
        if hasattr(obj, k) and v is not None:
            setattr(obj, k, v)
    return obj


# ---------------- 单例 ----------------

_settings: Optional[Settings] = None


def load_settings(reload: bool = False) -> Settings:
    """从 config/settings.yaml + .env + 环境变量加载集中配置。"""
    raw = load_raw_config(root=_PROJECT_ROOT)
    return Settings.from_dict(raw)


def get_settings(reload: bool = False) -> Settings:
    """获取全局单例配置。"""
    global _settings
    if _settings is None or reload:
        _settings = load_settings(reload=reload)
    return _settings
