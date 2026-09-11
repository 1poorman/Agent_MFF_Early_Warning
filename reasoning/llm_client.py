"""大模型接入抽象（OpenAI 兼容接口，可切换底座）。

配置来源（优先级从高到低）：
    1. 显式传入的 config dict
    2. 集中配置 config/settings.yaml 的 llm 段（可被 .env / 环境变量覆盖）
    3. 项目根 .env（YAML 风格 key: value，向后兼容）
优先使用 big_model（Qwen3.6-27B）做根因推理；推理模型的思考链在 reasoning 字段。
"""

from pathlib import Path
from typing import Optional

from openai import OpenAI


def load_env_config(path: str = ".env") -> dict:
    """解析 YAML 风格 .env（key: value）。"""
    cfg = {}
    p = Path(path)
    if not p.exists():
        return cfg
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def _load_central_or_env() -> dict:
    """优先集中配置（config 已合并 .env），无 url 时回退 .env。"""
    try:
        from config import get_settings
        s = get_settings()
        if s.llm.url:
            return s.llm.to_client_dict()
    except Exception:
        pass
    return load_env_config()


class LLMClient:
    """大模型客户端。

    prefer 选择端点：
      - "big"（默认）：url + big_model_name（Qwen3.6-27B，L3 根因推理主/兜底）
      - "front"：small_diag_url + small_diag_model（2B SFT 前置，MS6 级联，强制非思考）
      - 其它：base_url + small_model_name（旧小模型底座）
    """

    def __init__(self, config: Optional[dict] = None, prefer: str = "big",
                 timeout: Optional[float] = None,
                 enable_thinking: Optional[bool] = None):
        cfg = config or _load_central_or_env()
        # 思考模式/超时取集中配置（settings.yaml llm 段，可被 .env 覆盖）；
        # 显式传入的参数优先于配置。2B SFT 前置固定非思考（与训练/线上分布一致）。
        if enable_thinking is None:
            self.enable_thinking = False if prefer == "front" \
                else bool(cfg.get("enable_thinking", False))
        else:
            self.enable_thinking = enable_thinking
        if timeout is None:
            timeout = float(cfg.get("timeout") or 60.0)
        if prefer == "front" and cfg.get("small_diag_url"):
            base = cfg["small_diag_url"]
            base = base if base.startswith("http") else "http://" + base
            self.client = OpenAI(base_url=base, api_key=cfg.get("small_diag_key", "empty"), timeout=timeout)
            self.model = cfg.get("small_diag_model", "mff-sft-minicpm5-2b")
        elif prefer == "big" and cfg.get("url"):
            base = cfg["url"]
            base = base if base.startswith("http") else "http://" + base
            self.client = OpenAI(base_url=base, api_key=cfg.get("key", "empty"), timeout=timeout)
            self.model = cfg.get("big_model_name", "Qwen3.6-27B-INT4")
        else:
            base = cfg.get("base_url", "")
            base = base if base.startswith("http") else "http://" + base
            self.client = OpenAI(base_url=base, api_key=cfg.get("api_key", "empty"), timeout=timeout)
            self.model = cfg.get("small_model_name", "qwen3-14b-local")

    @classmethod
    def front(cls, config: Optional[dict] = None,
              timeout: Optional[float] = None) -> "LLMClient":
        """构造 2B SFT 前置客户端（级联，强制非思考）。"""
        return cls(config=config, prefer="front", timeout=timeout,
                   enable_thinking=False)

    def chat(self, prompt: str, system: str = "", max_tokens: int = 1500,
             temperature: float = 0.2) -> str:
        """对话推理，返回文本（兼容推理模型的 reasoning 字段）。

        enable_thinking=False 时通过 chat_template_kwargs 关闭思考模式（vLLM），
        大幅加速推理；输出直接进入 content。
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        extra = {}
        if not self.enable_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        r = self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
            extra_body=extra,
        )
        msg = r.choices[0].message
        # 兼容：思考模式关闭时输出在 content；开启时 content 可能为 None，思考在 reasoning
        content = getattr(msg, "content", None)
        if content:
            return content
        reasoning = getattr(msg, "reasoning", None)
        return reasoning or ""
