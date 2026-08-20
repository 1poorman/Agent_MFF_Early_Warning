"""大模型接入抽象（OpenAI 兼容接口，可切换底座）。

从 .env 读取配置（YAML 风格 key: value）。
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


class LLMClient:
    """大模型客户端。"""

    def __init__(self, config: Optional[dict] = None, prefer: str = "big"):
        cfg = config or load_env_config()
        if prefer == "big" and cfg.get("url"):
            base = cfg["url"]
            base = base if base.startswith("http") else "http://" + base
            self.client = OpenAI(base_url=base, api_key=cfg.get("key", "empty"))
            self.model = cfg.get("big_model_name", "Qwen3.6-27B-INT4")
        else:
            base = cfg.get("base_url", "")
            base = base if base.startswith("http") else "http://" + base
            self.client = OpenAI(base_url=base, api_key=cfg.get("api_key", "empty"))
            self.model = cfg.get("small_model_name", "qwen3-14b-local")

    def chat(self, prompt: str, system: str = "", max_tokens: int = 1500,
             temperature: float = 0.2) -> str:
        """对话推理，返回文本（兼容推理模型的 reasoning 字段）。"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        msg = r.choices[0].message
        # 推理模型：content 可能为 None，思考在 reasoning；优先 content，空则取 reasoning
        content = getattr(msg, "content", None)
        if content:
            return content
        reasoning = getattr(msg, "reasoning", None)
        return reasoning or ""
