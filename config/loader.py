"""集中配置加载器（轻量 YAML 子集解析 + 环境变量覆盖）。

无第三方依赖（不依赖 pyyaml），支持以下格式：

    # 注释
    app:
      host: 0.0.0.0
      port: 8000
      tags: [dev, local]        # 内联列表
    empty_value:

优先级（低 -> 高）：
    1. config/settings.yaml 内置默认值
    2. 项目根目录 .env（仅 llm 段，保留原有大模型配置习惯）
    3. 环境变量 MFF_<SECTION>_<KEY>（如 MFF_APP_PORT=8080）
"""

import os
import re
from pathlib import Path
from typing import Any, Dict

# 环境变量前缀
ENV_PREFIX = "MFF_"
# .env 中归属 llm 段的键（兼容 reasoning/llm_client.py 的 key: value 风格）
LLM_ENV_KEYS = {"url", "key", "big_model_name", "small_model_name",
                "api_key", "base_url"}

_BOOL_TRUE = {"true", "yes", "on", "1"}
_BOOL_FALSE = {"false", "no", "off", "0"}
_NULL = {"null", "none", "~", ""}


def _strip_comment(line: str) -> str:
    """去掉行尾注释（# 前必须为空白或行首）。"""
    for i, ch in enumerate(line):
        if ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _parse_scalar(raw: str) -> Any:
    """字符串 -> bool/int/float/内联列表/null/字符串。"""
    v = raw.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x) for x in inner.split(",")]
    low = v.lower()
    if low in _NULL:
        return None
    if low in _BOOL_TRUE:
        return True
    if low in _BOOL_FALSE:
        return False
    # 引号字符串原样返回
    if (v.startswith('"') and v.endswith('"')) or \
            (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    # 数字
    if re.fullmatch(r"[-+]?\d+", v):
        return int(v)
    if re.fullmatch(r"[-+]?\d*\.\d+([eE][-+]?\d+)?", v):
        return float(v)
    return v


def parse_yaml_lite(text: str) -> Dict[str, Any]:
    """解析轻量 YAML（缩进嵌套 + key: value + 内联列表）。返回嵌套 dict。"""
    root: Dict[str, Any] = {}
    # 栈：每层 (缩进, dict 容器)
    stack: list = [(-1, root)]
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if ":" not in content:
            continue
        key, _, value = content.partition(":")
        key = key.strip()
        value = value.strip()

        # 弹出缩进不小于当前的层
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not stack:
            stack.append((-1, root))
        parent = stack[-1][1]

        if value == "":
            # 进入新的子 section
            node: Dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """递归合并 dict（override 优先）。"""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_env_llm(env_path: str = ".env") -> Dict[str, Any]:
    """读取 YAML 风格 .env 中的 llm 段（url/key/big_model_name 等）。"""
    cfg: Dict[str, Any] = {}
    p = Path(env_path)
    if not p.is_file():          # is_file 兼容"挂载了同名目录"的边界情况
        return cfg
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line or line.startswith("#"):
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k in LLM_ENV_KEYS:
            cfg[k] = v
    return cfg


def _load_env_overrides() -> Dict[str, Any]:
    """环境变量 MFF_<SECTION>_<KEY> 覆盖，如 MFF_APP_PORT=8080、MFF_LLM_KEY=xxx。"""
    out: Dict[str, Any] = {}
    for name, val in os.environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        parts = name[len(ENV_PREFIX):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, key = parts
        out.setdefault(section, {})[key] = _parse_scalar(val)
    return out


def load_raw_config(
    yaml_path: str = "config/settings.yaml",
    env_path: str = ".env",
    root: Path | None = None,
) -> Dict[str, Any]:
    """加载集中配置：settings.yaml + .env(llm) + 环境变量，返回嵌套 dict。"""
    base_dir = (root or Path(__file__).resolve().parent.parent)
    yp = Path(yaml_path)
    if not yp.is_absolute():
        yp = base_dir / yp
    cfg: Dict[str, Any] = {}
    if yp.is_file():
        cfg = parse_yaml_lite(yp.read_text(encoding="utf-8"))

    env_cfg = _load_env_llm(str(base_dir / env_path))
    if env_cfg:
        cfg = _deep_merge(cfg, {"llm": env_cfg})

    overrides = _load_env_overrides()
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg
