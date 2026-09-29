"""大模型根因推理执行体（MS4）。"""

from .knowledge_graph import KnowledgeGraph
from .anti_hallucination import AntiHallucinationChecker
from .confidence import ConfidenceGate, CascadeGate


def __getattr__(name):
    """Offline reward scoring does not need the optional HTTP client stack."""
    if name in ("LLMClient", "load_env_config"):
        from . import llm_client
        return getattr(llm_client, name)
    if name in ("RootCauseReasoner", "DiagnosisResult"):
        from . import root_cause
        return getattr(root_cause, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "KnowledgeGraph",
    "LLMClient",
    "load_env_config",
    "RootCauseReasoner",
    "DiagnosisResult",
    "AntiHallucinationChecker",
    "ConfidenceGate",
    "CascadeGate",
]
