"""Agent backend selection for zoom generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import shutil
from typing import Any

AGENT_ORDER = ("claude", "codex", "opencode")


@dataclass(frozen=True)
class AgentSelection:
    name: str
    path: str | None
    available: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_agent_backend(preferred: str = "auto") -> AgentSelection:
    if preferred == "none":
        return AgentSelection("none", None, True, "agent generation disabled")

    if preferred != "auto":
        path = shutil.which(preferred)
        if path is None:
            return AgentSelection(preferred, None, False, f"{preferred} not found on PATH")
        return AgentSelection(preferred, path, True, f"{preferred} selected explicitly")

    for candidate in AGENT_ORDER:
        path = shutil.which(candidate)
        if path:
            return AgentSelection(candidate, path, True, f"{candidate} found on PATH")

    return AgentSelection("none", None, True, "no supported agent found on PATH")


def doctor() -> dict[str, Any]:
    checked = {}
    for candidate in AGENT_ORDER:
        path = shutil.which(candidate)
        checked[candidate] = {"available": path is not None, "path": path}
    checked["auto"] = select_agent_backend("auto").to_dict()
    return checked
