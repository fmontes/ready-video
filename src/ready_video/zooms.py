from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import ResolvedConfig
from .errors import ZoomValidationError
from .io import atomic_write_json, atomic_write_text
from .transcript import Transcript, Word
from .agent import AgentSelection


class ProbeResult(BaseModel):
    installed: bool
    usable: bool
    detail: str = ""


class AgentBackend(Protocol):
    name: str

    def probe(self) -> ProbeResult: ...

    def plan_zooms(self, prompt_path: Path, schema_path: Path, timeout_s: int) -> str: ...


class ZoomCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    start_word_index: int
    end_word_index: int
    intensity: float = 1.12
    reason: str = ""


class Zoom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    intensity: float
    rank: int
    start_word_index: int
    end_word_index: int
    reason: str = ""


class ZoomPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "none"
    zooms: list[Zoom] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)
    warnings: list[dict] = Field(default_factory=list)


class CliBackend:
    def __init__(self, name: str) -> None:
        self.name = name
        self.executable: str | None = None

    def probe(self) -> ProbeResult:
        executable = shutil.which(self.name)
        if not executable:
            return ProbeResult(installed=False, usable=False, detail="not on PATH")
        completed = subprocess.run([executable, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        if completed.returncode == 0:
            self.executable = executable
        return ProbeResult(installed=True, usable=completed.returncode == 0, detail=(completed.stdout or completed.stderr).strip())

    def plan_zooms(self, prompt_path: Path, schema_path: Path, timeout_s: int) -> str:
        executable = self.executable or shutil.which(self.name) or self.name
        with tempfile.TemporaryDirectory(prefix="ready-video-agent-") as tmp:
            cwd = Path(tmp)
            if self.name == "claude":
                prompt = prompt_path.read_text()
                # --json-schema takes the schema inline as JSON, not a file path.
                # The CLI returns structured output via the StructuredOutput tool,
                # so it must be allowed; every other (write/read-capable) tool is
                # disabled. The result lands in the envelope's structured_output.
                schema_json = schema_path.read_text()
                completed = subprocess.run(
                    [
                        executable,
                        "--print",
                        "--output-format",
                        "json",
                        "--json-schema",
                        schema_json,
                        "--allowedTools",
                        "StructuredOutput",
                        "--disallowedTools",
                        "Bash,Read,Write,Edit,WebFetch,WebSearch,Glob,Grep",
                    ],
                    input=prompt,
                    cwd=cwd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_s,
                )
            elif self.name == "codex":
                prompt = prompt_path.read_text()
                response = cwd / "response.json"
                completed = subprocess.run(
                    [
                        executable,
                        "exec",
                        "--ephemeral",
                        "--sandbox",
                        "read-only",
                        "--output-schema",
                        str(schema_path),
                        "--output-last-message",
                        str(response),
                        "-",
                    ],
                    input=prompt,
                    cwd=cwd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_s,
                )
                if response.exists():
                    return response.read_text()
            else:
                completed = subprocess.run(
                    [executable, "run", "--pure", "--format", "json", "--file", str(prompt_path)],
                    cwd=cwd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_s,
                )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr)
            return completed.stdout


def select_backend(config: ResolvedConfig) -> AgentBackend | None:
    if config.agent.backend == "none":
        return None
    names = ["claude", "codex", "opencode"] if config.agent.backend == "auto" else [config.agent.backend]
    for name in names:
        backend = CliBackend(name)
        probe = backend.probe()
        if probe.installed and probe.usable:
            return backend
    return None


def plan_zooms(transcript: Transcript, output_path: Path, config: ResolvedConfig, *, no_agent: bool = False) -> ZoomPlan:
    if no_agent:
        plan = ZoomPlan(backend="none")
        atomic_write_json(output_path, plan)
        return plan
    backend = select_backend(config)
    if backend is None or not transcript.words:
        plan = ZoomPlan(backend="none")
        atomic_write_json(output_path, plan)
        return plan
    prompt = _prompt_for(transcript)
    prompt_path = output_path.parent / "agent-prompt.txt"
    schema_path = output_path.parent / "agent-schema.json"
    atomic_write_text(prompt_path, prompt, mode=0o600)
    atomic_write_json(schema_path, _schema(), mode=0o600)
    try:
        print(
            f"ready-video: selected local agent CLI '{backend.name}' for zoom planning; transcript words will be sent to that CLI.",
            file=sys.stderr,
        )
        response = backend.plan_zooms(prompt_path, schema_path, config.agent.timeout_s)
        atomic_write_text(output_path.parent / "agent-response.txt", response, mode=0o600)
        plan = validate_agent_response(response, transcript, config, backend=backend.name)
    except Exception as exc:
        retry_message = str(exc)
        retry_prompt_path = output_path.parent / "agent-prompt-retry.txt"
        atomic_write_text(retry_prompt_path, _retry_prompt(prompt, retry_message), mode=0o600)
        try:
            response = backend.plan_zooms(retry_prompt_path, schema_path, config.agent.timeout_s)
            atomic_write_text(output_path.parent / "agent-response-retry.txt", response, mode=0o600)
            plan = validate_agent_response(response, transcript, config, backend=backend.name)
            plan.warnings.append({"code": "ZOOM_PLANNING_RETRIED", "message": retry_message})
        except Exception as retry_exc:
            plan = ZoomPlan(
                backend=backend.name,
                warnings=[{"code": "ZOOM_PLANNING_FAILED", "message": str(retry_exc), "first_error": retry_message}],
            )
    atomic_write_json(output_path, plan)
    return plan


def _prompt_for(transcript: Transcript) -> str:
    lines = [
        "Select up to eight ranked hard punch-in zoom candidates from this edited-timeline transcript.",
        "Return strict JSON only. Do not include markdown, commentary, or keys outside this schema:",
        json.dumps(_schema(), sort_keys=True),
        "Use inclusive word indexes. Choose short emphasis moments; do not rewrite or remove spoken content.",
    ]
    for segment in transcript.segments:
        words = transcript.words[segment.word_range[0] : segment.word_range[1] + 1]
        text = " ".join(f"{word.i}:{word.w}" for word in words)
        lines.append(f"Segment {segment.id}: {text}")
    return "\n".join(lines)


def _retry_prompt(original_prompt: str, error: str) -> str:
    return "\n".join(
        [
            original_prompt,
            "",
            "The previous response could not be parsed or validated.",
            f"Validation error: {error}",
            "Return corrected strict JSON only, matching the schema exactly.",
        ]
    )


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "zooms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["rank", "start_word_index", "end_word_index", "intensity", "reason"],
                    "properties": {
                        "rank": {"type": "integer"},
                        "start_word_index": {"type": "integer"},
                        "end_word_index": {"type": "integer"},
                        "intensity": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                },
            }
        },
        "required": ["zooms"],
    }


def validate_agent_response(raw: str, transcript: Transcript, config: ResolvedConfig, *, backend: str = "manual") -> ZoomPlan:
    try:
        data = json.loads(_json_payload(raw))
        candidates = [ZoomCandidate.model_validate(item) for item in data.get("zooms", [])]
    except (AttributeError, json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise ZoomValidationError(f"malformed agent zoom response: {exc}") from exc
    accepted: list[Zoom] = []
    rejected: list[dict] = []
    seen_ranks: set[int] = set()
    words = transcript.words
    for candidate in sorted(candidates, key=lambda item: item.rank):
        if candidate.rank <= 0 or candidate.rank in seen_ranks:
            rejected.append({"rank": candidate.rank, "reason": "duplicate_or_nonpositive_rank"})
            continue
        seen_ranks.add(candidate.rank)
        if candidate.start_word_index < 0 or candidate.end_word_index >= len(words) or candidate.start_word_index > candidate.end_word_index:
            rejected.append({"rank": candidate.rank, "reason": "invalid_word_indices"})
            continue
        start_index, end_index = _snap_indices(candidate.start_word_index, candidate.end_word_index, words)
        start = words[start_index].start
        end = words[end_index].end
        intensity = min(1.15, max(1.08, candidate.intensity))
        start_index, end_index, start, end = _fit_duration(start_index, end_index, start, end, words, config)
        if end - start + 1e-6 < config.zooms.min_duration_s:
            rejected.append({"rank": candidate.rank, "reason": "too_short"})
            continue
        if any(_violates_spacing(start, end, z.start, z.end, config.zooms.min_gap_s) for z in accepted):
            rejected.append({"rank": candidate.rank, "reason": "overlap_or_gap"})
            continue
        accepted.append(
            Zoom(
                start=round(start, 3),
                end=round(end, 3),
                intensity=round(intensity, 3),
                rank=candidate.rank,
                start_word_index=start_index,
                end_word_index=end_index,
                reason=candidate.reason,
            )
        )
        if len(accepted) >= config.zooms.max_per_clip:
            break
    return ZoomPlan(backend=backend, zooms=sorted(accepted, key=lambda item: item.start), rejected=rejected)


def validate_zoom_plan(plan: ZoomPlan, transcript: Transcript, config: ResolvedConfig, *, edited_duration: float | None = None) -> None:
    errors: list[str] = []
    if len(plan.zooms) > config.zooms.max_per_clip:
        errors.append(f"too_many_zooms count={len(plan.zooms)} max={config.zooms.max_per_clip}")
    seen_ranks: set[int] = set()
    previous_start: float | None = None
    previous_end: float | None = None
    upper_bound = edited_duration if edited_duration is not None else transcript.duration
    for index, zoom in enumerate(plan.zooms):
        if zoom.rank <= 0 or zoom.rank in seen_ranks:
            errors.append(f"zoom[{index}] duplicate_or_nonpositive_rank rank={zoom.rank}")
        seen_ranks.add(zoom.rank)
        if zoom.start < 0 or zoom.end > upper_bound + 1e-3 or zoom.end <= zoom.start:
            errors.append(f"zoom[{index}] invalid_bounds start={zoom.start} end={zoom.end} duration={upper_bound}")
        duration = zoom.end - zoom.start
        if duration < config.zooms.min_duration_s - 1e-3 or duration > config.zooms.max_duration_s + 1e-3:
            errors.append(f"zoom[{index}] invalid_duration duration={duration:.3f}")
        if zoom.intensity < 1.08 or zoom.intensity > 1.15:
            errors.append(f"zoom[{index}] invalid_intensity intensity={zoom.intensity}")
        if zoom.start_word_index < 0 or zoom.end_word_index >= len(transcript.words) or zoom.start_word_index > zoom.end_word_index:
            errors.append(f"zoom[{index}] invalid_word_indices")
        else:
            snapped_start, snapped_end = _snap_indices(zoom.start_word_index, zoom.end_word_index, transcript.words)
            if (snapped_start, snapped_end) != (zoom.start_word_index, zoom.end_word_index):
                errors.append(
                    f"zoom[{index}] interpolated_boundary start_word_index={zoom.start_word_index} end_word_index={zoom.end_word_index}"
                )
            start_word = transcript.words[zoom.start_word_index]
            end_word = transcript.words[zoom.end_word_index]
            if start_word.timing == "interpolated" and _nearby_aligned_start(start_word, transcript.words):
                errors.append(f"zoom[{index}] interpolated_boundary start_word_index={zoom.start_word_index}")
            if end_word.timing == "interpolated" and _nearby_aligned_end(end_word, transcript.words):
                errors.append(f"zoom[{index}] interpolated_boundary end_word_index={zoom.end_word_index}")
            if not _nearly_equal(zoom.start, start_word.start) or not _nearly_equal(zoom.end, end_word.end):
                errors.append(
                    f"zoom[{index}] not_on_word_boundaries expected={start_word.start:.3f}-{end_word.end:.3f} actual={zoom.start:.3f}-{zoom.end:.3f}"
                )
        if previous_start is not None and zoom.start < previous_start - 1e-3:
            errors.append(f"zoom[{index}] not_chronological")
        if previous_end is not None and previous_start is not None and _violates_spacing(
            zoom.start, zoom.end, previous_start, previous_end, config.zooms.min_gap_s
        ):
            errors.append(f"zoom[{index}] overlap_or_gap")
        previous_start = zoom.start
        previous_end = zoom.end
    if errors:
        raise ZoomValidationError("saved zoom plan failed validation", details="; ".join(errors))


def _nearly_equal(left: float, right: float, *, tolerance: float = 1e-3) -> bool:
    return abs(left - right) <= tolerance


def _nearby_aligned_start(word: Word, words: list[Word]) -> bool:
    return any(item.timing == "aligned" and abs(item.start - word.start) <= 0.300 for item in words)


def _nearby_aligned_end(word: Word, words: list[Word]) -> bool:
    return any(item.timing == "aligned" and abs(item.end - word.end) <= 0.300 for item in words)


def _json_payload(raw: str) -> str:
    text = raw.strip()
    if text.startswith("{") or text.startswith("["):
        return _unwrap_envelope(text)
    return _strip_fence(text)


def _unwrap_envelope(text: str) -> str:
    """Unwrap a CLI result envelope to the JSON object holding ``zooms``.

    Claude's ``--output-format json`` wraps the answer in a result envelope. The
    schema-validated object is under ``structured_output``; the model text is
    under ``result`` (which may itself be fenced JSON). Codex/opencode and
    hand-written responses are already the bare object and pass through.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(parsed, dict):
        return text
    if "zooms" in parsed:
        return text  # already the bare payload
    structured = parsed.get("structured_output")
    if isinstance(structured, dict):
        return json.dumps(structured)
    result = parsed.get("result")
    if isinstance(result, str):
        return _strip_fence(result.strip())
    return text


def _strip_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    # Fallback: a fenced ```json block embedded in surrounding prose.
    fence = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def _snap_indices(start: int, end: int, words: list[Word]) -> tuple[int, int]:
    if words[start].timing == "interpolated":
        for i in range(start, end + 1):
            if words[i].timing == "aligned" and words[i].start - words[start].start <= 0.300:
                start = i
                break
    if words[end].timing == "interpolated":
        for i in range(end, start - 1, -1):
            if words[i].timing == "aligned" and words[end].end - words[i].end <= 0.300:
                end = i
                break
    return start, end


def _fit_duration(start_index: int, end_index: int, start: float, end: float, words: list[Word], config: ResolvedConfig) -> tuple[int, int, float, float]:
    max_duration = config.zooms.max_duration_s
    while end - start > max_duration and end_index > start_index:
        end_index -= 1
        end = words[end_index].end
    while end - start < config.zooms.min_duration_s and end_index + 1 < len(words) and words[end_index + 1].end - start <= max_duration:
        end_index += 1
        end = words[end_index].end
    while end - start < config.zooms.min_duration_s and start_index > 0 and end - words[start_index - 1].start <= max_duration:
        start_index -= 1
        start = words[start_index].start
    return start_index, end_index, start, end


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def _violates_spacing(a_start: float, a_end: float, b_start: float, b_end: float, min_gap: float) -> bool:
    if _overlaps(a_start, a_end, b_start, b_end):
        return True
    if a_end <= b_start:
        return b_start - a_end < min_gap
    return a_start - b_end < min_gap


def generate_zooms(selection: AgentSelection, transcript: dict, output_path: Path) -> dict:
    payload = {"backend": selection.name, "zooms": []}
    atomic_write_json(output_path, payload)
    return payload


def validate_zooms(payload: dict, *, media_duration: float) -> dict:
    normalized = []
    for item in payload.get("zooms", []):
        start = float(item["start"])
        end = float(item["end"])
        scale = float(item.get("scale", item.get("zoom", 1.12)))
        if start < 0:
            raise ZoomValidationError("zoom starts before zero")
        if end <= start:
            raise ZoomValidationError("zoom end must be after start")
        if end > media_duration:
            raise ZoomValidationError("zoom ends after media duration")
        normalized.append({"start": start, "end": end, "scale": scale})
    return {"zooms": normalized}
