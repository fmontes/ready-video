from __future__ import annotations

import json
import importlib.util
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import ResolvedConfig
from .errors import ReadyVideoError, TranscriptError
from .ffmpeg import BinaryPair
from .io import atomic_write_json
from .media import media_duration


class Word(BaseModel):
    model_config = ConfigDict(extra="forbid")

    i: int
    w: str
    start: float
    end: float
    timing: str = "aligned"


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    text: str
    start: float
    end: float
    word_range: tuple[int, int]


class Transcript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = "unknown"
    duration: float
    words: list[Word] = Field(default_factory=list)
    segments: list[TranscriptSegment] = Field(default_factory=list)
    warnings: list[dict[str, str]] = Field(default_factory=list)


def transcribe(speech_wav: Path, output_path: Path, pair: BinaryPair, config: ResolvedConfig) -> Transcript:
    duration = media_duration(speech_wav, pair)
    try:
        transcript = _transcribe_with_whisperx(speech_wav, duration, config)
    except ImportError as exc:
        raise ReadyVideoError(
            "TRANSCRIPTION_UNAVAILABLE",
            "WhisperX is required for transcription and word-level subtitle timing.",
            'Install the transcription dependencies with `python -m pip install -e ".[transcription]"`, then rerun `ready-video doctor`.',
            details=str(exc),
        ) from exc
    except ReadyVideoError:
        raise
    except Exception as exc:
        raise ReadyVideoError("TRANSCRIPTION_FAILED", "WhisperX transcription failed.", details=str(exc)) from exc
    atomic_write_json(output_path, transcript)
    return transcript


def _transcribe_with_whisperx(speech_wav: Path, duration: float, config: ResolvedConfig) -> Transcript:
    import whisperx  # type: ignore

    device = "cpu" if config.transcription.device == "auto" else config.transcription.device
    compute_type = "int8" if config.transcription.compute_type == "auto" else config.transcription.compute_type
    model_name = "small" if config.transcription.model == "auto" else config.transcription.model
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=None if config.transcription.language == "auto" else config.transcription.language)
    result = _call_whisperx_transcribe(model, speech_wav, config)
    language = result.get("language") or "unknown"
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(result["segments"], model_a, metadata, str(speech_wav), device, return_char_alignments=False)
    return normalize_whisperx(aligned, duration, language)


def _call_whisperx_transcribe(model: Any, speech_wav: Path, config: ResolvedConfig) -> dict[str, Any]:
    signature = inspect.signature(model.transcribe)
    supported = set(signature.parameters)
    kwargs: dict[str, Any] = {}
    if "batch_size" in supported:
        kwargs["batch_size"] = 8
    if "language" in supported and config.transcription.language != "auto":
        kwargs["language"] = config.transcription.language
    if config.transcription.initial_prompt:
        if "initial_prompt" not in supported:
            raise ReadyVideoError(
                "TRANSCRIPTION_FAILED",
                "The installed WhisperX transcribe API does not support transcription.initial_prompt.",
                "Remove transcription.initial_prompt or install a WhisperX version that supports it.",
            )
        kwargs["initial_prompt"] = config.transcription.initial_prompt
    result = model.transcribe(str(speech_wav), **kwargs)
    if not isinstance(result, dict):
        try:
            result = dict(result)
        except Exception as exc:
            raise ReadyVideoError("TRANSCRIPTION_FAILED", "WhisperX returned an unsupported transcription result.", details=str(type(result))) from exc
    return result


def normalize_whisperx(raw: dict[str, Any], duration: float, language: str = "unknown") -> Transcript:
    words: list[Word] = []
    segments: list[TranscriptSegment] = []
    warnings: list[dict[str, str]] = []
    bounded_duration = max(0.0, _finite_float(duration, 0.0))
    floor = 0.0
    for segment_id, segment in enumerate(raw.get("segments", [])):
        segment_words = segment.get("words") or []
        first_index = len(words)
        start, end = _segment_bounds(segment, bounded_duration, floor)
        if end < start:
            warnings.append(_segment_warning(segment_id, segment, "invalid_segment_bounds"))
            continue
        normalized_words = _normalize_segment_words(segment_words, start, end, bounded_duration)
        if not normalized_words:
            warnings.append(_segment_warning(segment_id, segment, "no_textual_word_tokens"))
            continue
        for item in normalized_words:
            words.append(Word(i=len(words), **item))
        floor = words[-1].end
        segments.append(
            TranscriptSegment(
                id=segment_id,
                text=str(segment.get("text", "")).strip(),
                start=round(start, 3),
                end=round(max(start, words[-1].end, min(end, bounded_duration)), 3),
                word_range=(first_index, len(words) - 1),
            )
        )
    return Transcript(language=language, duration=bounded_duration, words=words, segments=segments, warnings=warnings)


def _normalize_segment_words(items: list[dict[str, Any]], start: float, end: float, duration: float) -> list[dict[str, Any]]:
    tokens = _word_tokens(items)
    if not tokens or end < start:
        return []

    bounded_start = _clamp(start, 0.0, duration)
    bounded_end = _clamp(end, bounded_start, duration)
    _mark_aligned_tokens(tokens, bounded_start, bounded_end, duration)

    anchors = [index for index, token in enumerate(tokens) if token.timing == "aligned"]
    if anchors:
        previous = -1
        previous_end = bounded_start
        for anchor in anchors:
            _allocate_interpolated(tokens, previous + 1, anchor, previous_end, tokens[anchor].start)
            previous = anchor
            previous_end = tokens[anchor].end
        _allocate_interpolated(tokens, previous + 1, len(tokens), previous_end, bounded_end)
    else:
        _allocate_interpolated(tokens, 0, len(tokens), bounded_start, bounded_end)

    normalized: list[dict[str, Any]] = []
    cursor = bounded_start
    for token in tokens:
        word_start = _clamp(token.start, cursor, bounded_end)
        word_end = _clamp(token.end, word_start, bounded_end)
        cursor = word_end
        normalized.append({"w": token.w, "start": round(word_start, 3), "end": round(word_end, 3), "timing": token.timing})
    return normalized


@dataclass
class _Token:
    w: str
    raw_start: float | None
    raw_end: float | None
    start: float = 0.0
    end: float = 0.0
    timing: str = "interpolated"


def _word_tokens(items: list[dict[str, Any]]) -> list[_Token]:
    tokens: list[_Token] = []
    for item in items:
        text = _word_text(item)
        if not text:
            continue
        tokens.append(
            _Token(
                w=text,
                raw_start=_optional_float(item.get("start")),
                raw_end=_optional_float(item.get("end")),
            )
        )
    return tokens


def _mark_aligned_tokens(tokens: list[_Token], start: float, end: float, duration: float) -> None:
    cursor = start
    for token in tokens:
        if token.raw_start is None or token.raw_end is None:
            continue
        if token.raw_end < token.raw_start:
            continue
        word_start = _clamp(token.raw_start, start, min(end, duration))
        word_end = _clamp(token.raw_end, word_start, min(end, duration))
        if word_end < word_start:
            continue
        word_start = max(word_start, cursor)
        word_end = max(word_start, word_end)
        if word_start > end:
            continue
        token.start = word_start
        token.end = min(word_end, end)
        token.timing = "aligned"
        cursor = token.end


def _word_text(item: dict[str, Any]) -> str:
    if "word" in item:
        value = item["word"]
    elif "w" in item:
        value = item["w"]
    else:
        value = item.get("text", "")
    return str(value).strip()


def _allocate_interpolated(tokens: list[_Token], first: int, stop: int, start: float, end: float) -> None:
    if first >= stop:
        return
    bounded_end = max(start, end)
    span = bounded_end - start
    weights = [max(1, len(token.w)) for token in tokens[first:stop]]
    total = sum(weights)
    cursor = start
    cumulative = 0
    for offset, (token, weight) in enumerate(zip(tokens[first:stop], weights, strict=True), start=first):
        token.start = cursor
        cumulative += weight
        cursor = bounded_end if offset == stop - 1 else start + span * (cumulative / total)
        token.end = max(token.start, cursor)
        token.timing = "interpolated"


def _segment_bounds(segment: dict[str, Any], duration: float, floor: float) -> tuple[float, float]:
    start = _finite_float(segment.get("start"), floor)
    end = _finite_float(segment.get("end"), duration)
    bounded_start = max(floor, _clamp(start, 0.0, duration))
    bounded_end = _clamp(end, 0.0, duration)
    return bounded_start, bounded_end


def _segment_warning(segment_id: int, segment: dict[str, Any], reason: str) -> dict[str, str]:
    text = str(segment.get("text", "")).strip()
    return {
        "code": "UNUSABLE_TRANSCRIPT_SEGMENT",
        "segment_id": str(segment_id),
        "reason": reason,
        "message": text,
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = _finite_float(value, math.nan)
    return None if math.isnan(parsed) else parsed


def _finite_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _clamp(value: float, lower: float, upper: float) -> float:
    if upper < lower:
        return lower
    return min(upper, max(lower, value))


def read_transcript(path: Path) -> Transcript:
    return Transcript.model_validate(json.loads(path.read_text()))


def create_transcript(speech_wav: Path, output_path: Path, *, fallback: str | None = None) -> dict:
    if fallback == "empty":
        payload = {"engine": "empty-fallback", "segments": [], "words": []}
        atomic_write_json(output_path, payload)
        return payload
    if importlib.util.find_spec("whisperx") is None:
        raise TranscriptError("WhisperX is not installed")
    raise TranscriptError("WhisperX execution is available through transcribe(), not create_transcript().")
