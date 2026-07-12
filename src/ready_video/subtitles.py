"""SRT and ASS subtitle generation."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .timeline import TimelineSegment, format_ass_timestamp, format_srt_timestamp, normalize_segments


@dataclass(frozen=True)
class SubtitleStyle:
    name: str = "Bold"
    font: str = "Arial"
    font_size: int = 72
    primary_color: str = "&H00FFFFFF"
    secondary_color: str = "&H000000FF"
    outline_color: str = "&H00000000"
    back_color: str = "&H64000000"
    bold: bool = True
    italic: bool = False
    border_style: int = 1
    outline: int = 4
    shadow: int = 1
    alignment: int = 2
    margin_l: int = 80
    margin_r: int = 80
    margin_v: int = 90


def generate_srt(cues: Iterable[TimelineSegment]) -> str:
    blocks = []
    for index, cue in enumerate(_valid_cues(cues), start=1):
        text = _clean_srt_text(cue.text)
        blocks.append(
            f"{index}\n"
            f"{format_srt_timestamp(cue.start)} --> {format_srt_timestamp(cue.end)}\n"
            f"{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def generate_ass(
    cues: Iterable[TimelineSegment],
    style: Optional[SubtitleStyle] = None,
    title: str = "ready-video subtitles",
    resolution: Sequence[int] = (1080, 1920),
) -> str:
    selected_style = style or SubtitleStyle()
    width, height = resolution
    if width <= 0 or height <= 0:
        raise ValueError("resolution dimensions must be positive")

    header = [
        "[Script Info]",
        f"Title: {title}",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        _format_style(selected_style),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events = [
        "Dialogue: 0,{start},{end},{style},,0,0,0,,{text}".format(
            start=format_ass_timestamp(cue.start),
            end=format_ass_timestamp(cue.end),
            style=selected_style.name,
            text=_escape_ass_text(cue.text),
        )
        for cue in _valid_cues(cues)
    ]
    return "\n".join(header + events) + "\n"


def cues_from_words(words: Iterable[dict], max_gap: float = 0.8) -> list[TimelineSegment]:
    if max_gap < 0:
        raise ValueError("max_gap cannot be negative")

    cues: list[TimelineSegment] = []
    current_words: list[str] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        text = str(word.get("word") or word.get("text") or word.get("w") or "").strip()
        if not text:
            continue
        if current_start is None:
            current_start = start
            current_end = end
            current_words = [text]
            continue
        assert current_end is not None
        if start - current_end > max_gap:
            cues.append(TimelineSegment(current_start, current_end, " ".join(current_words)))
            current_start = start
            current_words = [text]
        else:
            current_words.append(text)
        current_end = end

    if current_start is not None and current_end is not None and current_words:
        cues.append(TimelineSegment(current_start, current_end, " ".join(current_words)))
    return cues


def ass_color(hex_color: str) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = value[0:2], value[2:4], value[4:6]
    return f"&H00{blue}{green}{red}"


def ass_time(seconds: float) -> str:
    return format_ass_timestamp(seconds)


def srt_time(seconds: float) -> str:
    return format_srt_timestamp(seconds)


def escape_ass(text: str) -> str:
    return _escape_ass_text(text).replace("\\N", " ")


def generate_subtitles(
    transcript: Any,
    ass_path: Path,
    srt_path: Optional[Path],
    config: Any,
    resolution: tuple[int, int],
) -> None:
    if config.subtitles.mode == "none" or not transcript.words:
        return
    events = _karaoke_events(transcript.words, config) if config.subtitles.mode == "karaoke" else _line_events(transcript.words, config)
    width, height = resolution
    alignment, margin_v = _vertical_placement(config.subtitles.vertical_position, height)
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,"
        f"{config.subtitles.font_family},{config.subtitles.font_size},{ass_color(config.subtitles.primary_color)},"
        f"{ass_color(config.subtitles.highlight_color)},{ass_color(config.subtitles.outline_color)},&H80000000,"
        f"1,0,0,0,100,100,0,0,1,{config.subtitles.outline_width},{config.subtitles.shadow},{alignment},80,80,{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    body = [f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{text}" for start, end, text in events if end > start]
    _atomic_write_text(ass_path, "\n".join(header + body) + "\n")
    if srt_path:
        write_srt(events, srt_path)


def write_srt(events: list[tuple[float, float, str]], path: Path) -> None:
    lines = []
    for i, (start, end, text) in enumerate(events, start=1):
        plain = re.sub(r"\{[^}]*\}", "", text)
        lines += [str(i), f"{srt_time(start)} --> {srt_time(end)}", plain, ""]
    _atomic_write_text(path, "\n".join(lines))


def _vertical_placement(vertical_position: float, height: int) -> tuple[int, int]:
    """Map a normalized vertical_position (0 = top, 1 = bottom) to an ASS
    alignment and margin.

    - < 0.4  -> top-anchored (alignment 8); margin_v = distance down from top
    - 0.4-0.6 -> middle-center (alignment 5); margin_v is ignored by libass
    - > 0.6  -> bottom-anchored (alignment 2); margin_v = distance up from bottom

    The margin is clamped to a platform-safe band so text never hugs an edge.
    """
    safe = max(0, min(height // 2, 120))
    if vertical_position < 0.4:
        margin = max(safe, min(height - safe, int(height * vertical_position)))
        return 8, margin
    if vertical_position > 0.6:
        # distance up from the bottom = height * (1 - position)
        margin = max(safe, min(height - safe, int(height * (1.0 - vertical_position))))
        return 2, margin
    return 5, 0


def _karaoke_events(words: list[Any], config: Any) -> list[tuple[float, float, str]]:
    chunks = _chunks(words, config.subtitles.max_words_per_line, config.subtitles.max_chars_per_line)
    events = []
    for chunk in chunks:
        text_parts = []
        for word in chunk:
            duration_cs = max(1, int(round((word.end - word.start) * 100)))
            text = word.w.upper() if config.subtitles.uppercase else word.w
            text_parts.append(f"{{\\k{duration_cs}}}{escape_ass(text)}")
        events.append((chunk[0].start, chunk[-1].end, " ".join(text_parts)))
    return events


def _line_events(words: list[Any], config: Any) -> list[tuple[float, float, str]]:
    chunks = _chunks(words, config.subtitles.max_words_per_line, config.subtitles.max_chars_per_line)
    events = []
    for chunk in chunks:
        text = " ".join(word.w for word in chunk)
        if config.subtitles.uppercase:
            text = text.upper()
        events.append((chunk[0].start, chunk[-1].end, escape_ass(text)))
    return events


def _chunks(words: list[Any], max_words: int, max_chars: int) -> list[list[Any]]:
    chunks: list[list[Any]] = []
    current: list[Any] = []
    for word in words:
        prospective = current + [word]
        char_count = len(" ".join(item.w for item in prospective))
        if current and (len(prospective) > max_words or char_count > max_chars):
            chunks.append(current)
            current = [word]
        else:
            current = prospective
    if current:
        chunks.append(current)
    return chunks


def _valid_cues(cues: Iterable[TimelineSegment]) -> list[TimelineSegment]:
    return [
        cue
        for cue in normalize_segments(cues)
        if cue.start is not None and cue.end is not None and cue.end > cue.start and cue.text.strip()
    ]


def _format_style(style: SubtitleStyle) -> str:
    return (
        "Style: "
        f"{style.name},{style.font},{style.font_size},{style.primary_color},"
        f"{style.secondary_color},{style.outline_color},{style.back_color},"
        f"{_ass_bool(style.bold)},{_ass_bool(style.italic)},0,0,100,100,0,0,"
        f"{style.border_style},{style.outline},{style.shadow},{style.alignment},"
        f"{style.margin_l},{style.margin_r},{style.margin_v},1"
    )


def _ass_bool(value: bool) -> int:
    return -1 if value else 0


def _escape_ass_text(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return re.sub(r"\r?\n", r"\\N", escaped.strip())


def _clean_srt_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines())


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(text)
    tmp.replace(path)
