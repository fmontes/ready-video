"""Timeline conversion and source-to-edited mapping helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable, Optional

from .errors import ReadyVideoError

FRAME_30 = 1 / 30
EPS = 1e-6


@dataclass
class TimelineSegment:
    start: Optional[float] = None
    end: Optional[float] = None
    text: str = ""
    i: Optional[int] = None
    source_start: Optional[float] = None
    source_end: Optional[float] = None
    edited_start: Optional[float] = None
    edited_end: Optional[float] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args:
            if len(args) not in {2, 3}:
                raise TypeError("TimelineSegment accepts start, end, optional text")
            kwargs.setdefault("start", args[0])
            kwargs.setdefault("end", args[1])
            if len(args) == 3:
                kwargs.setdefault("text", args[2])

        self.start = _optional_float(kwargs.pop("start", None))
        self.end = _optional_float(kwargs.pop("end", None))
        self.text = str(kwargs.pop("text", ""))
        self.i = kwargs.pop("i", None)
        self.source_start = _optional_float(kwargs.pop("source_start", None))
        self.source_end = _optional_float(kwargs.pop("source_end", None))
        self.edited_start = _optional_float(kwargs.pop("edited_start", None))
        self.edited_end = _optional_float(kwargs.pop("edited_end", None))
        if kwargs:
            raise TypeError(f"Unknown TimelineSegment fields: {', '.join(sorted(kwargs))}")

        if self.start is not None or self.end is not None:
            if self.start is None or self.end is None:
                raise ValueError("subtitle cue segments require both start and end")
            if self.start < 0:
                raise ValueError("segment start cannot be negative")
            if self.end < self.start:
                raise ValueError("segment end must be greater than or equal to start")

        mapping_fields = [self.source_start, self.source_end, self.edited_start, self.edited_end]
        if any(value is not None for value in mapping_fields):
            if any(value is None for value in mapping_fields):
                raise ValueError("mapping segments require source and edited ranges")
            if self.source_end <= self.source_start:
                raise ValueError("timeline source ranges must be positive")
            if self.edited_end <= self.edited_start:
                raise ValueError("timeline edited ranges must be positive")

    @property
    def duration(self) -> float:
        if self.start is None or self.end is None:
            return self.edited_duration
        return self.end - self.start

    @property
    def source_duration(self) -> float:
        if self.source_start is None or self.source_end is None:
            raise AttributeError("segment has no source range")
        return self.source_end - self.source_start

    @property
    def edited_duration(self) -> float:
        if self.edited_start is None or self.edited_end is None:
            raise AttributeError("segment has no edited range")
        return self.edited_end - self.edited_start

    def shifted(self, offset: float) -> "TimelineSegment":
        if self.start is None or self.end is None:
            raise AttributeError("only subtitle cue segments can be shifted")
        return TimelineSegment(
            start=max(0.0, self.start + offset),
            end=max(0.0, self.end + offset),
            text=self.text,
        )

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None and value != ""}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimelineSegment):
            return False
        return self.model_dump() == other.model_dump()


@dataclass
class Timeline:
    source_duration: float
    edited_duration: float
    segments: list[TimelineSegment]

    def __post_init__(self) -> None:
        previous_source = -EPS
        previous_edited = -EPS
        for expected_i, segment in enumerate(self.segments):
            if segment.i != expected_i:
                raise ValueError("timeline segment indices must be contiguous")
            if segment.source_start is None or segment.source_end is None:
                raise ValueError("timeline segments require source ranges")
            if segment.edited_start is None or segment.edited_end is None:
                raise ValueError("timeline segments require edited ranges")
            if segment.source_start + EPS < previous_source:
                raise ValueError("timeline source ranges must be sorted")
            if segment.edited_start + EPS < previous_edited:
                raise ValueError("timeline edited ranges must be sorted")
            previous_source = segment.source_end
            previous_edited = segment.edited_end

    @classmethod
    def model_validate(cls, data: Any) -> "Timeline":
        if isinstance(data, Timeline):
            return data
        segments = [TimelineSegment(**item) for item in data.get("segments", [])]
        return cls(
            source_duration=float(data["source_duration"]),
            edited_duration=float(data["edited_duration"]),
            segments=segments,
        )

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "source_duration": self.source_duration,
            "edited_duration": self.edited_duration,
            "segments": [segment.model_dump() for segment in self.segments],
        }

    def source_to_edited(self, time_s: float) -> Optional[float]:
        for segment in self.segments:
            if segment.source_start - EPS <= time_s <= segment.source_end + EPS:
                value = segment.edited_start + max(0.0, min(time_s, segment.source_end) - segment.source_start)
                return min(value, segment.edited_end)
        return None

    def edited_to_source(self, time_s: float) -> float:
        for segment in self.segments:
            if segment.edited_start - EPS <= time_s <= segment.edited_end + EPS:
                value = segment.source_start + max(0.0, min(time_s, segment.edited_end) - segment.edited_start)
                return min(value, segment.source_end)
        raise ReadyVideoError(
            "TIMELINE_MAPPING_FAILED",
            f"Edited timestamp {time_s:.3f}s is outside the timeline.",
        )

    def edited_range_to_source_ranges(self, start_s: float, end_s: float) -> list[tuple[float, float]]:
        if end_s < start_s:
            raise ReadyVideoError("TIMELINE_MAPPING_FAILED", "Range end is before range start.")
        ranges: list[tuple[float, float]] = []
        for segment in self.segments:
            start = max(start_s, segment.edited_start)
            end = min(end_s, segment.edited_end)
            if end > start + EPS:
                ranges.append(
                    (
                        segment.source_start + (start - segment.edited_start),
                        segment.source_start + (end - segment.edited_start),
                    )
                )
        if not ranges and end_s > start_s:
            raise ReadyVideoError("TIMELINE_MAPPING_FAILED", "Edited range does not intersect the timeline.")
        return ranges


@dataclass(frozen=True)
class SilenceParams:
    enabled: bool = True
    threshold_db: float = -35.0
    min_silence_s: float = 0.45
    margin_before_s: float = 0.12
    margin_after_s: float = 0.18


def seconds_to_frames(seconds: float, fps: float) -> int:
    _validate_fps(fps)
    if seconds < 0:
        raise ValueError("seconds cannot be negative")
    frames = Decimal(str(seconds)) * Decimal(str(fps))
    return int(frames.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def frames_to_seconds(frames: int, fps: float) -> float:
    _validate_fps(fps)
    if frames < 0:
        raise ValueError("frames cannot be negative")
    return frames / fps


def seconds_to_milliseconds(seconds: float) -> int:
    if seconds < 0:
        raise ValueError("seconds cannot be negative")
    milliseconds = Decimal(str(seconds)) * Decimal("1000")
    return int(milliseconds.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def milliseconds_to_seconds(milliseconds: int) -> float:
    if milliseconds < 0:
        raise ValueError("milliseconds cannot be negative")
    return milliseconds / 1000.0


def format_srt_timestamp(seconds: float) -> str:
    total_ms = seconds_to_milliseconds(seconds)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


def format_ass_timestamp(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("seconds cannot be negative")
    raw_centiseconds = Decimal(str(seconds)) * Decimal("100")
    centiseconds = int(raw_centiseconds.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02}:{whole_seconds:02}.{centiseconds:02}"


def clamp_segments(segments: Iterable[TimelineSegment], duration: float) -> list[TimelineSegment]:
    if duration < 0:
        raise ValueError("duration cannot be negative")
    clamped = []
    for segment in segments:
        if segment.start is None or segment.end is None:
            continue
        start = min(segment.start, duration)
        end = min(segment.end, duration)
        if end > start:
            clamped.append(TimelineSegment(start=start, end=end, text=segment.text))
    return clamped


def shift_segments(segments: Iterable[TimelineSegment], offset: float) -> list[TimelineSegment]:
    return [segment.shifted(offset) for segment in segments]


def normalize_segments(segments: Iterable[TimelineSegment]) -> list[TimelineSegment]:
    return sorted(segments, key=lambda segment: (segment.start or 0.0, segment.end or 0.0, segment.text))


def merge_ranges(ranges: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if not merged or start > merged[-1][1] + EPS:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def timeline_from_silences(
    source_duration: float,
    silent_ranges: Iterable[tuple[float, float]],
    params: SilenceParams,
) -> Timeline:
    if source_duration <= 0:
        raise ReadyVideoError("INVALID_MEDIA", "Source duration must be positive.")
    if not params.enabled:
        kept = [(0.0, source_duration)]
    else:
        silences = merge_ranges((max(0, start), min(source_duration, end)) for start, end in silent_ranges)
        kept = []
        cursor = 0.0
        for silence_start, silence_end in silences:
            if silence_start > cursor:
                kept.append(
                    (
                        max(0.0, cursor - params.margin_before_s),
                        min(source_duration, silence_start + params.margin_after_s),
                    )
                )
            cursor = max(cursor, silence_end)
        if cursor < source_duration:
            kept.append((max(0.0, cursor - params.margin_before_s), source_duration))
        kept = merge_ranges(kept)
        kept = _drop_tiny_gaps(kept, max(2 * FRAME_30, EPS))
        kept = _merge_short_segments(kept, 0.10)
        if not kept:
            raise ReadyVideoError(
                "NO_SPEECH_DETECTED",
                "No speech-like audio remained after silence analysis.",
                "Lower silence.threshold_db or disable silence removal.",
            )
    segments: list[TimelineSegment] = []
    edited = 0.0
    for i, (source_start, source_end) in enumerate(kept):
        duration = source_end - source_start
        segments.append(
            TimelineSegment(
                i=i,
                source_start=round(source_start, 6),
                source_end=round(source_end, 6),
                edited_start=round(edited, 6),
                edited_end=round(edited + duration, 6),
            )
        )
        edited += duration
    return Timeline(source_duration=source_duration, edited_duration=round(edited, 6), segments=segments)


def _drop_tiny_gaps(ranges: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    if not ranges:
        return []
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        if start - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _merge_short_segments(ranges: list[tuple[float, float]], min_duration: float) -> list[tuple[float, float]]:
    ranges = list(ranges)
    i = 0
    while i < len(ranges):
        start, end = ranges[i]
        if end - start >= min_duration or len(ranges) == 1:
            i += 1
            continue
        if i == 0:
            ranges[1] = (start, ranges[1][1])
            ranges.pop(i)
        else:
            ranges[i - 1] = (ranges[i - 1][0], end)
            ranges.pop(i)
    return ranges


def _validate_fps(fps: float) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)
