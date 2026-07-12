from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .config import ResolvedConfig
from .errors import ReadyVideoError
from .ffmpeg import BinaryPair, available_filters, ffprobe_json, run
from .io import atomic_write_json
from .timeline import SilenceParams, Timeline, timeline_from_silences


class SourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration: float
    start_time: float = 0.0
    width: int
    height: int
    video_stream_index: int
    audio_stream_index: int
    video_codec: str = ""
    audio_codec: str = ""
    pix_fmt: str = ""
    sample_aspect_ratio: str = "1:1"
    frame_rate: str = ""
    avg_frame_rate: str = ""
    variable_frame_rate: bool = False
    color_primaries: str = ""
    color_transfer: str = ""
    color_matrix: str = ""
    color_range: str = ""
    rotation: int = 0
    raw: dict = Field(default_factory=dict)


def probe_source(input_path: Path, pair: BinaryPair) -> SourceInfo:
    data = ffprobe_json(
        pair,
        [
            "-show_streams",
            "-show_format",
            input_path,
        ],
    )
    streams = data.get("streams", [])
    video = _choose_stream(streams, "video")
    audio = _choose_stream(streams, "audio")
    if not video:
        raise ReadyVideoError("NO_VIDEO_STREAM", "Input has no usable video stream.")
    if not audio:
        raise ReadyVideoError("NO_AUDIO_STREAM", "Input has no usable audio stream.")
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise ReadyVideoError("INVALID_MEDIA", "Input duration could not be determined.")
    start_time = float(video.get("start_time") or data.get("format", {}).get("start_time") or 0)
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    rotation = _rotation(video)
    if abs(rotation) in {90, 270}:
        width, height = height, width
    info = SourceInfo(
        duration=duration,
        start_time=start_time,
        width=width,
        height=height,
        video_stream_index=int(video["index"]),
        audio_stream_index=int(audio["index"]),
        video_codec=video.get("codec_name", ""),
        audio_codec=audio.get("codec_name", ""),
        pix_fmt=video.get("pix_fmt", ""),
        sample_aspect_ratio=video.get("sample_aspect_ratio") or "1:1",
        frame_rate=video.get("r_frame_rate", ""),
        avg_frame_rate=video.get("avg_frame_rate", ""),
        variable_frame_rate=_is_probably_vfr(video),
        color_primaries=video.get("color_primaries", ""),
        color_transfer=video.get("color_transfer", ""),
        color_matrix=video.get("color_space", ""),
        color_range=video.get("color_range", ""),
        rotation=rotation,
        raw=data,
    )
    if _is_hdr(info) and not {"zscale", "tonemap"} <= available_filters(pair.ffmpeg):
        raise ReadyVideoError(
            "UNSUPPORTED_HDR",
            "HDR input requires an FFmpeg build with zscale and tonemap before transcription can start.",
            "Use the managed FFmpeg runtime or install FFmpeg with zimg/libzimg support.",
        )
    decode_probe(input_path, pair, info)
    return info


def _choose_stream(streams: list[dict], codec_type: str) -> dict | None:
    candidates = [s for s in streams if s.get("codec_type") == codec_type]
    for stream in candidates:
        if stream.get("disposition", {}).get("default") == 1:
            return stream
    return candidates[0] if candidates else None


def _rotation(stream: dict) -> int:
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        try:
            return int(float(tags["rotate"])) % 360
        except ValueError:
            return 0
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            try:
                return int(float(side_data["rotation"])) % 360
            except ValueError:
                return 0
    return 0


def _is_probably_vfr(video: dict) -> bool:
    avg = video.get("avg_frame_rate")
    nominal = video.get("r_frame_rate")
    if not avg or not nominal or avg in {"0/0", "N/A"} or nominal in {"0/0", "N/A"}:
        return False
    return avg != nominal


def _is_hdr(info: SourceInfo) -> bool:
    transfer = info.color_transfer.lower()
    primaries = info.color_primaries.lower()
    pix_fmt = info.pix_fmt.lower()
    return transfer in {"smpte2084", "arib-std-b67"} or primaries == "bt2020" or "p10" in pix_fmt or "p12" in pix_fmt


def decode_probe(input_path: Path, pair: BinaryPair, source: SourceInfo) -> None:
    positions = [max(0.0, source.start_time), max(0.0, source.start_time + source.duration - 0.5)]
    seen: set[float] = set()
    for position in positions:
        rounded = round(position, 3)
        if rounded in seen:
            continue
        seen.add(rounded)
        completed = run(
            [
                pair.ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-ss",
                f"{position:.3f}",
                "-i",
                input_path,
                "-map",
                f"0:{source.video_stream_index}",
                "-frames:v",
                "2",
                "-f",
                "null",
                "-",
            ],
            timeout=30,
        )
        if completed.returncode != 0:
            raise ReadyVideoError(
                "INVALID_MEDIA",
                "Input failed a short decode probe.",
                "Try re-exporting the clip or transcoding it to H.264/AAC MP4.",
                details=completed.stderr,
            )


SILENCE_START = re.compile(r"silence_start: (?P<t>[0-9.]+)")
SILENCE_END = re.compile(r"silence_end: (?P<t>[0-9.]+)")
SILENCE_DURATION = re.compile(r"silence_duration: (?P<t>[0-9.]+)")


def parse_silencedetect(output: str) -> list[dict[str, float]]:
    ranges: list[dict[str, float]] = []
    pending: float | None = None
    for line in output.splitlines():
        if match := SILENCE_START.search(line):
            pending = float(match.group("t"))
        if match := SILENCE_END.search(line):
            start = pending if pending is not None else 0.0
            end = float(match.group("t"))
            duration_match = SILENCE_DURATION.search(line)
            ranges.append({"start": start, "end": end, "duration": float(duration_match.group("t")) if duration_match else end - start})
            pending = None
    return ranges


def analyze_silence(input_path: Path, pair: BinaryPair, config: ResolvedConfig, source: SourceInfo) -> Timeline:
    if not config.silence.enabled:
        return timeline_from_silences(source.duration, [], SilenceParams(enabled=False))
    completed = run(
        [
            pair.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-i",
            input_path,
            "-map",
            f"0:{source.audio_stream_index}",
            "-af",
            f"silencedetect=noise={config.silence.threshold_db}dB:d={config.silence.min_silence_s}",
            "-f",
            "null",
            "-",
        ],
        timeout=max(60, int(source.duration * 4)),
    )
    if completed.returncode != 0:
        raise ReadyVideoError("INVALID_MEDIA", "FFmpeg silence analysis failed.", details=completed.stderr)
    silent_ranges: list[tuple[float, float]] = []
    pending: float | None = None
    for line in (completed.stderr or "").splitlines():
        if match := SILENCE_START.search(line):
            pending = float(match.group("t"))
        if match := SILENCE_END.search(line):
            start = pending if pending is not None else 0.0
            silent_ranges.append((start, float(match.group("t"))))
            pending = None
    if pending is not None:
        silent_ranges.append((pending, source.duration))
    params = SilenceParams(
        enabled=True,
        threshold_db=config.silence.threshold_db,
        min_silence_s=config.silence.min_silence_s,
        margin_before_s=config.silence.margin_before_s,
        margin_after_s=config.silence.margin_after_s,
    )
    return timeline_from_silences(source.duration, silent_ranges, params)


def build_speech_wav(input_path: Path, output_path: Path, pair: BinaryPair, source: SourceInfo, timeline: Timeline) -> None:
    filters = []
    labels = []
    for segment in timeline.segments:
        label = f"a{segment.i}"
        filters.append(
            f"[0:{source.audio_stream_index}]atrim=start={segment.source_start}:end={segment.source_end},asetpts=PTS-STARTPTS[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[aout]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    completed = run(
        [
            pair.ffmpeg,
            "-y",
            "-hide_banner",
            "-nostdin",
            "-i",
            input_path,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[aout]",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            tmp,
        ],
        timeout=max(60, int(timeline.source_duration * 4)),
    )
    if completed.returncode != 0:
        raise ReadyVideoError("INVALID_MEDIA", "Could not build speech.wav.", details=completed.stderr)
    duration = media_duration(tmp, pair)
    if abs(duration - timeline.edited_duration) > 0.020:
        tmp.unlink(missing_ok=True)
        raise ReadyVideoError(
            "TIMELINE_MAPPING_FAILED",
            "speech.wav duration does not match the edited timeline.",
            details=f"speech={duration:.3f} timeline={timeline.edited_duration:.3f}",
        )
    tmp.replace(output_path)


def media_duration(path: Path, pair: BinaryPair) -> float:
    data = ffprobe_json(pair, ["-show_entries", "format=duration", path])
    return float(data.get("format", {}).get("duration") or 0)


def write_source(path: Path, source: SourceInfo) -> None:
    atomic_write_json(path, source)


def read_timeline(path: Path) -> Timeline:
    return Timeline.model_validate(json.loads(path.read_text()))
